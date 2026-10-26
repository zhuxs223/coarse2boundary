"""Three independent parcel models, paired training and val-only selection.

Order: fine parcel -> pseudo-cut parcel -> coarse mask -> final test of all bests.
The upstream teacher is transductive. Downstream splits are spatially disjoint.
"""
import argparse
from contextlib import contextmanager
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import segmentation_models_pytorch as smp

HERE = Path(__file__).resolve().parent
ARMS = ('fine_parcel', 'pseudo_parcel', 'coarse_mask')
DOMAINS = ('parcel_global', 'parcel_boundary_band', 'boundary_inside_coarse')
MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]


class ValidationInterruption(RuntimeError):
    """Failure-injection sentinel used only by the small E2E validation."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def atomic_csv(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


@contextmanager
def run_lock(root):
    # OS locks release on process death, unlike stale PID marker files.
    with (root / 'run.lock').open('a+b') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(mode='bilinear'):
    return dict(architecture='smp.Unet', encoder='resnet50', decoder_interpolation=mode,
                in_channels=3, classes=1, smp_version=str(smp.__version__))


def model(weights=None, mode='bilinear'):
    if smp.__version__ != '0.5.0':
        raise RuntimeError('This package requires segmentation-models-pytorch==0.5.0')
    net = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1)
    for block in net.decoder.blocks:
        block.interpolation_mode = mode
    if weights:
        state = torch.load(weights, map_location='cpu', weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith('fc.')}
        net.encoder.load_state_dict(state, strict=True)
    return net


def teacher(path, expected_mode):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    saved = ck.get('model_config', {})
    if saved.get('architecture') != 'smp.Unet' or saved.get('encoder') != 'resnet50':
        raise ValueError('Expected a ResNet50 SMP U-Net teacher with model_config metadata')
    if saved.get('decoder_interpolation') != expected_mode:
        raise ValueError('Teacher decoder mode does not match configuration')
    net = model(mode=expected_mode)
    # Upstream experiment wraps the SMP network under .net.
    state = ck['model']
    if all(k.startswith('net.') for k in state):
        state = {k[4:]: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return net, dict(epoch=ck.get('epoch'), round=ck.get('round'), model_config=saved)


def state_sha(net):
    h = hashlib.sha256()
    for k, value in net.state_dict().items():
        h.update(k.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_rgb(root, key):
    with Image.open(Path(root) / 'images/train' / (key + '.png')) as im:
        a = np.asarray(im.convert('RGB')).copy()
    if a.shape != (512, 512, 3):
        raise ValueError('Expected 512x512 RGB: ' + key)
    return a


def binary(path):
    with Image.open(path) as im:
        a = np.asarray(im).copy()
    if a.shape != (512, 512) or not np.isin(a, [0, 1, 255]).all():
        raise ValueError('Expected a binary 512x512 label: ' + str(path))
    return a > 0


def layer(root, name, key):
    return binary(Path(root) / name / 'train' / (key + '.png'))


def parcel_source_layers(cfg, key):
    """Existing parcel file, or its explicit coarse-minus-fine-boundary definition."""
    p = Path(cfg['data_root']) / 'parcel_labels/train' / (key + '.png')
    return ('parcel_labels',) if p.is_file() else ('boundary_labels',)


def fine_parcel_target(cfg, key, coarse, bnd=None):
    if parcel_source_layers(cfg, key) == ('parcel_labels',):
        return layer(cfg['data_root'], 'parcel_labels', key)
    if bnd is None:
        bnd = layer(cfg['data_root'], 'boundary_labels', key)
    return coarse & ~bnd


def valid_mask(rgb, coarse):
    valid = ~((rgb.max(axis=2) == 0) & ~coarse)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    return valid


def normalized(rgb):
    return np.ascontiguousarray((rgb.transpose(2, 0, 1).astype(np.float32)/255 - MEAN)/STD)


class Tiles(Dataset):
    def __init__(self, cfg, keys, arm=None, evaluation=False):
        self.cfg, self.keys, self.arm, self.evaluation = cfg, list(keys), arm, evaluation

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        epoch = 0
        if isinstance(index, tuple):
            index, epoch = index
        key = self.keys[index]
        rgb = load_rgb(self.cfg['data_root'], key)
        coarse = layer(self.cfg['data_root'], 'mask_labels', key)
        valid = valid_mask(rgb, coarse)
        if self.evaluation:
            # Fine labels never enter pseudo/coarse training datasets.
            bnd = layer(self.cfg['data_root'], 'boundary_labels', key)
            target = fine_parcel_target(self.cfg, key, coarse, bnd)
            band = ndi.distance_transform_edt(~bnd) <= self.cfg['boundary_band_radius'] if bnd.any() else np.zeros_like(bnd)
            return dict(img=torch.from_numpy(normalized(rgb)), target=torch.from_numpy(target),
                        coarse=torch.from_numpy(coarse), valid=torch.from_numpy(valid),
                        boundary=torch.from_numpy(bnd), band=torch.from_numpy(band & valid), idx=index)
        if self.arm == 'fine_parcel':
            target = fine_parcel_target(self.cfg, key, coarse)
        elif self.arm == 'pseudo_parcel':
            target = binary(Path(self.cfg['output_root']) / 'pseudo_labels' / (key + '.png'))
        elif self.arm == 'coarse_mask':
            target = coarse.copy()
        elif self.arm is None:
            return dict(img=torch.from_numpy(normalized(rgb)), coarse=torch.from_numpy(coarse), idx=index)
        else:
            raise ValueError(self.arm)
        if self.cfg['augmentation'] == 'd4':
            rng = np.random.default_rng(self.cfg['seed'] + epoch*1000003 + index)
            turns, flip = int(rng.integers(4)), bool(rng.integers(2))
            rgb, target, valid = [np.rot90(a, turns) for a in (rgb, target, valid)]
            if flip:
                rgb, target, valid = [np.flip(a, axis=1) for a in (rgb, target, valid)]
        return dict(img=torch.from_numpy(normalized(rgb)), target=torch.from_numpy(np.ascontiguousarray(target)),
                    valid=torch.from_numpy(np.ascontiguousarray(valid)), idx=index)


class EpochSampler(Sampler):
    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 1

    def __iter__(self):
        ids = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        return iter((i, self.epoch) for i in ids)

    def __len__(self):
        return self.n


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def loader(ds, cfg, training=False):
    sampler = EpochSampler(len(ds), cfg['seed']) if training else None
    return DataLoader(ds, batch_size=cfg['batch_size'] if training else cfg['infer_batch_size'],
                      sampler=sampler, num_workers=cfg['workers'], pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg['workers'] > 0, worker_init_fn=worker_init,
                      generator=torch.Generator().manual_seed(cfg['seed']), drop_last=False)


def close_loader(dl):
    iterator = getattr(dl, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()


def loss(logits, target, valid):
    y, v = target.float(), valid.float()
    count = v.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction='none')*v).sum()/count
    p = torch.sigmoid(logits.float())*v
    dice = 1 - (2*(p*y).sum()+1)/(p.sum()+(y*v).sum()+1)
    return bce+dice, bce, dice


def cell(pred, truth, domain):
    p, g = pred & domain, truth & domain
    tp = int((p & g).sum())
    return np.array([tp, int(p.sum())-tp, int(g.sum())-tp, int(domain.sum())], dtype=np.int64)


def ratio(a, b):
    return float(a/b) if b else None


def metric_rows(net, dl, device, cfg, split, epoch):
    net.eval()
    counts = {(g, d): np.zeros(4, np.int64) for g in ('ALL', 'CC', 'SW') for d in DOMAINS}
    n_images = dict(ALL=0, CC=0, SW=0)
    with torch.inference_mode():
        for batch in dl:
            prob = torch.sigmoid(net(batch['img'].to(device))[:, 0].float()).cpu().numpy()
            for j, index in enumerate(batch['idx'].tolist()):
                key = dl.dataset.keys[index]
                region = 'CC' if key.startswith('Central China/') else 'SW'
                pred = prob[j] >= cfg['prediction_threshold']
                valid, target, coarse, bnd, band = [batch[k][j].numpy() for k in ('valid','target','coarse','boundary','band')]
                values = {
                    'parcel_global': cell(pred, target, valid),
                    'parcel_boundary_band': cell(pred, target, band),
                    # Evaluate the complementary boundary throughout cropland,
                    # including false cuts away from GT; never only on GT positives.
                    'boundary_inside_coarse': cell(coarse & ~pred, bnd, valid & coarse)}
                for group in ('ALL', region):
                    n_images[group] += 1
                    for domain, c in values.items():
                        counts[group, domain] += c
    rows = []
    for (group, domain), (tp, fp, fn, area) in counts.items():
        rows.append(dict(epoch=epoch, split=split, group=group, domain=domain,
                         n_images=n_images[group], TP=int(tp), FP=int(fp), FN=int(fn), domain_px=int(area),
                         P=ratio(tp,tp+fp), R=ratio(tp,tp+fn), IoU=ratio(tp,tp+fp+fn),
                         boundary_hit_rate=ratio(tp,tp+fn) if domain=='boundary_inside_coarse' else None,
                         boundary_missed_as_parcel_rate=ratio(fn,tp+fn) if domain=='boundary_inside_coarse' else None))
    return rows


def pick(rows, domain, group='ALL'):
    return next(r for r in rows if r['domain'] == domain and r['group'] == group)


def audit_split(split):
    names = ('train_sd', 'val_sd', 'test_sd', 'dropped_buffer')
    sets = {n:set(split[n]) for n in names}
    if any(len(sets[n]) != len(split[n]) for n in names):
        raise ValueError('Duplicate split keys')
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if sets[a] & sets[b]:
                raise ValueError('Split key overlap')
    grouped = {}
    for n in names[:3]:
        grouped[n] = {}
        for key in split[n]:
            source, x, y = key.rsplit('_',2)
            grouped[n].setdefault(source, []).append((int(x),int(y)))
    overlaps = {}
    for a,b in (('train_sd','val_sd'),('train_sd','test_sd'),('val_sd','test_sd')):
        total = 0
        for source in set(grouped[a]) & set(grouped[b]):
            aa,bb = np.array(grouped[a][source]),np.array(grouped[b][source])
            for block in range(0,len(aa),256):
                d = np.abs(aa[block:block+256,None,:]-bb[None,:,:])
                total += int(((d[:,:,0]<512)&(d[:,:,1]<512)).sum())
        overlaps[a+'__'+b] = total
        if total:
            raise ValueError('Spatial pixel overlap across splits: '+str(overlaps))
    return dict(sizes={n:len(split[n]) for n in names}, overlapping_pairs=overlaps)


def preflight(cfg, split, device):
    info = audit_split(split)
    missing, stamps = [], []
    parcel_sources = {}
    for name in ('train_sd','val_sd','test_sd'):
        parcel_sources[name] = dict(existing_parcel=0, coarse_minus_boundary=0)
        for key in split[name]:
            source = parcel_source_layers(cfg, key)
            parcel_sources[name]['existing_parcel' if source==('parcel_labels',) else 'coarse_minus_boundary'] += 1
            layers = tuple(dict.fromkeys(('images','mask_labels') + source +
                           (() if name=='train_sd' else ('boundary_labels',))))
            for item in layers:
                p = Path(cfg['data_root']) / item / 'train' / (key+'.png')
                if not p.is_file():
                    missing.append(str(p))
                else:
                    stat=p.stat()
                    stamps.append((name,key,item,stat.st_size,stat.st_mtime_ns))
    if missing:
        raise FileNotFoundError('Missing dataset files under data_root='+str(cfg['data_root'])+
                                ': '+str(missing[:5])+'. If parcel_labels is absent, boundary_labels is required '
                                'to derive fine parcel = mask_labels & ~boundary_labels. Check config.json data_root.')
    for k in ('resnet_weights','pseudo_checkpoint'):
        if not Path(cfg[k]).is_file():
            raise FileNotFoundError(k+': '+cfg[k])
    net, metadata = teacher(cfg['pseudo_checkpoint'], cfg['pseudo_decoder_interpolation'])
    net.to(device).eval()
    ds = Tiles(cfg, split['train_sd'][:2])
    with torch.inference_mode():
        x = torch.stack([ds[i]['img'] for i in range(len(ds))]).to(device)
        y = net(x)
        if y.shape != (len(ds),1,512,512) or not torch.isfinite(y).all():
            raise ValueError('Invalid teacher output')
    del net, x, y
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    code = {p.name:sha(p) for p in HERE.glob('*.py') if p.name != 'smoke_test.py'}
    info.update(teacher=metadata, data_fingerprint=digest(stamps), parcel_target_sources=parcel_sources,
                teacher_sha256=sha(cfg['pseudo_checkpoint']), resnet_sha256=sha(cfg['resnet_weights']),
                split_sha256=sha(cfg['split_file']), code=code, model_config=model_config(),
                upstream_protocol='teacher saw full target collection; downstream holds out spatial stripes')
    scientific = {k:v for k,v in cfg.items() if k not in ('workers','preview_every')}
    info['signature'] = digest(dict(config=scientific, provenance=info))
    return info


def prepare_pseudo(cfg, keys, info, device):
    root = Path(cfg['output_root']) / 'pseudo_labels'
    root.mkdir(exist_ok=True)
    manifest = root / 'manifest.json'
    expected = dict(signature=info['signature'], keys_sha256=digest(keys), teacher_sha256=info['teacher_sha256'],
                    threshold=cfg['pseudo_boundary_threshold'], target='coarse_mask & ~(sigmoid(teacher)>=threshold)',
                    n_images=len(keys), fine_labels_used=False)
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if {k:previous[k] for k in expected} != expected:
            raise RuntimeError('Pseudo-label cache provenance changed')
        if previous.get('complete'):
            actual={k:sha(root/(k+'.png')) for k in keys}
            if previous.get('label_sha256') != actual:
                raise RuntimeError('Completed pseudo-label cache changed')
    else:
        atomic_json(manifest,dict(expected,complete=False))
    missing = [k for k in keys if not (root/(k+'.png')).exists()]
    if missing:
        print(f'[pseudo] generating {len(missing)}/{len(keys)} TRAIN targets only',flush=True)
        net,_ = teacher(cfg['pseudo_checkpoint'],cfg['pseudo_decoder_interpolation'])
        net.to(device).eval()
        dl = loader(Tiles(cfg,missing),cfg)
        try:
            with torch.inference_mode():
                for step,batch in enumerate(dl,1):
                    q = torch.sigmoid(net(batch['img'].to(device))[:,0].float()).cpu().numpy()
                    for j,index in enumerate(batch['idx'].tolist()):
                        key=missing[index]
                        target=batch['coarse'][j].numpy() & (q[j] < cfg['pseudo_boundary_threshold'])
                        path=root/(key+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                        tmp=path.with_name(path.name+'.tmp')
                        Image.fromarray(target.astype(np.uint8)).save(tmp,format='PNG')
                        tmp.replace(path)
                    if step % 50==0:print(f'[pseudo] {min(step*cfg["infer_batch_size"],len(missing))}/{len(missing)}',flush=True)
        finally:
            close_loader(dl)
            del net,dl
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    atomic_json(manifest,dict(expected,complete=True,label_sha256={k:sha(root/(k+'.png')) for k in keys}))


def cpu_state(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def previews(net, cfg, keys, device, directory):
    directory.mkdir(parents=True,exist_ok=True)
    net.eval()
    ds=Tiles(cfg,keys,evaluation=True)
    with torch.inference_mode():
        for i,key in enumerate(keys):
            sample=ds[i]
            p=torch.sigmoid(net(sample['img'][None].to(device))[:,0].float())[0].cpu().numpy()
            stem=key.replace('/','__')
            Image.fromarray(np.rint(p*255).astype(np.uint8)).save(directory/(stem+'_parcel_probability.png'))
            Image.fromarray(((p>=cfg['prediction_threshold'])*255).astype(np.uint8)).save(directory/(stem+'_parcel_binary.png'))
            boundary=sample['coarse'].numpy() & (p<cfg['prediction_threshold'])
            Image.fromarray((boundary*255).astype(np.uint8)).save(directory/(stem+'_boundary_binary.png'))
            Image.fromarray((sample['target'].numpy()*255).astype(np.uint8)).save(directory/(stem+'_parcel_GT.png'))


def log_history(root,history):
    atomic_csv(root/'val_metrics.csv',[r for h in history for r in h['metrics']])
    atomic_csv(root/'train.csv',[{k:v for k,v in h.items() if k!='metrics'} for h in history])


def train_arm(arm,cfg,split,info,device,_interrupt_after_commit=None):
    root=Path(cfg['output_root'])/arm
    root.mkdir(exist_ok=True)
    seed_all(cfg['seed'])
    net=model(cfg['resnet_weights'])
    initial=state_sha(net)
    common=Path(cfg['output_root'])/'initialization.json'
    if common.exists():
        if json.loads(common.read_text())['state_sha256']!=initial:raise RuntimeError('Unpaired initialization')
    else:atomic_json(common,dict(state_sha256=initial,parameters=sum(p.numel() for p in net.parameters())))
    net.to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda')
    history,best,epoch=[],None,0
    last=root/'last.pth'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=True)
        if ck['signature']!=info['signature'] or ck['arm']!=arm or ck['initial_state_sha256']!=initial:
            raise RuntimeError('Training resume configuration changed')
        net.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        history,best,epoch=ck['history'],ck['best'],ck['epoch']
        atomic_torch(root/'best.pth',best)
        log_history(root,history)
        del ck
    if epoch<cfg['epochs']:
        if arm=='pseudo_parcel':prepare_pseudo(cfg,split['train_sd'],info,device)
        dl=loader(Tiles(cfg,split['train_sd'],arm=arm),cfg,training=True)
        vl=loader(Tiles(cfg,split['val_sd'],evaluation=True),cfg)
        probe_keys=[next(k for k in split['val_sd'] if k.startswith(region)) for region in ('Central China/','Southwest China/')]
        try:
            for epoch in range(epoch+1,cfg['epochs']+1):
                seed_all(cfg['seed']+epoch)
                dl.sampler.epoch=epoch
                net.train();start=time.time();sums=np.zeros(3,np.float64);steps=0
                for batch in dl:
                    x=batch['img'].to(device);y=batch['target'].to(device);v=batch['valid'].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,enabled=cfg['amp'] and device.type=='cuda'):
                        value,bce,dice=loss(net(x)[:,0],y,v)
                    if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss')
                    scaler.scale(value).backward();scaler.step(optimizer);scaler.update()
                    sums+=np.array([value.item(),bce.item(),dice.item()]);steps+=1
                metrics=metric_rows(net,vl,device,cfg,'val',epoch)
                selected=pick(metrics,cfg['selection_domain'])
                score=selected['IoU']
                if score is None:raise RuntimeError('Selection IoU is undefined; check boundary-band GT')
                improved=best is None or score>best['selected_val_IoU']
                if improved:
                    best=dict(model=cpu_state(net),model_config=model_config(),epoch=epoch,arm=arm,
                              signature=info['signature'],initial_state_sha256=initial,
                              selection_domain=cfg['selection_domain'],selected_val_IoU=score,
                              selected_val_metrics=metrics)
                history.append(dict(epoch=epoch,loss=float(sums[0]/steps),bce=float(sums[1]/steps),
                                    dice_loss=float(sums[2]/steps),steps=steps,seconds=time.time()-start,
                                    val_selection_IoU=score,best_epoch=best['epoch'],metrics=metrics))
                # Single source of truth for resume; best state is embedded so an
                # interruption between last/best/log commits is recoverable.
                ck=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                        epoch=epoch,arm=arm,signature=info['signature'],initial_state_sha256=initial,
                        best=best,history=history)
                atomic_torch(last,ck)
                if epoch==_interrupt_after_commit:
                    raise ValidationInterruption('Injected after last.pth commit, before best/log commit')
                if improved:atomic_torch(root/'best.pth',best)
                log_history(root,history)
                global_metric=pick(metrics,'parcel_global')
                line_metric=pick(metrics,'boundary_inside_coarse')
                print(f'[{arm} ep{epoch:03d}/{cfg["epochs"]}] loss={sums[0]/steps:.4f} '
                      f'val parcel IoU={global_metric["IoU"]:.4f} band IoU={pick(metrics,"parcel_boundary_band")["IoU"]:.4f} '
                      f'boundary IoU={line_metric["IoU"]:.4f} best=ep{best["epoch"]} '
                      f'({best["selected_val_IoU"]:.4f}) boundary hit={line_metric["boundary_hit_rate"]:.4f} '
                      f'missed-as-parcel={line_metric["boundary_missed_as_parcel_rate"]:.4f}',flush=True)
                if cfg['preview_every'] and (epoch%cfg['preview_every']==0 or epoch==cfg['epochs']):
                    previews(net,cfg,probe_keys,device,root/'probes'/f'val_ep{epoch:03d}')
        finally:
            close_loader(dl);close_loader(vl)
            del dl,vl
    atomic_json(root/'trained.json',dict(arm=arm,epochs=epoch,best_epoch=best['epoch'],
                selected_val_IoU=best['selected_val_IoU'],selection_domain=cfg['selection_domain'],
                signature=info['signature'],best_sha256=sha(root/'best.pth')))
    del net,optimizer,scaler,best
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()


def test_arm(arm,cfg,split,info,device):
    root=Path(cfg['output_root'])/arm
    best_path=root/'best.pth'
    best_sha=sha(best_path)
    result=root/'test_metrics.json'
    if result.exists():
        saved=json.loads(result.read_text(encoding='utf-8'))
        if saved['signature']!=info['signature'] or saved['best_sha256']!=best_sha:
            raise RuntimeError('Completed test provenance changed')
        return saved
    best=torch.load(best_path,map_location='cpu',weights_only=True)
    net=model();net.load_state_dict(best['model'],strict=True);net.to(device)
    dl=loader(Tiles(cfg,split['test_sd'],evaluation=True),cfg)
    try:rows=metric_rows(net,dl,device,cfg,'test',best['epoch'])
    finally:close_loader(dl)
    probe_keys=[next(k for k in split['test_sd'] if k.startswith(region)) for region in ('Central China/','Southwest China/')]
    previews(net,cfg,probe_keys,device,root/'probes'/'test_best')
    saved=dict(arm=arm,best_epoch=best['epoch'],selected_val_IoU=best['selected_val_IoU'],
               selection_domain=cfg['selection_domain'],signature=info['signature'],best_sha256=best_sha,metrics=rows)
    atomic_json(result,saved);atomic_csv(root/'test_metrics.csv',rows)
    del net,dl,best
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(HERE/'config.json'))
    parser.add_argument('--precheck',action='store_true',help='No training or full pseudo-label inference')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    cp=Path(args.config).resolve();cfg=json.loads(cp.read_text(encoding='utf-8'))
    for key in ('data_root','split_file','resnet_weights','pseudo_checkpoint','output_root'):
        p=Path(cfg[key]);cfg[key]=str((cp.parent/p).resolve() if not p.is_absolute() else p.resolve())
    if cfg['selection_domain'] not in DOMAINS or cfg['boundary_band_radius']<1:
        raise ValueError('Invalid selection domain or boundary band')
    if cfg['epochs']<1 or cfg['batch_size']<1 or cfg['workers']<0 or cfg['augmentation'] not in ('d4','none'):
        raise ValueError('Invalid training configuration')
    if not 0<cfg['prediction_threshold']<1 or not 0<cfg['pseudo_boundary_threshold']<1:
        raise ValueError('Invalid threshold')
    device=torch.device(args.device)
    cv2.setNumThreads(1);torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    split=json.loads(Path(cfg['split_file']).read_text(encoding='utf-8'))
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        info=preflight(cfg,split,device)
        plan=root/'plan.json'
        if plan.exists() and json.loads(plan.read_text(encoding='utf-8'))['provenance']['signature']!=info['signature']:
            raise RuntimeError('Output belongs to a different experiment; use a new output_root')
        atomic_json(plan,dict(config=cfg,provenance=info,arms=ARMS,
                    training_loss='valid-pixel BCE + batch soft Dice, no fine-boundary-derived weights',
                    checkpoint_selection='highest pooled ALL val IoU in selection_domain; earliest exact tie',
                    test_policy='all three arms finish before any test evaluation'))
        print('[precheck]',json.dumps(dict(device=str(device),sizes=info['sizes'],
              overlapping_pairs=info['overlapping_pairs'],teacher=info['teacher'],
              parcel_target_sources=info['parcel_target_sources'],
              selection=cfg['selection_domain'],band_radius=cfg['boundary_band_radius'])),flush=True)
        if args.precheck:return
        for arm in ARMS:train_arm(arm,cfg,split,info,device)
        print('[test] all arms trained; evaluating val-selected checkpoints',flush=True)
        summary=[]
        for arm in ARMS:
            result=test_arm(arm,cfg,split,info,device)
            row=dict(arm=arm,best_epoch=result['best_epoch'],val_selection_IoU=result['selected_val_IoU'])
            for domain in DOMAINS:
                metrics=pick(result['metrics'],domain)
                for name in ('P','R','IoU'):row[domain+'_'+name]=metrics[name]
            boundary_metrics=pick(result['metrics'],'boundary_inside_coarse')
            for name in ('boundary_hit_rate','boundary_missed_as_parcel_rate'):
                row[name]=boundary_metrics[name]
            summary.append(row)
            print('[test]',json.dumps(row),flush=True)
        atomic_csv(root/'comparison_test.csv',summary)
        atomic_json(root/'complete.json',dict(signature=info['signature'],summary=summary))


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Interrupted. Rerun the same command to resume from the last committed epoch.',flush=True)
        sys.exit(130)
