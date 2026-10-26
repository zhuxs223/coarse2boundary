"""Independent S2L-Adapted runner. latest.pth atomically commits model + EMA."""
import runtime_setup
import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import cfg_standard as C
import data as D
import model_student as M
import supervision as S
import utils as U
from dataset_protocol import load_targets, smoke_targets
from reference import assess as A


def configuration():
    names = ('EPOCHS', 'WARM_EPOCHS', 'EMA_INTERVAL', 'EMA_ALPHA', 'CONFIDENCE', 'PSEUDO_WEIGHT',
             'BATCH_SIZE', 'INFER_BATCH', 'NUM_WORKERS', 'AMP', 'LR', 'WEIGHT_DECAY', 'SEED',
             'IMG_SIZE', 'EDGE_TRIM', 'EVALUATE_EVERY', 'SAVE_EVERY', 'DECODER_INTERPOLATION',
             'LAYERS', 'IMAGENET_MEAN', 'IMAGENET_STD', 'AUGMENT', 'PROBE_KEYS')
    return dict(protocol='s2l-adapted-full5227-v1', settings={n: getattr(C, n) for n in names},
                optimizer='torch.optim.RAdam (all trainable parameters)', scheduler='constant',
                ema_initialization='first unaugmented prediction after epoch 1',
                selection='fixed final epoch; GT for retrospective monitoring only',
                no_small=True, no_geometry=True, source_background='constant weight 1')


def plan(smoke=False):
    keys, _, groups, cohort = load_targets(C.SPLIT_JSON)
    if smoke:
        keys, _ = smoke_targets(groups)
    if not 0 < C.WARM_EPOCHS < C.EPOCHS or C.EMA_INTERVAL != 1:
        raise ValueError('Expected a warm stage followed by pseudo training and epoch-wise EMA')
    for key in keys:
        for layer in C.LAYERS.values():
            p = C.DATA_ROOT / layer / 'train' / (key + '.png')
            if not p.is_file():
                raise FileNotFoundError(p)
    if U.sha(C.RESNET50_W) != '8c28bce1adf71c9dbb928e91724832100e67dc6ff68af1a3a79f53cc39833bbf':
        raise ValueError('ImageNet ResNet50 weight SHA256 differs')
    code = {p.relative_to(C.HERE).as_posix(): U.sha(p) for p in sorted(C.HERE.rglob('*.py'))
            if not p.name.startswith('test_') and '__pycache__' not in p.parts}
    signature = json.loads(json.dumps(dict(config=configuration(), keys=keys, cohort=cohort, smoke=smoke,
        split_sha256=U.sha(C.SPLIT_JSON), resnet_sha256=U.sha(C.RESNET50_W), code=code,
        data_root=str(C.DATA_ROOT.resolve()))))
    root = C.output()
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))['signature'] != signature:
        raise RuntimeError('Scientific code/config/inputs changed; use an independent output directory')
    U.json_write(path, dict(signature=signature))
    return keys, signature


def precheck(keys, signature):
    U.seed(C.SEED)
    net = M.build_model(str(C.RESNET50_W), verbose=False)
    config = M.model_config(net)
    if config['decoder_interpolation'] != 'bilinear' or any(isinstance(m, torch.nn.ConvTranspose2d) for m in net.modules()):
        raise RuntimeError('Student decoder differs from common baseline')
    item = D.Tiles(keys[:1])[0]
    if 'bnd' in item or 'ema' in item:
        raise RuntimeError('Weak-only data unexpectedly read fine/pseudo labels')
    n = len(keys)
    cache = n * C.IMG_SIZE**2 * 4
    info = dict(train=n, evaluation=n, overlap=n, fine_gt_use='evaluation only', small_teacher=False,
        warm_epochs=C.WARM_EPOCHS, pseudo_epochs=C.EPOCHS-C.WARM_EPOCHS, epochs=C.EPOCHS,
        batch=C.BATCH_SIZE, optimizer_steps_per_epoch=math.ceil(n/C.BATCH_SIZE),
        optimizer_steps_total=math.ceil(n/C.BATCH_SIZE)*C.EPOCHS,
        ema=dict(alpha=C.EMA_ALPHA, interval=C.EMA_INTERVAL, dtype='float32',
                 initialize='first prediction', cache_GiB=cache/1024**3, peak_generations=2),
        confidence=C.CONFIDENCE, pseudo_weight=C.PSEUDO_WEIGHT,
        optimizer='RAdam', lr=C.LR, weight_decay=C.WEIGHT_DECAY, scheduler='constant',
        train_workers=C.NUM_WORKERS, infer_batch=C.INFER_BATCH,
        student_parameters=sum(p.numel() for p in net.parameters()), model_config=config,
        output=str(C.output()), cuda=torch.cuda.is_available(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        runtime=runtime_setup.SETTINGS, estimated_output_GiB=2*cache/1024**3+6)
    U.json_write(C.output()/'precheck.json', info)
    print('[precheck]', json.dumps(info), flush=True)
    del net, item
    gc.collect()
    return info


def train_epoch(net, opt, scaler, keys, ema_path, epoch, device):
    U.seed(C.SEED + epoch)
    enabled = epoch > C.WARM_EPOCHS
    ds = D.Tiles(keys, epoch=epoch, training=True, ema_path=ema_path if enabled else None)
    net.train()
    start = time.perf_counter()
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    steps = 0
    updates = 0
    for batch in D.loader(ds, shuffle=True):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=C.AMP and device.type=='cuda'):
            logits = net(batch['img'])[:, 0]
            loss, ls, lp, pos, neg = S.losses(logits, batch, enabled)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        updates += int(scaler.get_scale() >= scale_before)
        totals += torch.stack((loss.detach(), ls.detach(), lp.detach(), pos.sum(), neg.sum(),
                              (batch['valid'] & (batch['mk3']==1)).sum())).double()
        steps += 1
        if steps % 100 == 0:
            print(f'[train ep{epoch:03d}] {steps}/{math.ceil(len(keys)/C.BATCH_SIZE)} {time.perf_counter()-start:.1f}s', flush=True)
    values = totals.cpu().tolist()
    del ds
    gc.collect()  # Close parent/worker EMA mmap handles before deleting old generations on Windows.
    return dict(epoch=epoch, phase='pseudo' if enabled else 'warm', steps=steps,
        optimizer_updates=updates, amp_skipped_steps=steps-updates,
        loss=values[0]/steps, source_loss=values[1]/steps, pseudo_loss=values[2]/steps,
        trained_pseudo_positive=int(values[3]), trained_pseudo_negative=int(values[4]),
        trained_internal=int(values[5]), train_seconds=time.perf_counter()-start)


def cache_path(epoch):
    return C.output()/'ema'/f'ema_ep{epoch:03d}.npy'


def cleanup_ema(protected):
    root = (C.output()/'ema').resolve()
    keep = Path(protected).resolve() if protected else None
    for p in root.glob('ema_ep*.npy'):
        if p.resolve() == keep:
            continue
        if not re.fullmatch(r'ema_ep\d{3}\.npy', p.name) or p.is_symlink() or p.resolve().parent != root:
            raise RuntimeError('Unsafe EMA cleanup path')
        p.unlink()


def dump_probe(key, epoch, raw, ema, valid, mk3):
    path = C.output()/'timeline'/f'epoch_{epoch:03d}'/key
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path)+'_raw.npy', raw, allow_pickle=False)
    np.save(str(path)+'_ema.npy', ema, allow_pickle=False)
    for suffix, image in (('raw', np.rint(raw*255).clip(0,255).astype(np.uint8)),
                          ('ema', np.rint(ema*255).clip(0,255).astype(np.uint8))):
        if not cv2.imwrite(str(path)+'_'+suffix+'.png', image):
            raise OSError(path)
    internal = valid & (mk3==1)
    pos, neg = S.labels(ema, internal)
    # Invalid: black; fixed source P/N: purple/cyan; U ignore: gray; pseudo P/N: blue/orange.
    panel = np.zeros((*ema.shape,3), np.uint8)
    panel[valid & (mk3==0)] = (86,180,233)
    panel[valid & (mk3==2)] = (204,121,167)
    panel[internal] = (150,150,150)
    panel[pos] = (0,114,178)
    panel[neg] = (230,159,0)
    if not cv2.imwrite(str(path)+'_labels.png', cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise OSError(path)


def ensemble(net, inference_loader, keys, old_path, epoch, device):
    """Write a new generation; leave old committed generation untouched until checkpoint commit."""
    start = time.perf_counter()
    target = cache_path(epoch)
    target.parent.mkdir(parents=True, exist_ok=True)
    required = len(keys)*C.IMG_SIZE**2*4 + 600*1024**2
    if shutil.disk_usage(C.output()).free < required + C.MIN_FREE_GIB*1024**3:
        raise RuntimeError('Not enough disk space for next EMA + checkpoint and free-space reserve')
    shape = (len(keys), C.IMG_SIZE, C.IMG_SIZE)
    old = np.load(old_path, mmap_mode='r', allow_pickle=False) if old_path else None
    current = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32, shape=shape)
    hist = np.zeros(10, np.int64)
    stats = dict(internal_px=0, ema_positive=0, ema_negative=0, ema_ignore=0,
                 raw_positive=0, raw_negative=0, positive_retained=0, positive_to_negative=0,
                 negative_to_positive=0, positive_entered=0, negative_entered=0,
                 previous_positive=0, previous_negative=0, ema_sum=0., raw_sum=0.)
    seen = np.zeros(len(keys), bool)
    probes = set(C.PROBE_KEYS)
    net.eval()
    with torch.inference_mode():
        for batch in inference_loader:
            # FP32 inference intentionally avoids mixed-precision probability artifacts at confidence thresholds.
            p = torch.sigmoid(net(batch['img'].to(device, non_blocking=True)).float())[:,0].cpu().numpy()
            for j, idx in enumerate(batch['idx'].tolist()):
                if seen[idx]:
                    raise RuntimeError('Duplicate EMA sample')
                seen[idx] = True
                raw = p[j]
                if not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):
                    raise RuntimeError('Invalid prediction probability')
                prev = np.array(old[idx], copy=True) if old is not None else None
                value = raw if prev is None else C.EMA_ALPHA*raw+(1-C.EMA_ALPHA)*prev
                current[idx] = value
                valid, mk3 = batch['valid'][j].numpy(), batch['mk3'][j].numpy()
                inner = valid & (mk3==1)
                pos, neg = S.labels(value, inner)
                stats['internal_px'] += int(inner.sum())
                stats['ema_positive'] += int(pos.sum())
                stats['ema_negative'] += int(neg.sum())
                stats['ema_ignore'] += int((inner & ~(pos|neg)).sum())
                rp, rn = S.labels(raw, inner)
                stats['raw_positive'] += int(rp.sum())
                stats['raw_negative'] += int(rn.sum())
                stats['ema_sum'] += float(value[inner].sum(dtype=np.float64))
                stats['raw_sum'] += float(raw[inner].sum(dtype=np.float64))
                hist += np.histogram(value[inner], bins=np.linspace(0,1,11))[0]
                if prev is not None:
                    pp, pn = S.labels(prev, inner)
                    for name, mask in (('positive_retained',pp&pos), ('positive_to_negative',pp&neg),
                                       ('negative_to_positive',pn&pos), ('positive_entered',pos&~pp),
                                       ('negative_entered',neg&~pn), ('previous_positive',pp), ('previous_negative',pn)):
                        stats[name] += int(mask.sum())
                if keys[idx] in probes:
                    dump_probe(keys[idx], epoch, raw, value, valid, mk3)
            done = int(seen.sum())
            if done % 256 == 0 or done == len(keys):
                print(f'[EMA ep{epoch:03d}] {done}/{len(keys)} {time.perf_counter()-start:.1f}s', flush=True)
    if not seen.all():
        raise RuntimeError('EMA missing samples')
    current.flush()
    del current, old
    gc.collect()
    with target.open('r+b') as f:
        import os
        os.fsync(f.fileno())
    for i, n in enumerate(hist):
        stats[f'ema_bin_{i}'] = int(n)
    if stats['internal_px'] != stats['ema_positive']+stats['ema_negative']+stats['ema_ignore']:
        raise RuntimeError('Pseudo-label partition mismatch')
    stats['ema_seconds'] = time.perf_counter()-start
    meta = dict(epoch=epoch, file=target.relative_to(C.output()).as_posix(), bytes=target.stat().st_size,
                shape=list(shape), dtype='float32', sha256=U.sha(target))
    return meta, stats


def evaluate(net, keys, ema_path, epoch, device):
    start = time.perf_counter()
    ds = D.Tiles(keys, ema_path=ema_path, evaluation=True)
    meters = {(g,d):A.Meter() for g in A.GROUPS for d in A.DOMAINS}
    counts = dict(epoch=epoch, U_pixels=0, U_gt=0, pseudo_positive=0, positive_gt=0,
                  positive_near3=0, pseudo_negative=0, negative_gt=0, ignore_gt=0)
    net.eval()
    with torch.inference_mode():
        for batch in D.loader(ds, inference=True):
            p = torch.sigmoid(net(batch['img'].to(device, non_blocking=True)).float())[:,0].cpu().numpy()
            for j, idx in enumerate(batch['idx'].tolist()):
                gt, mk3, valid = batch['bnd'][j].numpy(), batch['mk3'][j].numpy(), batch['valid'][j].numpy()
                doms = A.build_domains(C.IMG_SIZE,C.IMG_SIZE,batch['msk'][j].numpy(),gt,
                                       batch['nod'][j].numpy(),mk3,C.EDGE_TRIM)
                region = {'Central China':'CC','Southwest China':'SW'}[keys[idx].split('/')[0]]
                A.evaluate_batch(meters,p[j],gt,doms,.5,region)
                internal = valid & (mk3==1)
                pos, neg = S.labels(batch['ema'][j].numpy(), internal)
                yy,xx = np.ogrid[-3:4,-3:4]
                near = cv2.dilate((gt & valid).astype(np.uint8), ((yy*yy+xx*xx)<=9).astype(np.uint8))>0
                for name, mask in (('U_pixels',internal), ('U_gt',internal&gt), ('pseudo_positive',pos),
                                   ('positive_gt',pos&gt), ('positive_near3',pos&near), ('pseudo_negative',neg),
                                   ('negative_gt',neg&gt), ('ignore_gt',internal&gt&~(pos|neg))):
                    counts[name] += int(mask.sum())
    del ds
    gc.collect()
    counts['positive_purity_strict'] = A.safe_div(counts['positive_gt'],counts['pseudo_positive'])
    counts['positive_purity_near3'] = A.safe_div(counts['positive_near3'],counts['pseudo_positive'])
    counts['gt_in_negative_fraction'] = A.safe_div(counts['negative_gt'],counts['U_gt'])
    counts['eval_seconds'] = time.perf_counter()-start
    rows = A.rows_for_csv(epoch,'target_full',meters)
    main = next(r for r in rows if r['reg']=='ALL' and r['domain']=='agri')
    print(f'[s2l ep{epoch:03d} target_full] P={main["P_seg"]:.4f} R={main["R_seg"]:.4f} IoU={main["IoU_seg"]:.4f}', flush=True)
    return rows, counts


def write_logs(ck):
    U.csv_write(C.output()/'diagnostics.csv',ck['diagnostics'])
    U.csv_write(C.output()/'metrics.csv',ck['metrics'],A.CSV_COLUMNS)
    U.csv_write(C.output()/'pseudo_quality.csv',ck['pseudo_quality'])


def snapshot(ck):
    path = C.output()/'checkpoints'/f'ep{ck["epoch"]:03d}.pth'
    if not path.exists():
        U.torch_write(path, dict(model=ck['model'],model_config=ck['model_config'],epoch=ck['epoch'],
            optimizer_steps=ck['optimizer_steps'],signature=ck['signature'],
            purpose='inference snapshot; resume training from latest.pth only'))


def load_commit(net, opt, scaler, signature):
    latest = C.output()/'latest.pth'
    if not latest.exists():
        cleanup_ema(None)
        return None, None
    ck = torch.load(latest,map_location='cpu',weights_only=True)
    if ck['signature'] != signature:
        raise RuntimeError('Resume signature differs')
    expected = M.model_config(net)
    structural = ('architecture','encoder','encoder_stride','decoder_interpolation','decoder_blocks','align_corners','reference_model_sha256')
    if any(ck['model_config'].get(k)!=expected.get(k) for k in structural):
        raise RuntimeError('Resume model architecture differs')
    meta = ck['ema']
    path = C.output()/meta['file']
    if path.resolve()!=cache_path(ck['epoch']).resolve() or meta['epoch']!=ck['epoch']:
        raise RuntimeError('Resume model/EMA epoch mismatch')
    if not path.is_file() or path.stat().st_size!=meta['bytes'] or U.sha(path)!=meta['sha256']:
        raise RuntimeError('Committed EMA missing/corrupt; never silently rebuild with another model')
    net.load_state_dict(ck['model'],strict=True)
    opt.load_state_dict(ck['optimizer'])
    scaler.load_state_dict(ck['scaler'])
    write_logs(ck)
    U.json_write(C.output()/'progress.json',dict(epoch=ck['epoch'],epochs=C.EPOCHS,warm=C.WARM_EPOCHS,
        optimizer_steps=ck['optimizer_steps'],ema=meta,complete=ck['epoch']==C.EPOCHS))
    if ck['epoch']%C.SAVE_EVERY==0 or ck['epoch']==C.EPOCHS:
        snapshot(ck)
    cleanup_ema(path)
    print(f'[resume] completed epoch={ck["epoch"]}; optimizer_steps={ck["optimizer_steps"]}; EMA verified',flush=True)
    return ck, path


def fit(keys, signature, args):
    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError('Full training requires CUDA; --precheck is allowed on CPU')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    U.seed(C.SEED)
    net = M.build_model(str(C.RESNET50_W),verbose=False).to(device)
    opt = torch.optim.RAdam(net.parameters(),lr=C.LR,weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda',enabled=C.AMP and device.type=='cuda')
    previous, ema_path = load_commit(net,opt,scaler,signature)
    completed = previous['epoch'] if previous else 0
    steps = previous['optimizer_steps'] if previous else 0
    diagnostics = previous['diagnostics'] if previous else []
    metrics = previous['metrics'] if previous else []
    quality = previous['pseudo_quality'] if previous else []
    del previous
    if completed == C.EPOCHS:
        print('[complete] fixed epoch budget already committed',flush=True)
        return
    inference_loader = D.loader(D.Tiles(keys),inference=True,persistent=True)
    for epoch in range(completed+1,C.EPOCHS+1):
        epoch_start = time.perf_counter()
        print(f'=== s2l epoch {epoch:03d}/{C.EPOCHS} ({"warm" if epoch<=C.WARM_EPOCHS else "pseudo"}) ===',flush=True)
        row = train_epoch(net,opt,scaler,keys,ema_path,epoch,device)
        steps += row['optimizer_updates']
        meta, stats = ensemble(net,inference_loader,keys,ema_path,epoch,device)
        if args.smoke_fail_after_ema==epoch:
            raise RuntimeError('Intentional smoke interruption after uncommitted EMA')
        row.update(stats)
        row.update(optimizer_steps=steps, ema_for_training_epoch=(epoch-1 if epoch>C.WARM_EPOCHS else 0),
                   evaluation_seconds=0.)
        if epoch%C.EVALUATE_EVERY==0 or epoch in (C.WARM_EPOCHS,C.EPOCHS):
            new_metrics, q = evaluate(net,keys,cache_path(epoch),epoch,device)
            metrics += new_metrics
            quality.append(q)
            row['evaluation_seconds'] = q['eval_seconds']
        row['epoch_seconds_before_commit'] = time.perf_counter()-epoch_start
        diagnostics.append(row)
        ck = dict(model=net.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
                  model_config=M.model_config(net),ema=meta,optimizer_steps=steps,signature=signature,
                  diagnostics=diagnostics,metrics=metrics,pseudo_quality=quality)
        commit_start = time.perf_counter()
        U.torch_write(C.output()/'latest.pth',ck)
        # latest.pth is the sole commit record. Old EMA is safe to remove only now.
        ema_path = cache_path(epoch)
        cleanup_ema(ema_path)
        write_logs(ck)
        if epoch%C.SAVE_EVERY==0 or epoch==C.EPOCHS:
            snapshot(ck)
        U.json_write(C.output()/'progress.json',dict(epoch=epoch,epochs=C.EPOCHS,warm=C.WARM_EPOCHS,
            optimizer_steps=steps,ema=meta,complete=epoch==C.EPOCHS))
        inner = max(1,row['internal_px'])
        print(f'[epoch {epoch:03d}] loss={row["loss"]:.5f} source={row["source_loss"]:.5f} pseudo={row["pseudo_loss"]:.5f}; '
              f'EMA P/N/I={row["ema_positive"]/inner:.3%}/{row["ema_negative"]/inner:.3%}/{row["ema_ignore"]/inner:.3%}; '
              f'train={row["train_seconds"]:.1f}s EMA={row["ema_seconds"]:.1f}s eval={row["evaluation_seconds"]:.1f}s '
              f'commit={time.perf_counter()-commit_start:.1f}s',flush=True)
        del ck
        if args.stop_after_epoch==epoch:
            print('[stop] requested epoch boundary; run the same command to resume',flush=True)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precheck',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-id',default='v1')
    parser.add_argument('--smoke-workers',type=int,default=0)
    parser.add_argument('--smoke-epochs',type=int,default=3)
    parser.add_argument('--smoke-warm',type=int,default=1)
    parser.add_argument('--smoke-fail-after-ema',type=int,default=0)
    parser.add_argument('--stop-after-epoch',type=int,default=0)
    args = parser.parse_args()
    runtime_setup.configure()
    if args.smoke:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.smoke_id):
            raise ValueError('Unsafe smoke id')
        C.RUN_NAME = '_smoke_unet_s2l_'+args.smoke_id
        C.EPOCHS,C.WARM_EPOCHS = args.smoke_epochs,args.smoke_warm
        C.BATCH_SIZE=C.INFER_BATCH=2
        C.EVALUATE_EVERY=C.SAVE_EVERY=1
        C.NUM_WORKERS=args.smoke_workers
        C.MIN_FREE_GIB=1
        C.PROBE_KEYS = []  # Filled with the exact four former-group smoke images below.
        torch.use_deterministic_algorithms(True)
    elif args.smoke_fail_after_ema or args.smoke_workers:
        parser.error('Fault injection and smoke worker overrides require --smoke')
    C.NUM_WORKERS=min(C.NUM_WORKERS,max(0,runtime_setup.cpu_capacity(C.CPU_LIMIT)-2))
    with U.run_lock(C.output()):
        if args.smoke:
            _,_,groups,_=load_targets(C.SPLIT_JSON)
            C.PROBE_KEYS=smoke_targets(groups)[0]
        keys,signature=plan(args.smoke)
        precheck(keys,signature)
        if not args.precheck:
            fit(keys,signature,args)


if __name__=='__main__':
    main()
