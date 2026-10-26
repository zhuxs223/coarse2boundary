"""Paired fixed-target SMALL experiments. Fine boundaries are scoring-only."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import assess as A
import model_reference as M

BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('EXP810_DATA', str(BASE.parents[1]/'data/GTPBD_enhenced_png')))
FULL_SPLIT = Path(os.environ.get('EXP810_SPLIT', str(BASE.parents[1]/'meta/splits_full5227.json')))
OLD_SPLIT = BASE.parents[1]/'meta/splits_v4_sd.json'
WEIGHTS = Path(os.environ.get('EXP810_RESNET', str(BASE.parents[1]/'weights/resnet50-0676ba61.pth')))
# The development-only 4,127-tile teacher is not part of this release.
MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]
DOMAINS = (*A.DOMAINS, 'training_U')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def append_csv(path, rows):
    if not rows:
        return
    exists = Path(path).exists()
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Samples(Dataset):
    def __init__(self, keys, scoring=False, root=ROOT):
        self.keys = list(keys)
        self.scoring = scoring
        self.root = Path(root)

    def __len__(self):
        return len(self.keys)

    def read(self, layer, key, rgb=False):
        path = self.root / layer / 'train' / (key + '.png')
        a = cv2.imread(str(path), cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if rgb else a

    def __getitem__(self, i):
        key = self.keys[i]
        rgb = self.read('images', key, True)
        msk = self.read('mask_labels', key) > 0
        mk3 = self.read('mask_labels_3px', key)
        if rgb.shape != (512, 512, 3) or msk.shape != (512, 512) or mk3.shape != (512, 512):
            raise ValueError(f'Unexpected shape: {key}')
        if not np.isin(mk3, [0, 1, 2]).all():
            raise ValueError(f'Unexpected weak-label values: {key}')
        nod = (rgb.max(axis=2) == 0) & ~msk
        x = (rgb.transpose(2, 0, 1).astype(np.float32)/255 - MEAN)/STD
        sample = dict(img=torch.from_numpy(np.ascontiguousarray(x)),
                      msk=torch.from_numpy(msk), mk3=torch.from_numpy(mk3),
                      nod=torch.from_numpy(nod), idx=i)
        # This branch is never entered by the training loader.
        if self.scoring:
            sample['bnd'] = torch.from_numpy(self.read('boundary_labels', key) > 0)
        return sample


class Small(M.BSENet):
    def __init__(self, h2=True):
        super().__init__(str(WEIGHTS), verbose=False)
        self.h2 = h2

    def forward(self, x):
        f1 = self.relu(self.bn1(self.conv1(x)))
        f2 = self.layer1(self.maxpool(f1))
        return self.head(f2, f1 if self.h2 else torch.zeros_like(f1))


def state_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def loader(ds, workers, training=False):
    # Evaluation iteration cannot perturb the training sampler's generator.
    gen = torch.Generator().manual_seed(42 if training else 10042)
    return DataLoader(ds, batch_size=12, shuffle=training, num_workers=workers,
                      pin_memory=True, drop_last=False, generator=gen,
                      persistent_workers=workers > 0)


def loss_fn(logits, batch, device):
    mk3 = batch['mk3'].to(device)
    valid = ~batch['nod'].to(device)
    valid[:, 0, :] = False
    valid[:, -1, :] = False
    valid[:, :, 0] = False
    valid[:, :, -1] = False
    w = ((mk3 != 1) & valid).float()
    target = (mk3 == 2).float()
    return (F.binary_cross_entropy_with_logits(logits[:, 0].float(), target,
                                               reduction='none')*w).sum()/w.sum().clamp_min(1)


@torch.inference_mode()
def evaluate(model, dl, device, old_groups, probes=(), save_dir=None, label=''):
    model.eval()
    meters = {}
    per_image = []
    probe_set = set(probes)
    start = time.time()
    for batch in dl:
        prob = torch.sigmoid(model(batch['img'].to(device, non_blocking=True)))[:, 0].cpu().numpy()
        for j, idx in enumerate(batch['idx'].tolist()):
            key = dl.dataset.keys[idx]
            msk = batch['msk'][j].numpy()
            mk3 = batch['mk3'][j].numpy()
            nod = batch['nod'][j].numpy()
            gt = batch['bnd'][j].numpy()
            doms = A.build_domains(512, 512, msk, gt, nod, mk3, 1)
            doms['training_U'] = doms['valid'] & (mk3 == 1)
            group = old_groups[key]
            region = 'CC' if key.startswith('Central China/') else 'SW'
            groups = ('ALL', region, group, 'old4127' if group == 'train_sd' else 'added1100')
            pred = prob[j] >= .5
            for domain, dom in doms.items():
                cell = A.cell_stats(pred, gt, dom)
                for g in groups:
                    meter = meters.setdefault((g, domain), A.Meter())
                    meter.add_cell(*cell)
                if domain in ('agri', 'training_U'):
                    meter = A.Meter()
                    meter.add_cell(*cell)
                    per_image.append(dict(key=key, source_group=group, domain=domain, **meter.row()))
            if save_dir and key in probe_set:
                dest = Path(save_dir) / key.replace('/', '__')
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(dest) + '.npy', prob[j].astype(np.float32))
                Image.fromarray(np.rint(prob[j]*255).astype(np.uint8)).save(str(dest) + '.png')
    rows = [dict(group=g, domain=d, **m.row(), TP=m.tp, FP=m.fp, FN=m.fn)
            for (g, d), m in sorted(meters.items())]
    headline = meters[('ALL', 'agri')].row()
    print(f'[{label}] n={len(dl.dataset)} agri P={headline["P_seg"]:.4f} '
          f'R={headline["R_seg"]:.4f} IoU={headline["IoU_seg"]:.4f} '
          f'eval_s={time.time()-start:.1f}', flush=True)
    return rows, per_image


def data_info():
    full = json.loads(FULL_SPLIT.read_text(encoding='utf-8'))
    old = json.loads(OLD_SPLIT.read_text(encoding='utf-8'))
    keys = full['train_full']
    assert len(keys) == len(set(keys)) == 5227
    groups = {key: group for group, values in full['source_groups'].items() for key in values}
    original = json.loads((BASE.parents[1]/'meta/paper_teacher_manifest.json').read_text(encoding='utf-8'))
    probes = original['probes']
    assert all(key in groups for key in probes)
    return keys, old['val_sd'], groups, probes


def precheck(keys, groups):
    for layer in ('images', 'mask_labels', 'mask_labels_3px', 'boundary_labels'):
        missing = [k for k in keys if not (ROOT/layer/'train'/(k+'.png')).is_file()]
        assert not missing, (layer, missing[:3])
    seed_all()
    on = Small(True)
    seed_all()
    off = Small(False)
    assert state_sha(on) == state_sha(off), 'Paired initial weights differ'
    ds = Samples([keys[0], keys[-1]])
    batch = next(iter(loader(ds, 0)))
    assert 'bnd' not in batch, 'Training loader exposes fine GT'
    # Architecture and weak loss/gradient validation on real 512px images.
    device = torch.device('cuda')
    results = []
    for net in (on, off):
        net.to(device).train()
        capture = []
        hook = net.head.register_forward_pre_hook(lambda module, args: capture.append(args[1].detach()))
        with torch.autocast('cuda'):
            logits = net(batch['img'].to(device))
            loss = loss_fn(logits, batch, device)
        loss.backward()
        hook.remove()
        assert logits.shape == (2, 1, 512, 512) and torch.isfinite(loss)
        assert net.conv1.weight.grad.abs().sum() > 0
        skip_nonzero = int(torch.count_nonzero(capture[0]))
        assert (skip_nonzero > 0) == net.h2
        before = net.conv1.weight.detach().clone()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
        optimizer.step()
        assert not torch.equal(before, net.conv1.weight), 'Optimizer did not update the backbone'
        rows, _ = evaluate(net, loader(Samples([keys[0], keys[-1]], True), 0), device,
                           groups, label=f'precheck h2={net.h2}')
        assert next(r for r in rows if r['group']=='ALL' and r['domain']=='agri')['n_img'] == 2
        results.append(dict(h2=net.h2, loss=float(loss.detach()), skip_nonzero=skip_nonzero))
        del optimizer, before
        net.cpu()
    torch.cuda.empty_cache()
    print('[precheck] PASS: 5227 files x4; identical initialization; training has no fine GT; '
          f'512px forward/backward {results}', flush=True)
    return results


def train_arm(name, h2, args, keys, val_keys, groups, probes):
    out = args.out/name
    out.mkdir(parents=True, exist_ok=False)
    seed_all()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device('cuda')
    model = Small(h2).to(device)
    init_sha = state_sha(model)
    decay = [p for p in model.parameters() if p.ndim != 1]
    no_decay = [p for p in model.parameters() if p.ndim == 1]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.01),
                                   dict(params=no_decay, weight_decay=0)], lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    train_dl = loader(Samples(keys), args.workers, True)
    val_dl = loader(Samples(val_keys, True), args.workers)
    full_dl = loader(Samples(keys, True), args.workers)
    probe_dl = loader(Samples(probes, True), 0)
    manifest = dict(arm=name, h2_skip=h2, initial_state_sha256=init_sha,
                    n_train=len(keys), batch_size=12, epochs=10, seed=42,
                    lr=1e-4, weight_decay=.01, optimizer='AdamW (1D tensors: no decay)',
                    AMP=True, threshold=.5, edge_trim=1, augmentation='none',
                    params=sum(p.numel() for p in model.parameters()),
                    positive='mk3==2', background='mk3==0', ignored='mk3==1 or nodata or 1px frame',
                    fine_gt='scoring only; fixed terminal epoch10; no best checkpoint selection',
                    steps_per_epoch=len(train_dl), train_keys=keys, probes=probes,
                    off_definition='zero only H/2 skip before concatenation, from epoch1')
    dump(out/'manifest.json', manifest)
    start = time.time()
    attempts = updates = 0
    for epoch in range(1, 11):
        model.train()
        epoch_start = time.time()
        total_loss = 0.
        order_hash = hashlib.sha256()
        for step, batch in enumerate(train_dl, 1):
            order_hash.update(np.asarray(batch['idx'], dtype=np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss = loss_fn(model(batch['img'].to(device, non_blocking=True)), batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {name} ep{epoch} step{step}')
            scaler.scale(loss).backward()
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates += int(scaler.get_scale() >= before)
            attempts += 1
            total_loss += float(loss.detach())
            if step % 100 == 0 or step == len(train_dl):
                print(f'[{name} ep{epoch:02d}] step={step}/{len(train_dl)} '
                      f'loss={total_loss/step:.6f} elapsed_s={time.time()-epoch_start:.1f}', flush=True)
        checkpoint = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), cfg_exp=name,
                          model_config=dict(architecture='SMALL', h2_skip=h2),
                          sup=dict(pos=2, ignore=1, bg_weight=1),
                          train_attempts=attempts, optimizer_updates=updates,
                          initial_state_sha256=init_sha, manifest=manifest)
        tmp = out/f'ep{epoch:02d}.pth.tmp'
        torch.save(checkpoint, tmp)
        tmp.replace(out/f'ep{epoch:02d}.pth')
        rows, _ = evaluate(model, val_dl, device, groups, label=f'{name} ep{epoch} old_val414')
        append_csv(out/'metrics.csv', [dict(epoch=epoch, split='old_val414', **r) for r in rows])
        evaluate(model, probe_dl, device, groups, probes, out/'probes'/f'ep{epoch:02d}',
                 label=f'{name} ep{epoch} probes')
        append_csv(out/'train.csv', [dict(epoch=epoch, loss=total_loss/len(train_dl),
                   attempts=attempts, optimizer_updates=updates,
                   sample_order_sha256=order_hash.hexdigest(), seconds=time.time()-epoch_start)])
        dump(out/'status.json', dict(status='running', epoch=epoch, attempts=attempts,
                                     optimizer_updates=updates))
    rows, images = evaluate(model, full_dl, device, groups, label=f'{name} ep10 full5227')
    append_csv(out/'metrics.csv', [dict(epoch=10, split='full5227', **r) for r in rows])
    append_csv(out/'full5227_per_image.csv', images)
    dump(out/'status.json', dict(status='complete', epoch=10, attempts=attempts,
                                 optimizer_updates=updates, runtime_s=time.time()-start,
                                 peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3))
    del train_dl, val_dl, full_dl, probe_dl, optimizer, model, checkpoint
    torch.cuda.empty_cache()


def summarize(out):
    records = []
    for name in ('old4127_h2_on_ep10', 'full_h2_on_ep10', 'full_h2_off_ep10'):
        for r in csv.DictReader((out/name/'metrics.csv').open(encoding='utf-8-sig')):
            if r['split'] == 'full5227':
                records.append(dict(arm=name, **r))
    append_csv(out/'comparison.csv', records)
    a = list(csv.DictReader((out/'full_h2_on_ep10'/'train.csv').open(encoding='utf-8-sig')))
    b = list(csv.DictReader((out/'full_h2_off_ep10'/'train.csv').open(encoding='utf-8-sig')))
    assert [r['sample_order_sha256'] for r in a] == [r['sample_order_sha256'] for r in b]
    on = json.loads((out/'full_h2_on_ep10'/'manifest.json').read_text(encoding='utf-8'))
    off = json.loads((out/'full_h2_off_ep10'/'manifest.json').read_text(encoding='utf-8'))
    assert on['initial_state_sha256'] == off['initial_state_sha256']
    dump(out/'pair_checks.json', dict(initial_weights_identical=True, epoch_sample_orders_identical=True,
                                     expected_attempts_per_arm=4360, fixed_checkpoint='ep10'))
    print('[pair] COMPLETE; matched initialization and all 10 epoch sample orders verified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train the fixed full-cohort SMALL H/4 initializer.')
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, default=BASE.parents[1]/'out/initialization')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True
    keys, val_keys, groups, probes = data_info()
    if args.precheck:
        precheck(keys, groups)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('The reported SMALL training uses a CUDA GPU.')
    args.out.mkdir(parents=True, exist_ok=True)
    train_arm('full_h2_on_ep10', True, args, keys, val_keys, groups, probes)


if __name__ == '__main__':
    main()
