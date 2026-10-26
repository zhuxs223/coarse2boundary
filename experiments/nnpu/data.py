"""Weak-only training data with scalar Oracle priors; aligned augmentation."""
import runtime_setup
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import cfg_standard as C
from reference import assess as A


def read(path, rgb=False):
    a = cv2.imread(str(path), cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if rgb else a


def augment(rgb, fields, rng):
    """Transform all masks/probabilities identically; photometry changes RGB only."""
    axes = []
    if rng.random() < C.AUGMENT['horizontal_flip']:
        axes.append(1)
    if rng.random() < C.AUGMENT['vertical_flip']:
        axes.append(0)
    turns = int(rng.integers(4)) if C.AUGMENT['rotate90'] else 0

    def spatial(a):
        for axis in axes:
            a = np.flip(a, axis=axis)
        return np.rot90(a, turns, axes=(0, 1))

    rgb = spatial(rgb).astype(np.float32) / 255
    fields = {k: spatial(v) for k, v in fields.items()}
    if rng.random() < C.AUGMENT['brightness_contrast_probability']:
        contrast = 1 + rng.uniform(-C.AUGMENT['contrast_limit'], C.AUGMENT['contrast_limit'])
        brightness = rng.uniform(-C.AUGMENT['brightness_limit'], C.AUGMENT['brightness_limit'])
        rgb = np.clip(rgb * contrast + brightness, 0, 1)
    return rgb, fields


class Tiles(Dataset):
    def __init__(self, keys, *, epoch=0, training=False, priors=None, evaluation=False):
        if training and evaluation:
            raise ValueError('Training must not read fine GT')
        self.keys = list(keys)
        self.root = C.DATA_ROOT
        self.size, self.trim = C.IMG_SIZE, C.EDGE_TRIM
        self.epoch, self.training, self.evaluation = epoch, training, evaluation
        self.priors = priors
        if training and priors is None:
            raise ValueError('Training requires scalar priors')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]

        def path(layer):
            return self.root / C.LAYERS[layer] / 'train' / (key + '.png')

        rgb, msk, mk3 = read(path('img'), True), read(path('msk')) > 0, read(path('mk3'))
        if rgb.shape != (self.size, self.size, 3) or msk.shape != mk3.shape or mk3.shape != rgb.shape[:2]:
            raise ValueError('Unexpected image/mask shape: ' + key)
        if not np.isin(mk3, (0, 1, 2)).all():
            raise ValueError('Unexpected coarse label code: ' + key)
        nod = (rgb.max(axis=2) == 0) & ~msk
        valid = A.interior_region(self.size, self.size, self.trim) & ~nod
        fields = dict(mk3=mk3, msk=msk, nod=nod, valid=valid)
        if self.priors is not None:
            rec = self.priors[key]
            if rec['positive'] != int((valid & (mk3==2)).sum()) or rec['unlabeled'] != int((valid & (mk3==1)).sum()):
                raise RuntimeError('Weak data/domain changed since prior preparation: '+key)
        if self.evaluation:
            fields['bnd'] = read(path('bnd')) > 0
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([C.SEED, self.epoch, i]))
            rgb, fields = augment(rgb, fields, rng)
        else:
            rgb = rgb.astype(np.float32) / 255
        rgb = ((rgb - np.array(C.IMAGENET_MEAN, np.float32)) / np.array(C.IMAGENET_STD, np.float32)).transpose(2, 0, 1)
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in fields.items()}
        out.update(img=torch.from_numpy(np.ascontiguousarray(rgb)), idx=i)
        if self.priors is not None:
            out['pi'] = torch.tensor(self.priors[key]['pi'], dtype=torch.float32)
        return out


def worker_init(_):
    runtime_setup.configure()


def loader(ds, *, shuffle=False, inference=False, workers=None, persistent=False, batch_sampler=None):
    nw = C.NUM_WORKERS if workers is None else workers
    options = dict(batch_size=C.INFER_BATCH if inference else C.BATCH_SIZE, shuffle=shuffle,
                   num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=False,
                   generator=torch.Generator().manual_seed(C.SEED + ds.epoch))
    if batch_sampler is not None:
        options.pop('batch_size')
        options.pop('shuffle')
        options.pop('drop_last')
        options['batch_sampler'] = batch_sampler
    if nw:
        options.update(prefetch_factor=2, multiprocessing_context='spawn',
                       worker_init_fn=worker_init, persistent_workers=persistent)
    return DataLoader(ds, **options)
