"""Fixed teacher identity and a one-time U-only reset, without reading fine GT."""
import gc
import json
from pathlib import Path
import torch
import numpy as np
import cfg_standard as C
import utils as U
from dataset_protocol import load_targets, key_digest
import teacher_model

# Digests are read from the explicitly registered local teacher provenance.


def validate_teacher():
    provenance = json.loads(C.TEACHER_PROVENANCE.read_text(encoding='utf-8'))
    TEACHER_SHA = provenance['teacher_sha256']
    RESNET_SHA = provenance['resnet_sha256']
    checks = [(C.SMALL_TEACHER, TEACHER_SHA), (C.RESNET50_W, RESNET_SHA),
              (C.SPLIT_JSON, provenance['split_sha256']),
              (C.BASE/provenance['manifest_file'], provenance['manifest_sha256'])]
    if provenance['teacher_sha256'] != TEACHER_SHA or provenance['resnet_sha256'] != RESNET_SHA:
        raise ValueError('Expected the original full5227 SMALL teacher provenance')
    for path, digest in checks:
        if U.sha(path) != digest: raise ValueError(f'Teacher input differs: {path}')
    ck = torch.load(C.SMALL_TEACHER, map_location='cpu', weights_only=True)
    manifest = ck['manifest']
    sidecar = json.loads((C.BASE/provenance['manifest_file']).read_text(encoding='utf-8'))
    keys, _, _, _ = load_targets(C.SPLIT_JSON)
    if (manifest != sidecar or manifest['train_keys'] != keys or manifest['n_train'] != 5227
        or manifest['params'] != 557505 or not manifest['h2_skip'] or ck['epoch'] != 10
        or ck['model_config'] != dict(architecture='SMALL', h2_skip=True) or ck.get('train_attempts', ck['optimizer_updates']) != 4360):
        raise ValueError('Teacher manifest/model configuration differs')
    return dict(sha256=TEACHER_SHA, epoch=10, train_tiles=len(keys), train_keys_sha256=key_digest(keys),
                optimizer_updates=ck['optimizer_updates'], params=557505, path=str(C.SMALL_TEACHER),
                model_source_sha256=U.sha(Path(teacher_model.__file__)))


def load_teacher(device):
    # Forking RNG avoids introducing a random-stream change into the student.
    with torch.random.fork_rng(devices=[]):
        net = teacher_model.build_model(str(C.RESNET50_W), verbose=False)
    ck = torch.load(C.SMALL_TEACHER, map_location='cpu', weights_only=True)
    net.load_state_dict(ck['model'], strict=True)
    del ck
    net.requires_grad_(False).eval().to(device)
    if sum(p.numel() for p in net.parameters()) != 557505:
        raise ValueError('SMALL parameter count differs')
    return net


def reset_internal(value, teacher_probability, internal):
    """No quantization, LUT, confidence filtering, geometry or GT-dependent edit."""
    if value.shape != teacher_probability.shape or internal.shape != value.shape:
        raise ValueError('Prior image/domain shape differs')
    if teacher_probability.dtype != np.float32 or not np.isfinite(teacher_probability).all():
        raise ValueError('Invalid SMALL probability dtype/value')
    if (teacher_probability < 0).any() or (teacher_probability > 1).any():
        raise ValueError('SMALL probability outside [0,1]')
    value = value.copy()
    value[internal] = teacher_probability[internal]
    return value
