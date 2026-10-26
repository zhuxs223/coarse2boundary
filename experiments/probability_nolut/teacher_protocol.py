"""Reject old/partial teachers and verify the complete target cohort."""
import hashlib
import json
from pathlib import Path

import torch

import cfg_standard as C
from dataset_protocol import load_targets, key_digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_teacher():
    provenance_path = C.BASE/'meta/full_teacher_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    for path, field in ((C.INITIAL_SMALL_TEACHER, 'teacher_sha256'),
                        (C.SPLIT_JSON, 'split_sha256'),
                        (C.RESNET50_W, 'resnet_sha256'),
                        (C.BASE/provenance['manifest_file'], 'manifest_sha256')):
        if sha(path) != provenance[field]:
            raise ValueError(f'Full-teacher package input differs: {path}')
    keys, _, _, _ = load_targets(C.SPLIT_JSON, C.EXPECTED_TARGET_TILES)
    ck = torch.load(C.INITIAL_SMALL_TEACHER, map_location='cpu', weights_only=True)
    manifest = ck['manifest']
    sidecar = json.loads((C.BASE/provenance['manifest_file']).read_text(encoding='utf-8'))
    if manifest != sidecar:
        raise ValueError('Teacher checkpoint and manifest differ')
    if (ck['epoch'] != 10 or manifest['n_train'] != 5227 or not manifest['h2_skip']
            or manifest['train_keys'] != keys or manifest['params'] != 557505
            or ck['model_config'] != dict(architecture='SMALL', h2_skip=True)
            or ck.get('train_attempts', ck['optimizer_updates']) != 4360):
        raise ValueError('Expected fixed ep10 full5227 H/4 teacher with H/2 skip')
    return dict(architecture='SMALL.H4', h2_skip=True, epoch=10, train_tiles=len(keys),
                train_keys_sha256=key_digest(keys), sha256=provenance['teacher_sha256'],
                optimizer_updates=ck['optimizer_updates'], path=str(C.INITIAL_SMALL_TEACHER))
