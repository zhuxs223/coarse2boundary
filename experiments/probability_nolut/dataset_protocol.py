"""Validate the intentional same-target training/evaluation protocol without GT."""
import hashlib
import json
from pathlib import Path

GROUPS=('train_sd','val_sd','test_sd','dropped_buffer')
PROTOCOL='fixed-target-full5227-v1'


def key_digest(keys):
    return hashlib.sha256(('\n'.join(keys)+'\n').encode('utf-8')).hexdigest()


def safe_key(key):
    if (not isinstance(key,str) or not key or '\\' in key or ':' in key
        or key.startswith('/') or any(p in ('','..','.') for p in key.split('/'))):
        raise ValueError(f'Unsafe dataset key: {key!r}')


def load_targets(path,expected_count=5227):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('protocol')!=PROTOCOL:raise ValueError('Not a fixed-target full-data manifest')
    groups=data['source_groups']
    if set(groups)!=set(GROUPS):raise ValueError('Full cohort must include all four source groups')
    keys=[]
    for group in GROUPS:
        values=groups[group]
        if not isinstance(values,list) or not values:raise ValueError(f'Empty/invalid group: {group}')
        for key in values:safe_key(key)
        keys.extend(values)
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate key within/across source groups')
    if len(keys)!=expected_count or data['n_tiles']!=len(keys):raise ValueError('Full cohort size differs')
    if data['train_full']!=keys or data['target_full']!=keys:
        raise ValueError('Training and evaluation must cover the exact complete target cohort')
    if data['keys_sha256']!=key_digest(keys):raise ValueError('Target key order/digest differs')
    metadata=dict(protocol=PROTOCOL,source_group_counts={g:len(groups[g]) for g in GROUPS},
        full_target_tiles=len(keys),keys_sha256=key_digest(keys),
        train_eval_overlap=len(keys),evaluation_role='in-sample GT monitoring only',
        model_selection='fixed round budget; no GT early stopping or best-checkpoint selection')
    return keys,list(keys),groups,metadata


def smoke_targets(groups):
    # One tile from each former group, covering both regions and the original probe.
    train=groups['train_sd']
    first=next((k for k in train if k.endswith('11520_13312')),train[0])
    selected=[first]
    for group,region in zip(GROUPS[1:],('Southwest China/','Central China/','Southwest China/')):
        values=groups[group]
        selected.append(next((k for k in values if k.startswith(region)),values[0]))
    return selected,list(selected)
