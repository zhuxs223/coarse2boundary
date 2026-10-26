"""Conservative byte bounds and pruning only committed, obsolete NPZ state files."""
import json,shutil
from pathlib import Path
GIB=1024**3

def state_bound(n,shape):
    pixels=int(shape[0]*shape[1])
    # raw float16 + six packed masks + worst-case dense sparse-carry vectors.
    # Include compression overhead, archive headers, histogram and LUT.
    return int(n*(pixels*18+6*((pixels+7)//8)+20000)*1.02+2*1024**2)

def output_bytes(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def reserve(root,required,max_output_gib,min_free_gib,protected=(),keep_rounds=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    protected={Path(p).resolve() for p in protected if p is not None}
    candidates=[]
    for marker in sorted(root.glob('round_*/done_r*.json')):
        done=json.loads(marker.read_text());p=marker.parent/done['state_name']
        if not done.get('complete'):continue
        expected='state_'+marker.stem.removeprefix('done_')+'.npz'
        if p.name!=expected or p.parent.resolve()!=marker.parent.resolve():raise RuntimeError('Unexpected state prune target')
        if p.exists() and p.resolve() not in protected:candidates.append((marker,p))
    removed=[]
    def insufficient():
        return output_bytes(root)+required>max_output_gib*GIB or shutil.disk_usage(root).free-required<min_free_gib*GIB
    while candidates and (insufficient() or (keep_rounds is not None and len(candidates)>max(0,keep_rounds-1))):
        marker,p=candidates.pop(0);resolved=p.resolve()
        if not resolved.is_relative_to(root) or p.is_symlink():raise RuntimeError('Unsafe state prune path')
        size=p.stat().st_size;p.unlink()
        event=dict(state=str(p.relative_to(root)),bytes=size,reason='space budget / retention policy')
        removed.append(event)
        with (root/'state_pruning.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    if insufficient():
        raise RuntimeError(f'Insufficient space before writing: need {required/GIB:.2f} GiB plus {min_free_gib} GiB free reserve; download/remove older archives or reduce retention. Current predecessor is protected.')
    return removed
