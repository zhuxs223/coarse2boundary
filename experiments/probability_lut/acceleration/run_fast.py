"""Disk-worker entry with timing, per-epoch probes and round-local cleanup."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime_setup
import os,time
import numpy as np
import torch
import cfg_standard as C
import run_all as A
import run_pair as R
import cloud_data as D
from fast_data import Manager

ORIGINAL_TRAIN=R.train_epoch
ORIGINAL_EVAL=R.evaluate
ORIGINAL_ROUND=A.train_round
ORIGINAL_PRECHECK=A.precheck
MANAGER=Manager()
CONTEXT={}

def diagnostic_primitives(value):
    """Keep diagnostic values identical, but serialize NumPy scalars as primitives."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:diagnostic_primitives(v) for k,v in value.items()}
    if isinstance(value,list):return [diagnostic_primitives(v) for v in value]
    if isinstance(value,tuple):return tuple(diagnostic_primitives(v) for v in value)
    return value

def legacy_checkpoint_scalars():
    """Scoped weights-only compatibility for this runner's old numeric diagnostics.

    First-round repeat statistics were saved as NumPy float64 scalars. Preserve
    existing checkpoints and their hashes; never fall back to unrestricted pickle.
    The aliases cover checkpoints produced with either NumPy core module name.
    """
    scalar=np.float64(0).__reduce__()[0]
    return torch.serialization.safe_globals([
        (scalar,'numpy._core.multiarray.scalar'),
        (scalar,'numpy.core.multiarray.scalar'),
        np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32))])

def timeline(net,device):
    if not CONTEXT or os.environ.get('EXP810_TIMELINE','1')=='0':return
    import cv2
    ds=CONTEXT.get('ds')
    if ds is None:return
    epoch=CONTEXT['epoch'];index=CONTEXT['index'];r=A.label(index)
    chosen=ds.keys if CONTEXT['smoke'] else [k for k in C.PROBE_KEYS if k in ds.keys]
    if not chosen:return
    lookup={k:i for i,k in enumerate(ds.keys)}
    start=time.perf_counter();net.eval()
    with torch.inference_mode():
        for offset in range(0,len(chosen),C.INFER_BATCH):
            keys=chosen[offset:offset+C.INFER_BATCH];samples=[ds[lookup[k]] for k in keys]
            images=torch.stack([s['img'] for s in samples]).to(device)
            probabilities=torch.sigmoid(net(images))[:,0].float().cpu().numpy()
            for j,(key,sample) in enumerate(zip(keys,samples)):
                agri=sample['valid'].numpy()&(sample['msk'].numpy()|(sample['mk3'].numpy()==2))
                q=np.where(agri,probabilities[j],0)
                path=A.roundroot(index)/'timeline'/f'epoch_{epoch:03d}'/(key+f'_{r}_ep{epoch:03d}.png')
                path.parent.mkdir(parents=True,exist_ok=True)
                image=np.rint(q*255).clip(0,255).astype(np.uint8)
                if not cv2.imwrite(str(path),image,[cv2.IMWRITE_PNG_COMPRESSION,9]):raise OSError(path)
    print(f'[timeline ep{epoch}] {len(chosen)} probe PNGs, {time.perf_counter()-start:.2f}s',flush=True)

def train(net,opt,scaler,ds,device,smoke,global_epoch):
    epoch=C.START_EPOCH+global_epoch
    if CONTEXT:CONTEXT.update(ds=ds,epoch=epoch)
    print(f'[train ep{epoch}] preparing input',flush=True)
    entry=MANAGER.prepare(ds)
    start=time.perf_counter();diag=ORIGINAL_TRAIN(net,opt,scaler,ds,device,smoke,global_epoch)
    elapsed=time.perf_counter()-start;timed=entry.last
    # Retain the existing CSV fields when resuming a run made by the older entry.
    diag.update(cache_prepare_seconds=0.,data_first_batch_seconds=timed.first_seconds,
        data_wait_seconds=timed.wait_seconds,train_wall_seconds=elapsed,ram_cache=False)
    print(f'[train ep{epoch}] cache_prepare=0.00s first_batch={timed.first_seconds:.3f}s data_wait={timed.wait_seconds:.2f}s train_total={elapsed:.2f}s',flush=True)
    timeline(net,device)
    return diagnostic_primitives(diag)

def evaluate(net,ds,device,smoke):
    print('[eval] preparing input',flush=True);entry=MANAGER.prepare(ds)
    start=time.perf_counter();result=ORIGINAL_EVAL(net,ds,device,smoke)
    print(f'[eval] cache_prepare=0.00s data_wait={entry.last.wait_seconds:.2f}s eval_total={time.perf_counter()-start:.2f}s',flush=True)
    return result

def round_run(index,trainkeys,valkeys,sig,device,smoke=False):
    MANAGER.clear()
    CONTEXT.update(index=index,smoke=smoke)
    try:return ORIGINAL_ROUND(index,trainkeys,valkeys,sig,device,smoke)
    finally:MANAGER.clear();CONTEXT.clear()

def precheck(*args,**kwargs):
    result=ORIGINAL_PRECHECK(*args,**kwargs)
    here=Path(__file__).resolve().parent
    A.R.atomic_json(C.output()/'acceleration_runtime.json',dict(data_source='disk',cache_enabled=False,
        persistent_workers=True,train_workers=C.NUM_WORKERS,geometry_workers=C.GEOMETRY_WORKERS,
        state_keep_rounds=C.STATE_KEEP_ROUNDS,
        code={p.name:A.R.sha(p) for p in here.glob('*.py') if not p.name.startswith('test_')},
        scientific_entry='full-target run_all.py',sampler='original ordering retained',
        metric_split=C.METRIC_SPLIT,evaluate_every_epochs=C.EVALUATE_EVERY_EPOCHS))
    return result

def install():
    D.loader=MANAGER.loader;R.loader=MANAGER.loader
    R.train_epoch=train;R.evaluate=evaluate;A.train_round=round_run;A.precheck=precheck

def main():
    C.SMOKE_RUN_ID=os.environ.get('EXP810_SMOKE_ID',C.SMOKE_RUN_ID)
    retention=os.environ.get('EXP810_STATE_KEEP_ROUNDS','2').lower()
    C.STATE_KEEP_ROUNDS=None if retention in ('all','none') else int(retention)
    if C.STATE_KEEP_ROUNDS is not None and C.STATE_KEEP_ROUNDS<1:raise ValueError('Retain at least one committed state')
    install()
    print('[acceleration] cache=disk; persistent workers; per-epoch timeline enabled',flush=True)
    try:
        with legacy_checkpoint_scalars():A.main()
    finally:MANAGER.clear()

if __name__=='__main__':main()
