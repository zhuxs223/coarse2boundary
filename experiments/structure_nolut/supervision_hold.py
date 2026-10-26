"""Exact local carry of finalized supervision, independent of torch/fine GT."""
import json
from pathlib import Path
import numpy as np
from target_rules import supervision_targets,ignore_ambiguous_negatives

FIELDS=('indices','weight','soft_mask','soft_target','teacher_candidate','raw_base')

def unpack(path,shape):
    a=np.load(path,allow_pickle=False)
    if a.dtype!=np.uint8 or a.size!=(int(np.prod(shape))+7)//8:raise ValueError('Invalid packed state')
    return np.unpackbits(a,count=int(np.prod(shape))).reshape(shape).astype(bool)

def load_frozen(path,shape):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(FIELDS):raise ValueError('Invalid frozen fields')
        data={name:archive[name].copy() for name in FIELDS}
    ix=data['indices']
    if ix.ndim!=1 or ix.dtype!=np.uint32 or (ix.size and (int(ix[-1])>=np.prod(shape) or np.any(np.diff(ix.astype(np.int64))<=0))):
        raise ValueError('Frozen indices must be sorted, unique and in bounds')
    if any(data[n].shape!=ix.shape for n in FIELDS[1:]):raise ValueError('Frozen vector sizes differ')
    for n in ('weight','soft_target','raw_base'):
        if not np.isfinite(data[n]).all() or np.any((data[n]<0)|(data[n]>1)):raise ValueError('Invalid frozen '+n)
    for n in ('soft_mask','teacher_candidate'):
        if data[n].dtype!=np.bool_:raise ValueError('Invalid frozen boolean '+n)
    if np.any(data['weight'][data['soft_mask']]!=0) or np.any(data['soft_target'][data['soft_mask']]<.5) or np.any(data['soft_target'][~data['soft_mask']]!=0):
        raise ValueError('Inconsistent frozen positive/negative supervision')
    if data['raw_base'].dtype!=np.float16 or data['weight'].dtype!=np.float32 or data['soft_target'].dtype!=np.float32:
        raise ValueError('Invalid frozen precision')
    return data

def save_frozen(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(path)

def apply_frozen(weight,soft_mask,soft_target,teacher_candidate,data):
    for a,n in zip((weight,soft_mask,soft_target,teacher_candidate),('weight','soft_mask','soft_target','teacher_candidate')):
        a.ravel()[data['indices']]=data[n]

class Previous:
    def __init__(self,root):
        self.root=Path(root);self.info=json.loads((self.root/'manifest.json').read_text())
        if not self.info['complete']:raise RuntimeError('Predecessor supervision incomplete')
        self.lut=np.load(self.root/'target_lut.npy',allow_pickle=False)
        self.frozen_keys=set(self.info['guard']['frozen_keys'])

    def raw(self,key,shape):
        a=np.load(Path(self.info['pseudo_dir'])/(key+'.npy'),allow_pickle=False)
        if a.dtype!=np.float16 or a.shape!=shape or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
            raise ValueError('Invalid previous teacher field: '+key)
        return a

    def geometry(self,key,shape):
        return {n:unpack(Path(self.info[d])/(key+'.npy'),shape) for n,d in
            [('support','geometry_dir'),('allowed','bonus_dir'),('maintained','maintained_dir')]}

    def finalized(self,key,internal,raw=None,geometry=None):
        raw=self.raw(key,internal.shape) if raw is None else raw
        g=self.geometry(key,internal.shape) if geometry is None else geometry
        base=self.lut[raw.view(np.uint16)]
        sig=self.info['signature']
        w,s,t=supervision_targets(base,internal,g['support'],allowed=g['allowed'],
            threshold=.5,bonus=sig['bonus'],penalty=sig['penalty'],rejected_penalty=sig['rejected_penalty'],maintained=g['maintained'])
        tc=internal&(base.astype(np.float32)<=.5)
        if key in self.frozen_keys:
            saved=load_frozen(Path(self.info['frozen_dir'])/(key+'.npz'),internal.shape)
            apply_frozen(w,s,t,tc,saved)
        ignore_ambiguous_negatives(w,s,base,internal)
        return w,s,t,tc

    def snapshot(self,key,internal,hold,raw=None,geometry=None):
        raw=self.raw(key,internal.shape) if raw is None else raw
        w,s,t,tc=self.finalized(key,internal,raw,geometry)
        if np.any(hold&~internal):raise ValueError('Cannot hold source/invalid pixels')
        ix=np.flatnonzero(hold).astype(np.uint32)
        return dict(indices=ix,weight=w.ravel()[ix].astype(np.float32),soft_mask=s.ravel()[ix].astype(bool),
            soft_target=t.ravel()[ix].astype(np.float32),teacher_candidate=tc.ravel()[ix].astype(bool),raw_base=raw.ravel()[ix].astype(np.float16))

def merge_geometry(current,previous,hold,incoming):
    # Old qualified geometry is retained for diagnostics/final target identity.
    # Fresh path selection has already run outside hold; old anchors stay trusted.
    for n in ('support','allowed','maintained'):current[n][hold]=previous[n][hold]
    current['next_center'][hold]=incoming[hold]
    for n in ('accepted_center','contained_center'):current[n][hold]=False
    stats=current['stats'];stats.update(
        qualified_support_px=int(current['allowed'].sum()),
        maintained_support_px=int(current['maintained'].sum()),
        bonus_support_px=int((current['allowed']&~current['maintained']).sum()),
        incoming_anchor_px=int(incoming.sum()),next_anchor_px=int(current['next_center'].sum()),
        new_anchor_px=int((current['next_center']&~incoming).sum()),held_anchor_px=int((incoming&hold).sum()))
    if np.any(incoming&~current['next_center']):raise RuntimeError('Trusted incoming anchor removed')
    if np.any(current['accepted_center']&hold):raise RuntimeError('New anchor admitted in held region')
    if np.any(current['allowed']&~current['support']):raise RuntimeError('Held gate/support mismatch')
    return current
