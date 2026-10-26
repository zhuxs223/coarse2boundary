"""Preserved data/loss semantics, reading a round-local NPZ lazily per worker."""
import runtime_setup
import numpy as np
import torch
import cfg_standard as C
import run_pair as R
from state_store import Reader
from repeat_confidence import STAT_FIELDS

class Tiles(R.Tiles):
    def __init__(self,keys,state=None,evaluation=False):
        super().__init__(keys,evaluation=evaluation)
        self.state_path=str(state) if state is not None else None;self._state=None
    def __getstate__(self):
        a=self.__dict__.copy();a['_state']=None;return a
    def __getitem__(self,i):
        out=super().__getitem__(i)
        if self.state_path is None:return out
        if self._state is None:
            self._state=Reader(self.state_path)
            if self._state.keys!=self.keys:raise ValueError('Dataset/state key order differs')
        internal=out['valid'].numpy()&(out['mk3'].numpy()==1)
        g=self._state.geometry(i);diagnostics={}
        w,s,t,tc=self._state.finalized(i,internal,g=g,diagnostics=diagnostics)
        for name,a in [('weight',w),('soft_mask',s),('soft_target',t),('frozen_mask',g['hold']),
            ('teacher_candidate_mask',tc),('revoked_mask',tc&~s),('geometry_support',g['support']&internal),
            ('bonus_allowed',g['allowed']&internal),('maintained_ridge',g['maintained']&internal),
            ('negative_ignore_mask',diagnostics['negative_ignore_mask'])]:
            out[name]=torch.from_numpy(a)
        out['repeat_confidence_mask']=torch.from_numpy(g['repeat_confidence']&s)
        out['repeat_confidence_stats']=torch.from_numpy(np.array([diagnostics[k] for k in STAT_FIELDS],np.float64))
        return out

def loader(ds,smoke=False,shuffle=False,inference=False,epoch=0,workers=None):
    nw=C.NUM_WORKERS if workers is None else workers
    if smoke:nw=0
    options=dict(batch_size=2 if smoke else C.INFER_BATCH if inference else C.BATCH_SIZE,
        shuffle=shuffle,num_workers=nw,pin_memory=torch.cuda.is_available(),drop_last=False,
        generator=torch.Generator().manual_seed(C.SEED+epoch))
    if nw:options.update(prefetch_factor=2,multiprocessing_context='spawn')
    return torch.utils.data.DataLoader(ds,**options)
