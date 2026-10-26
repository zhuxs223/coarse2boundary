"""One independent, streaming NPZ per round; no all-round archive or huge RAM arrays."""
import runtime_setup
import json,os,zipfile
from pathlib import Path
import numpy as np
from target_rules import supervision_targets,ignore_ambiguous_negatives
from supervision_hold import FIELDS,apply_frozen
from repeat_confidence import strengthen

VERSION=1
MASKS=('support','allowed','maintained','anchors','hold')

def pack(a):return np.packbits(np.asarray(a,bool).ravel())
def unpack(a,shape):
    if a.dtype!=np.uint8 or a.size!=(int(np.prod(shape))+7)//8:raise ValueError('Invalid packed state')
    return np.unpackbits(a,count=int(np.prod(shape))).reshape(shape).astype(bool)

class Writer:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.tmp=self.path.with_name(self.path.name+'.building')
        self.z=zipfile.ZipFile(self.tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)
        self.written=set()
    def put(self,name,a):
        if name in self.written:raise ValueError('Duplicate NPZ field')
        a=np.asarray(a)
        if a.dtype.hasobject:raise ValueError('Object arrays forbidden')
        with self.z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,a,allow_pickle=False)
        self.written.add(name)
    def tile(self,i,raw,g,hold,frozen=None):
        prefix=f't{i:06d}_'
        self.put(prefix+'raw',raw)
        for name in MASKS:
            a=hold if name=='hold' else g['next_center'] if name=='anchors' else g[name]
            self.put(prefix+name,pack(a))
        if 'repeat_confidence' in g:
            self.put(prefix+'repeat_confidence',pack(g['repeat_confidence']))
        if frozen is not None:
            for name in FIELDS:self.put(prefix+'frozen_'+name,frozen[name])
    def commit(self,info,lut,hist):
        self.put('target_lut',lut);self.put('raw_histogram',hist)
        self.put('metadata',np.frombuffer(json.dumps(dict(version=VERSION,**info),ensure_ascii=False).encode('utf-8'),np.uint8))
        self.z.close()
        with self.tmp.open('rb+') as f:os.fsync(f.fileno())
        self.tmp.replace(self.path)
    def abort(self):
        self.z.close()
        if self.tmp.exists():self.tmp.unlink()

class Reader:
    def __init__(self,path):
        self.path=Path(path);self._z=None
        z=self.archive();self.info=json.loads(z['metadata'].tobytes().decode('utf-8'))
        if self.info['version']!=VERSION or not self.info.get('complete'):raise ValueError('Incomplete/unknown state')
        self.keys=self.info['keys'];self.lookup={k:i for i,k in enumerate(self.keys)}
        if len(self.lookup)!=len(self.keys):raise ValueError('Duplicate state keys')
        self.shape=tuple(self.info['shape']);self.lut=z['target_lut']
        if self.lut.shape!=(65536,) or self.lut.dtype!=np.float16 or not np.isfinite(self.lut).all():raise ValueError('Invalid LUT')
        self.frozen_indices=set(self.info['frozen_tiles'])
    def archive(self):
        if self._z is None:self._z=np.load(self.path,allow_pickle=False)
        return self._z
    def close(self):
        if self._z is not None:self._z.close();self._z=None
    def __getstate__(self):
        a=self.__dict__.copy();a['_z']=None;return a
    def raw(self,i):
        a=self.archive()[f't{i:06d}_raw']
        if a.dtype!=np.float16 or a.shape!=self.shape or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid raw state')
        return a
    def geometry(self,i):
        z=self.archive();prefix=f't{i:06d}_'
        g={n:unpack(z[prefix+n],self.shape) for n in MASKS}
        name=prefix+'repeat_confidence'
        if name in z:
            g['repeat_confidence']=unpack(z[name],self.shape)
        elif self.info.get('repeat_confidence',{}).get('enabled',False):
            raise ValueError('Missing repeat confidence mask')
        else:g['repeat_confidence']=np.zeros(self.shape,bool)
        return g
    def frozen(self,i):
        if i not in self.frozen_indices:return None
        z=self.archive();prefix=f't{i:06d}_frozen_';a={n:z[prefix+n] for n in FIELDS};ix=a['indices']
        if ix.ndim!=1 or ix.dtype!=np.uint32 or (ix.size and (ix[-1]>=np.prod(self.shape) or np.any(np.diff(ix.astype(np.int64))<=0))):raise ValueError('Invalid frozen indices')
        if any(a[n].shape!=ix.shape for n in FIELDS[1:]):raise ValueError('Invalid frozen lengths')
        for n,dtype in [('weight',np.float32),('soft_target',np.float32),('raw_base',np.float16)]:
            if a[n].dtype!=dtype or not np.isfinite(a[n]).all() or np.any((a[n]<0)|(a[n]>1)):raise ValueError('Invalid frozen '+n)
        if a['soft_mask'].dtype!=bool or a['teacher_candidate'].dtype!=bool:raise ValueError('Invalid frozen flags')
        if np.any(a['weight'][a['soft_mask']]!=0) or np.any(a['soft_target'][a['soft_mask']]<.5) or np.any(a['soft_target'][~a['soft_mask']]!=0):raise ValueError('Inconsistent carried targets')
        return a
    def finalized(self,i,internal,raw=None,g=None,diagnostics=None):
        raw=self.raw(i) if raw is None else raw;g=self.geometry(i) if g is None else g
        if np.any((raw>0)&~internal):raise ValueError('Teacher values outside internal domain')
        if np.any(g['hold']&~internal) or np.any(g['allowed']&~g['support']):raise ValueError('Invalid geometry domain')
        base=self.lut[raw.view(np.uint16)];rules=self.info['rules']
        w,s,t=supervision_targets(base,internal,g['support'],allowed=g['allowed'],maintained=g['maintained'],**rules)
        eligible=g.get('repeat_confidence',np.zeros(self.shape,bool))
        if np.any(eligible&(~internal|~g['maintained']|~g['allowed']|g['hold'])):
            raise ValueError('Invalid repeat confidence eligibility')
        t=strengthen(t,s,raw,eligible,self.info.get('repeat_confidence',{'enabled':False}),diagnostics)
        tc=internal&(base.astype(np.float32)<=.5)
        saved=self.frozen(i)
        if saved is not None:
            if not np.array_equal(saved['indices'],np.flatnonzero(g['hold']).astype(np.uint32)):raise ValueError('Carry/hold mismatch')
            apply_frozen(w,s,t,tc,saved)
        elif g['hold'].any():raise ValueError('Missing carried targets')
        ignored=ignore_ambiguous_negatives(w,s,base,internal)
        if diagnostics is not None:diagnostics['negative_ignore_mask']=ignored
        return w,s,t,tc
    def snapshot(self,i,internal,hold,raw,g):
        w,s,t,tc=self.finalized(i,internal,raw,g);ix=np.flatnonzero(hold).astype(np.uint32)
        return dict(indices=ix,weight=w.ravel()[ix].copy(),soft_mask=s.ravel()[ix].copy(),
            soft_target=t.ravel()[ix].copy(),teacher_candidate=tc.ravel()[ix].copy(),raw_base=raw.ravel()[ix].copy())
