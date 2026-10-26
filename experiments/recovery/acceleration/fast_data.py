"""Timed disk loaders with persistent workers and original sample ordering."""
import runtime_setup
import gc,time
import torch
import cfg_standard as C
import cloud_data as D

ORIGINAL_LOADER=D.loader

class TimedLoader:
    def __init__(self,loader,tag):
        self.loader=loader;self.tag=tag;self.first_seconds=0.;self.wait_seconds=0.
    def __len__(self):return len(self.loader)
    def __iter__(self):
        self.first_seconds=0.;self.wait_seconds=0.
        started=time.perf_counter();it=iter(self.loader);first=True
        while True:
            try:batch=next(it)
            except StopIteration:
                self.wait_seconds+=time.perf_counter()-started
                return
            seconds=time.perf_counter()-started;self.wait_seconds+=seconds
            if first:
                self.first_seconds=seconds;first=False
                print(f'[data {self.tag}] first_batch_wait={seconds:.3f}s workers={self.loader.num_workers} cache=disk',flush=True)
            yield batch
            started=time.perf_counter()

class Entry:
    def __init__(self,ds):
        self.ds=ds;self.loaders={};self.last=None

class Manager:
    def __init__(self):self.entries={}
    def eligible(self,ds):return bool(getattr(ds,'state_path',None)) or bool(getattr(ds,'evaluation',False))
    def prepare(self,ds):
        if id(ds) not in self.entries:self.entries[id(ds)]=Entry(ds)
        return self.entries[id(ds)]
    def loader(self,ds,smoke=False,shuffle=False,inference=False,epoch=0,workers=None):
        if not self.eligible(ds):
            # Teacher inference is a single pass; small probes need no worker startup.
            nw=0 if len(ds)<=C.INFER_BATCH else workers
            return ORIGINAL_LOADER(ds,smoke,shuffle,inference,epoch,nw)
        entry=self.prepare(ds);nw=0 if smoke else C.NUM_WORKERS if workers is None else workers
        batch=2 if smoke else C.INFER_BATCH if inference else C.BATCH_SIZE
        key=(shuffle,inference,nw,batch);seed=C.SEED+epoch
        if key not in entry.loaders:
            options=dict(batch_size=batch,shuffle=shuffle,num_workers=nw,pin_memory=torch.cuda.is_available(),
                drop_last=False,generator=torch.Generator())
            if nw:options.update(prefetch_factor=2,multiprocessing_context='spawn',persistent_workers=True)
            entry.loaders[key]=torch.utils.data.DataLoader(ds,**options)
        dl=entry.loaders[key];dl.generator.manual_seed(seed)
        # Persistent loaders omit the later base_seed draws made by fresh loaders.
        # Consume that draw explicitly to retain the original RandomSampler order.
        if nw and dl._iterator is not None:torch.empty((),dtype=torch.int64).random_(generator=dl.generator)
        timed=TimedLoader(dl,f'{C.METRIC_SPLIT if inference else "train"} ep{epoch}');entry.last=timed
        return timed
    def clear(self):
        try:
            for entry in self.entries.values():
                for dl in entry.loaders.values():
                    if dl._iterator is not None:
                        dl._iterator._shutdown_workers();dl._iterator=None
                state=getattr(entry.ds,'_state',None)
                if state is not None:state.close();entry.ds._state=None
        finally:self.entries.clear();gc.collect()
