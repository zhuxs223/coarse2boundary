"""CPU jobs: original gate, original hold detector, previous final supervision."""
import runtime_setup
import time
import numpy as np
from skimage.morphology import skeletonize
from state_store import Reader
from initial_anchor import single_anchor
from anchor_gate import select
from supervision_hold import merge_geometry
import gray_region_guard as G
from repeat_confidence import eligible_band
PREVIOUS=None
SETTINGS=None

def initialize(previous,settings):
    global PREVIOUS,SETTINGS
    import cv2
    cv2.setNumThreads(1)
    PREVIOUS=Reader(previous) if previous else None;SETTINGS=settings

def process(args):
    i,p,valid,mask,mk=args;started=time.perf_counter();inside=valid&(mk==1)
    hold=np.zeros(p.shape,bool);seed=hold.copy();candidates=[];saved=None;ref=None;old=None
    if PREVIOUS is None:
        center,_,_=single_anchor(p,skeletonize((mk==2)&valid),valid,SETTINGS['search'])
    else:
        old=PREVIOUS.geometry(i);center=old['anchors']
        raw=PREVIOUS.raw(i);ref=np.where(inside,1-raw.astype(np.float32),0)
        if SETTINGS['guard']:
            detection=G.detect(p,inside,ref);hold=detection['exclusion'];seed=detection['seed'];candidates=detection['candidates']
            if hold.any():saved=PREVIOUS.snapshot(i,inside,hold,raw,old)
    safe_p=p.copy()
    if hold.any():safe_p[hold]=ref[hold]
    g=select(safe_p,valid&~hold,mask,mk,center,SETTINGS['extra'],SETTINGS['band'])
    if not SETTINGS['no_repeat']:g['maintained'][:]=False
    if hold.any():merge_geometry(g,old,hold,center)
    else:g['stats']['held_anchor_px']=0
    confidence=SETTINGS.get('repeat_confidence')
    g['repeat_confidence']=eligible_band(mk,valid,g,hold,old,confidence) if confidence else np.zeros(p.shape,bool)
    g['stats']['repeat_confidence_band_px']=int(g['repeat_confidence'].sum())
    raw=np.where(inside,1-p,0).astype(np.float16)
    if saved is not None:raw.ravel()[saved['indices']]=saved['raw_base']
    return dict(i=i,raw=raw,g=g,hold=hold,seed=seed,candidates=candidates,saved=saved,
                reference=ref,seconds=time.perf_counter()-started)
