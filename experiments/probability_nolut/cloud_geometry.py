"""Probability-only state; retain independent local guard, never run geometry."""
import runtime_setup
import time
import numpy as np
from state_store import Reader
import gray_region_guard as G
PREVIOUS=None
SETTINGS=None
GEOMETRY_FIELDS=('paths','touching_paths','contained_touching_paths','accepted_paths',
    'fully_overlapping_paths','raw_ridge_px','accepted_center_px','bonus_support_px',
    'qualified_support_px','maintained_support_px','incoming_anchor_px','next_anchor_px',
    'new_anchor_px','abc_center_px','abc_support_px','steger_accepted_center_px',
    'abc_seeded_touching_paths','abc_A_windows','abc_B_windows','abc_C_chains',
    'abc_A_center_px','abc_B_center_px','abc_C_center_px','abc_contained_candidates_rejected',
    'abc_overlapping_candidates','held_anchor_px','repeat_confidence_band_px')

def initialize(previous,settings):
    global PREVIOUS,SETTINGS
    import cv2
    cv2.setNumThreads(1)
    if settings.get('repeat_confidence',{}).get('enabled',False):
        raise ValueError('Cannot enable structure-dependent repeat confidence')
    PREVIOUS=Reader(previous) if previous else None;SETTINGS=settings

def process(args):
    i,p,valid,mask,mk=args;started=time.perf_counter();inside=valid&(mk==1)
    hold=np.zeros(p.shape,bool);seed=hold.copy();candidates=[];saved=None;ref=None;old=None
    if PREVIOUS is not None:
        old=PREVIOUS.geometry(i)
        if any(old[name].any() for name in ('support','allowed','maintained','anchors','repeat_confidence')):
            raise ValueError('Cannot reuse structural state in probability-only ablation')
        raw=PREVIOUS.raw(i);ref=np.where(inside,1-raw.astype(np.float32),0)
        if SETTINGS['guard']:
            detection=G.detect(p,inside,ref);hold=detection['exclusion'];seed=detection['seed'];candidates=detection['candidates']
            if hold.any():saved=PREVIOUS.snapshot(i,inside,hold,raw,old)
    g={name:np.zeros(p.shape,bool) for name in
       ('support','allowed','maintained','next_center','repeat_confidence')}
    g['stats']={name:0 for name in GEOMETRY_FIELDS}
    g['stats']['probability_only_tiles']=1
    raw=np.where(inside,1-p,0).astype(np.float16)
    if saved is not None:raw.ravel()[saved['indices']]=saved['raw_base']
    return dict(i=i,raw=raw,g=g,hold=hold,seed=seed,candidates=candidates,saved=saved,
                reference=ref,seconds=time.perf_counter()-started)
