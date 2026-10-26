"""Gray-mass candidate localization and explicit pseudo-supervision abstention.

Prototype only: not imported by the training runner. A flagged region is not a
confirmed error. Do not use masked probabilities as new zero-valued labels.
"""
import numpy as np
from scipy import ndimage as ndi

SETTINGS=dict(prob_threshold=.2,min_half_width=5.,reference_rise=.15,
              min_seed_area=48,join_radius=2,padding_radius=12,
              quarantine_footprint='padded component box (includes dark oval holes)')

def detect(prob, internal, reference=None):
    p=np.asarray(prob,np.float32);v=np.asarray(internal,bool)
    if p.shape!=v.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid probability/domain')
    # Broad probability responses, rather than periodicity. No fine GT is read.
    mass=(p>=SETTINGS['prob_threshold'])&v
    width=ndi.distance_transform_edt(mass)
    seed=mass&(width>=SETTINGS['min_half_width'])
    if reference is not None:
        ref=np.asarray(reference,np.float32)
        if ref.shape!=p.shape or not np.isfinite(ref).all() or np.any((ref<0)|(ref>1)):
            raise ValueError('Invalid reference')
        seed &= p-ref>=SETTINGS['reference_rise']
    joined=ndi.binary_closing(seed,iterations=SETTINGS['join_radius'])&v
    labels,n=ndi.label(joined,np.ones((3,3)))
    rows=[];kept=np.zeros(p.shape,bool);exclusion=np.zeros(p.shape,bool)
    for i in range(1,n+1):
        component=labels==i;raw=component&seed;area=int(raw.sum())
        if area<SETTINGS['min_seed_area']:continue
        yy,xx=np.nonzero(component);kept |= component
        pad=SETTINGS['padding_radius']
        exclusion[max(0,yy.min()-pad):min(p.shape[0],yy.max()+1+pad),
                  max(0,xx.min()-pad):min(p.shape[1],xx.max()+1+pad)]=True
        rows.append(dict(seed_pixels=area,box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            max_half_width=float(width[raw].max()),mean_probability=float(p[raw].mean()),
            mean_reference_rise=float((p-ref)[raw].mean()) if reference is not None else None))
    exclusion &= v
    return dict(seed=kept,exclusion=exclusion,candidates=rows,
        note='Morphological width plus disagreement with fixed reference; candidate regions only, not proven anomalies.')

def abstain_targets(weight,soft_mask,soft_target,exclude):
    """Suppress both positive and negative pseudo loss, without creating zeros as negatives."""
    arrays=[np.asarray(x) for x in (weight,soft_mask,soft_target,exclude)]
    if not all(a.shape==arrays[0].shape for a in arrays):raise ValueError('Abstention shapes differ')
    w=np.array(weight,copy=True);s=np.array(soft_mask,dtype=bool,copy=True);t=np.array(soft_target,copy=True)
    e=np.asarray(exclude,bool);w[e]=0;s[e]=False;t[e]=0
    return w,s,t

def safe_geometry_inputs(valid,anchor_center,exclude):
    """Call BEFORE ridge extraction/path qualification, not only after anchor selection."""
    v=np.asarray(valid,bool);a=np.asarray(anchor_center,bool);e=np.asarray(exclude,bool)
    if not (v.shape==a.shape==e.shape):raise ValueError('Geometry shapes differ')
    return v&~e,a&~e

def histogram_domain(internal,exclude):
    """Excluded pixels must not influence refresh quantile/rank normalization."""
    v=np.asarray(internal,bool);e=np.asarray(exclude,bool)
    if v.shape!=e.shape:raise ValueError('Histogram shapes differ')
    return v&~e
