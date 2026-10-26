"""Local repeat-ridge confidence boost; no new positives or path histories."""
import math
import numpy as np
from scipy import ndimage as ndi

STAT_FIELDS=('eligible_px','raised_px','extra_px','already_high_px',
             'original_target_sum','raw_teacher_sum','final_target_sum','increase_sum',
             'target_ge_09_px','target_ge_099_px','target_one_px')


def validate(settings):
    eta=settings['eta'];stop=settings['stop']
    if not math.isfinite(eta) or not 0<=eta<=1:
        raise ValueError('Repeat confidence eta must be in [0,1]')
    if not math.isfinite(stop) or not .5<stop<=1:
        raise ValueError('Repeat confidence stop must be in (.5,1]')
    for name in ('outline_margin','tolerance'):
        value=settings[name]
        if not isinstance(value,int) or isinstance(value,bool) or value<0:
            raise ValueError('Repeat confidence radii must be nonnegative integers')


def eligible_band(mk,valid,current,hold,previous,settings):
    """Current repeated/allowed support, matched to previous active support."""
    validate(settings)
    eligible=np.zeros(mk.shape,bool)
    if not settings['enabled'] or previous is None:
        return eligible
    active=previous['allowed']&~previous['hold']
    if not active.any():
        return eligible
    radius=settings['tolerance']
    if radius:
        yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
        active=ndi.binary_dilation(active,structure=(xx*xx+yy*yy<=radius*radius))
    eligible=valid&(mk==1)&current['maintained']&current['allowed']&active&~hold
    outline=(mk==2)&valid
    if outline.any():
        eligible&=ndi.distance_transform_edt(~outline)>settings['outline_margin']
    return eligible


def strengthen(target,soft_mask,raw_base,eligible,settings,diagnostics=None):
    """Only raise existing soft positives. High bases retain their confidence."""
    chosen=np.asarray(eligible,bool)&np.asarray(soft_mask,bool)
    if not settings.get('enabled',False):
        chosen=np.zeros(target.shape,bool)
    if diagnostics is not None:
        diagnostics.update({k:0. for k in STAT_FIELDS})
    if not chosen.any():
        return target
    validate(settings)
    original=target[chosen]
    raw_probability=1-np.asarray(raw_base,np.float32)[chosen]
    if not np.isfinite(raw_probability).all() or np.any((raw_probability<0)|(raw_probability>1)):
        raise ValueError('Invalid raw probability for repeat confidence')
    base=np.maximum(original,raw_probability)
    extra=base<np.float32(settings['stop'])
    final=np.where(extra,base+np.float32(settings['eta'])*(1-base),base)
    final=np.clip(final,0,1).astype(np.float32)
    out=target.copy();out[chosen]=final
    if diagnostics is not None:
        diagnostics.update(eligible_px=int(chosen.sum()),raised_px=int((final>original).sum()),
            extra_px=int(extra.sum()),already_high_px=int((~extra).sum()),
            original_target_sum=float(original.sum(dtype=np.float64)),
            raw_teacher_sum=float(raw_probability.sum(dtype=np.float64)),
            final_target_sum=float(final.sum(dtype=np.float64)),
            increase_sum=float((final-original).sum(dtype=np.float64)),
            target_ge_09_px=int((final>=.9).sum()),target_ge_099_px=int((final>=.99).sum()),
            target_one_px=int((final==1).sum()))
    return out
