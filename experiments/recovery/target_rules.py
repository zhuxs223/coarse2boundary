"""Qualified ridge q+.1; ALL unqualified ridges q-.1; revoke below .5."""
import numpy as np

NEGATIVE_IGNORE_LOW=np.float32(.2)
NEGATIVE_IGNORE_HIGH=np.float32(.5)


def ignore_ambiguous_negatives(weight,soft_mask,base,internal):
    """Drop only existing negative supervision with mapped teacher q in [.2,.5)."""
    q=1-np.asarray(base,np.float32)
    ignored=(np.asarray(internal,bool)&~np.asarray(soft_mask,bool)&
             (q>=NEGATIVE_IGNORE_LOW)&(q<NEGATIVE_IGNORE_HIGH))
    weight[ignored]=0
    return ignored


def supervision_targets(base,internal,support,allowed=None,threshold=.5,bonus=.1,penalty=.1,rejected_penalty=.1,maintained=None):
    base=np.asarray(base,np.float32);internal=np.asarray(internal,bool);support=np.asarray(support,bool)&internal
    allow=support if allowed is None else np.asarray(allowed,bool)&support
    hold=np.zeros(base.shape,bool) if maintained is None else np.asarray(maintained,bool)
    if not all(x.shape==base.shape for x in (internal,support,allow,hold)):
        raise ValueError('Target/gate shapes differ')
    if np.any(hold&~allow):raise ValueError('Maintained ridges must remain qualified')
    if not np.isfinite(base).all() or np.any((base<0)|(base>1)):
        raise ValueError('Invalid mapped 1-q')
    if threshold!=.5 or bonus<0 or penalty<0 or rejected_penalty<0:raise ValueError('Invalid target parameters')
    q=1-base
    delta=np.where(support,np.where(allow,np.where(hold,np.float32(0),np.float32(bonus)),np.float32(-rejected_penalty)),np.float32(-penalty))
    adjusted=np.clip(q+delta,0,1)
    selected=internal&(base<=1-threshold)&(adjusted>=threshold)
    target=np.where(selected,adjusted,np.float32(0)).astype(np.float32)
    weight=np.where(internal&~selected,base**2,np.float32(0)).astype(np.float32)
    return weight,selected,target
