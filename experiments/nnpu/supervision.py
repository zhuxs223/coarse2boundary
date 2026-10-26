"""Image-wise logistic nnPU with aggregate negative-risk correction.

P: coarse code 2; N: coarse code 0; U: coarse code 1, always within valid.
Source BCE uses P and N. PU uses P and U, never fine annotation maps.
"""
import torch
import torch.nn.functional as F
import cfg_standard as C


def losses(logits, batch):
    z = logits.float()
    valid, mk3 = batch['valid'].bool(), batch['mk3']
    pi = batch['pi'].float()
    if pi.shape != (z.shape[0],) or not torch.isfinite(pi).all() or ((pi<0)|(pi>1)).any():
        raise ValueError('Invalid per-image class prior')
    p, u = valid & (mk3==2), valid & (mk3==1)
    source = valid & (mk3!=1)
    lp, ln = F.softplus(-z), F.softplus(z)
    ls = (torch.where(mk3==2, lp, ln)*source).sum()/source.sum().clamp(min=1)
    axes = tuple(range(1,z.ndim))
    np_, nu = p.sum(axes), u.sum(axes)
    # A tile without P borrows the pooled positive distribution in its batch.
    # The sampler guarantees that such a reference exists when pi>0 and U exists.
    need_p = (nu>0) & (pi>0)
    if not p.any() and need_p.any():
        raise RuntimeError('No coarse positive reference in PU batch')
    pool_p = (lp*p).sum()/p.sum().clamp(min=1)
    pool_n = (ln*p).sum()/p.sum().clamp(min=1)
    mp = torch.where(np_>0, (lp*p).sum(axes)/np_.clamp(min=1), pool_p)
    mn = torch.where(np_>0, (ln*p).sum(axes)/np_.clamp(min=1), pool_n)
    mu = (ln*u).sum(axes)/nu.clamp(min=1)
    eligible = nu>0
    denom = eligible.sum().clamp(min=1)
    rpos = (pi*mp*eligible).sum()/denom
    rneg = ((mu-pi*mn)*eligible).sum()/denom
    corrective = rneg.detach()<0
    # Algorithm 1: negative gradient ascent correction, not max(0, Rnegative).
    objective = torch.where(corrective, -rneg, rpos+rneg)
    total = ls+C.PU_WEIGHT*objective
    stats = dict(source=ls, pu=objective, rpos=rpos, rneg=rneg,
        corrected=corrective, fallback_tiles=((np_==0)&need_p).sum(),
        U_pixels=u.sum(), P_pixels=p.sum(), eligible_tiles=eligible.sum())
    return total, stats
