"""Only valid mk3==0 negatives are scaled; original denominators are fixed."""
import math
import torch
import torch.nn.functional as F

def parts(logits,batch,alpha,pseudo_weight=.25,soft_weight=1.,outline_target=1.):
    if not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('Outside weight must lie in [0,1]')
    valid=batch['valid'].bool();mk3=batch['mk3']
    outside=valid&(mk3==0);outline=valid&(mk3==2);internal=valid&(mk3==1)
    source=valid&(mk3!=1)
    source_denominator=source.sum().clamp(min=1)
    internal_denominator=internal.sum().clamp(min=1)
    source_bce=F.binary_cross_entropy_with_logits(logits.float(),(mk3==2).float()*outline_target,reduction='none')
    neg_bce=F.binary_cross_entropy_with_logits(logits.float(),torch.zeros_like(logits,dtype=torch.float32),reduction='none')
    soft_mask=batch['soft_mask'].detach().bool()&internal
    soft_bce=F.binary_cross_entropy_with_logits(logits.float(),batch['soft_target'].detach().float(),reduction='none')
    # At alpha=1 retain the original arithmetic, as well as the original scale.
    source_weight=torch.where(mk3==0,torch.full_like(logits,alpha,dtype=torch.float32),torch.ones_like(logits,dtype=torch.float32))
    ls=(source_bce*source*source_weight).sum()/source_denominator
    ln=(neg_bce*batch['weight'].detach()*internal).sum()/internal_denominator
    lsoft=(soft_bce*soft_mask).sum()/internal_denominator
    return dict(total=ls+pseudo_weight*ln+soft_weight*lsoft,source=ls,negative=ln,soft=lsoft,
        outside_unweighted=(source_bce*outside).sum()/source_denominator,
        outside_contribution=(source_bce*outside).sum()*alpha/source_denominator,
        outline_contribution=(source_bce*outline).sum()/source_denominator,
        internal=internal,soft_mask=soft_mask,outside=outside,outline=outline,
        source_denominator=source_denominator,internal_denominator=internal_denominator)

def losses(logits,batch,alpha,pseudo_weight=.25,soft_weight=1.,outline_target=1.):
    p=parts(logits,batch,alpha,pseudo_weight,soft_weight,outline_target)
    return p['total'],p['source'],p['negative'],p['soft'],p['internal'],p['soft_mask']
