"""Scribble2Label: separate normalized source CE and confident-unlabeled CE."""
import torch
import torch.nn.functional as F
import cfg_standard as C


def labels(ema, internal):
    positive = internal & (ema > C.CONFIDENCE)
    negative = internal & (ema < 1 - C.CONFIDENCE)
    return positive, negative


def losses(logits, batch, pseudo_enabled):
    logits = logits.float()
    valid, mk3 = batch['valid'].bool(), batch['mk3']
    source = valid & (mk3 != 1)
    target = (mk3 == 2).float()
    source_loss = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * source).sum() / source.sum().clamp(min=1)
    pos = torch.zeros_like(valid)
    neg = torch.zeros_like(valid)
    if pseudo_enabled:
        if 'ema' not in batch:
            raise RuntimeError('Pseudo phase requires previous completed epoch EMA')
        pos, neg = labels(batch['ema'].float(), valid & (mk3 == 1))
    selected = pos | neg
    pseudo_loss = (F.binary_cross_entropy_with_logits(logits, pos.float(), reduction='none') * selected).sum() / selected.sum().clamp(min=1)
    return source_loss + C.PSEUDO_WEIGHT * pseudo_loss, source_loss, pseudo_loss, pos, neg
