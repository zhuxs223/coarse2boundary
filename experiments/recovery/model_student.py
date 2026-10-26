"""Student selection; SMALL H/4 remains a separate initial teacher."""
import hashlib
from pathlib import Path

import torch

import cfg_standard as C
import model_h8 as H
import model_student_unet as U


def build_model(weights_path, verbose=True):
    if C.STUDENT_ARCH == 'unet':
        return U.build_model(weights_path, verbose=verbose)
    net = H.SmallH8(weights_path)
    if verbose:
        print('[student] H/8: ImageNet stem/layer1/layer2, fresh nearest U-head', flush=True)
    return net


def model_config(net):
    if isinstance(net, H.SmallH8):
        here = Path(__file__).resolve().parent
        h = hashlib.sha256()
        for name in ('model_h8.py', 'model_reference.py'):
            h.update(name.encode())
            h.update((here/name).read_bytes())
        return dict(
            architecture='SMALL.H8', encoder='resnet50.stem.layer1.layer2',
            encoder_stride=8, decoder_interpolation='nearest', decoder_blocks=3,
            align_corners=None, reference_model_sha256=h.hexdigest(),
            h4_fusion='concat(f4[256], upsample(h8_in(f8))[64]) -> 64',
            skips=['H/4', 'H/2'], new_h8_fusion_init='zero',
            parameters=sum(p.numel() for p in net.parameters()),
            torch_version=str(torch.__version__))
    return U.model_config(net)
