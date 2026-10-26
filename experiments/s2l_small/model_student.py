"""Original student architecture and weights, with explicit decoder interpolation.

Interpolation is not encoded in a state_dict. Training checkpoints must carry
model_config and be read through this experiment's run_pair.load_model.
"""
import hashlib
import importlib.util
import sys

import segmentation_models_pytorch as smp
import torch

import cfg_standard as C

_source = C.MODEL_DIR / 'model_unet.py'
_spec = importlib.util.spec_from_file_location('bilinear_student_reference', _source)
_reference = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _reference
_spec.loader.exec_module(_reference)


def set_interpolation(net, mode):
    if mode not in ('nearest', 'bilinear'):
        raise ValueError('Unsupported decoder interpolation: ' + mode)
    blocks = list(net.net.decoder.blocks)
    if len(blocks) != 5 or not all(hasattr(b, 'interpolation_mode') for b in blocks):
        raise RuntimeError('Expected five SMP UNet interpolation blocks')
    for block in blocks:
        block.interpolation_mode = mode
    if net.net.encoder.output_stride != 32:
        raise RuntimeError('Unexpected encoder stride')
    return net


def build_model(weights_path, verbose=True):
    net = _reference.build_model(weights_path, verbose=verbose)
    return set_interpolation(net, C.DECODER_INTERPOLATION)


def model_config(net):
    # SMALL teacher is constructed separately and retains its original head.
    if not hasattr(net, 'net') or not hasattr(net.net, 'decoder'):
        return None
    modes = [b.interpolation_mode for b in net.net.decoder.blocks]
    if len(modes) != 5 or len(set(modes)) != 1:
        raise RuntimeError('Mixed or missing decoder interpolation modes')
    return dict(architecture='smp.Unet', encoder='resnet50',
                encoder_stride=32, decoder_interpolation=modes[0],
                decoder_blocks=5, align_corners=False if modes[0] == 'bilinear' else None,
                reference_model_sha256=hashlib.sha256(_source.read_bytes()).hexdigest(),
                smp_version=str(smp.__version__), torch_version=str(torch.__version__))
