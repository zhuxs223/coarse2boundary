# -*- coding: utf-8 -*-
"""
The 32.52 M comparison network: `smp.Unet` with a full ResNet50 encoder.

Why this file exists
--------------------
The small model (0.558 M) uses only ResNet50's stem + layer1. To decide whether the
"outline-only supervision still produces interior ridge responses" result is a property
of the supervision or just of a tiny network, the same experiment is repeated with a
full UNet -- the architecture the previous self-training line used, at 32.52 M params.

Note both backbones are the SAME ResNet50 and both are ImageNet-pretrained, so this is
a clean capacity comparison rather than a change of family.

Construction matches the old pipeline exactly
---------------------------------------------
    smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    then the local ImageNet checkpoint is loaded into `encoder.*` by key.

Two deliberate differences from the old pipeline:
  * normalisation follows `cfg.NORM_MODE` (default ImageNet). The old UNet script only
    did `img / 255`, which the design doc records as a missed step; with an
    ImageNet-pretrained encoder the matched statistics are the ImageNet ones.
  * everything else (supervision construction, loss, epochs, lr schedule, evaluation
    module) comes from the shared cfg/assess of this folder, so the two architectures
    are compared under identical settings.
"""

import torch
import torch.nn as nn


class UNetNet(nn.Module):
    """smp.Unet(resnet50) with the ImageNet encoder weights installed by key."""

    def __init__(self, weights_path, verbose=True):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                            in_channels=3, classes=1)
        self._load_encoder(weights_path, verbose=verbose)

    def _load_encoder(self, weights_path, verbose=True):
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        enc = {"encoder." + k: v for k, v in sd.items()}
        res = self.net.load_state_dict(enc, strict=False)
        enc_missing = [k for k in res.missing_keys if k.startswith("encoder.")]
        if enc_missing:
            raise RuntimeError("encoder keys not matched: %s" % enc_missing[:8])
        if verbose:
            other_missing = [k for k in res.missing_keys if not k.startswith("encoder.")]
            print("[model-unet] encoder: %d keys loaded, 0 missing" % len(enc))
            print("[model-unet] decoder/head left at their own init: %d keys "
                  "(expected -- they are random in the old pipeline too)"
                  % len(other_missing))
            if res.unexpected_keys:
                print("[model-unet] ignored checkpoint keys: %s" % list(res.unexpected_keys))

    def backbone_parameters(self):
        return self.net.encoder.parameters()

    def forward(self, x):
        return self.net(x)


def build_model(weights_path, verbose=True):
    return UNetNet(weights_path, verbose=verbose)


def param_report(model, verbose=True):
    """No hard assertion here: the reference figure is a report, not a contract."""
    n_enc = sum(p.numel() for p in model.backbone_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    if verbose:
        print("[params] encoder (resnet50) = %d  (%.3f M)" % (n_enc, n_enc / 1e6))
        print("[params] decoder + head     = %d" % (n_all - n_enc))
        print("[params] total               = %d  (%.3f M)" % (n_all, n_all / 1e6))
        print("[params] reference figure    = 32521105 (32.521 M) -- old pipeline's UNet")
    return dict(backbone=n_enc, head=n_all - n_enc, total=n_all)
