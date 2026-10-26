# -*- coding: utf-8 -*-
"""
Model: ResNet50 stem + layer1 (ImageNet pre-trained) + a minimal U-head.

Structure (fixed by the design doc, section 2)
---------------------------------------------
    input RGB 512^2, already ImageNet-normalised by the Dataset
      conv1   Conv2d(3->64, k=7, s=2, p=3, bias=False)   -> H/2
      bn1 + relu                                          -> H/2, 64ch   (skip, f1)
      maxpool MaxPool2d(k=3, s=2, p=1)                    -> H/4
      layer1  3 x Bottleneck(64->64->256, s=1)            -> H/4, 256ch  (f2)
      head    conv_in 2xConv(256->64 / 64->64, 3x3) + ReLU
              nearest x2 -> H/2, concat f1 (64ch)
              refine1 Conv(128->64) + Conv(64->64), 3x3, + ReLU
              nearest x2 -> H
              refine2 Conv(64->64, 3x3) + ReLU + Conv(64->1, 1x1)  -> logits

Why the modules are built by hand
---------------------------------
`torchvision.models.resnet50(weights=None)` instantiates the whole 23.5M network and
then throws most of it away. We only need conv1/bn1/maxpool/layer1, so the stem and
layer1 are assembled from the *official* torchvision `Bottleneck` block and the
ImageNet weights are loaded by key. The resulting parameter names are byte-identical
to the torchvision checkpoint, which is why a plain `load_state_dict` works.

The head follows the design doc literally; no 7x7 kernels, two nearest-neighbour x2
upsamples. Note the finest feature grid is H/4 = 4 px at 512^2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck


class GapHead(nn.Module):
    """Minimal U-head: H/4 main feature + H/2 skip, nearest x2 twice, 1-ch logits."""

    def __init__(self, in_channels=256, skip_channels=64, hidden=64):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(hidden + skip_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feat, skip):
        x = self.conv_in(feat)                                  # H/4
        x = F.interpolate(x, scale_factor=2, mode="nearest")    # H/2
        x = torch.cat([x, skip], dim=1)                         # H/2
        x = self.refine1(x)                                     # H/2
        x = F.interpolate(x, scale_factor=2, mode="nearest")    # H
        return self.refine2(x)                                  # (B,1,H,W)


class BSENet(nn.Module):
    """Stem + layer1 + GapHead. Output = raw logits at full resolution."""

    def __init__(self, weights_path, hidden=64, verbose=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer1()
        self.head = GapHead(in_channels=256, skip_channels=64, hidden=hidden)

        self._load_backbone(weights_path, verbose=verbose)

    @staticmethod
    def _make_layer1():
        """torchvision ResNet50 `_make_layer(Bottleneck, 64, 3, stride=1)`.

        Block 0 needs a downsample because inplanes(64) != planes*expansion(256);
        blocks 1-2 are identity shortcuts. Parameter names match the checkpoint.
        """
        bn = nn.BatchNorm2d
        ds = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False), bn(256))
        return nn.Sequential(
            Bottleneck(64, 64, 1, ds, norm_layer=bn),
            Bottleneck(256, 64, 1, None, norm_layer=bn),
            Bottleneck(256, 64, 1, None, norm_layer=bn),
        )

    def _load_backbone(self, weights_path, verbose=True):
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        keep = {k: v for k, v in sd.items() if k.startswith(("conv1.", "bn1.", "layer1."))}
        missing, unexpected = self.load_state_dict(keep, strict=False)
        bb_missing = [k for k in missing if k.split(".")[0] in ("conv1", "bn1", "layer1")]
        if bb_missing:
            raise RuntimeError("backbone keys not matched: %s" % bb_missing[:8])
        if verbose:
            print("[model] loaded %d checkpoint keys (conv1/bn1/layer1)" % len(keep))
            print("[model] unloaded keys are head-only: %d" % len(missing))

    def backbone_parameters(self):
        mods = (self.conv1, self.bn1, self.layer1)
        return (p for m in mods for p in m.parameters())

    def forward(self, x):
        f1 = self.relu(self.bn1(self.conv1(x)))     # H/2, 64ch
        f2 = self.layer1(self.maxpool(f1))          # H/4, 256ch
        return self.head(f2, f1)                    # (B,1,H,W) logits


def build_model(weights_path, hidden=64, verbose=True):
    return BSENet(weights_path, hidden=hidden, verbose=verbose)


def param_report(model, assert_backbone=225344, total_ref=None, verbose=True):
    """Hard-assert the backbone count; report the rest (actual wins)."""
    n_bb = sum(p.numel() for p in model.backbone_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    n_head = n_all - n_bb
    if verbose:
        print("[params] backbone (conv1+bn1+layer1) = %d  (expected %d)"
              % (n_bb, assert_backbone))
        print("[params] head                          = %d" % n_head)
        print("[params] total                         = %d  (%.3f M)" % (n_all, n_all / 1e6))
        if total_ref is not None and n_all != total_ref:
            print("[params] NOTE: total differs from the reference figure %d "
                  "(difference %+d) -- actual wins, not a failure."
                  % (total_ref, n_all - total_ref))
    if assert_backbone is not None and n_bb != assert_backbone:
        raise AssertionError("backbone params %d != expected %d" % (n_bb, assert_backbone))
    return dict(backbone=n_bb, head=n_head, total=n_all)
