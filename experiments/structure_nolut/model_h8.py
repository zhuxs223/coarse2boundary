"""SMALL extended to ResNet50 layer2 (H/8), with a matching U-shaped decoder.

ImageNet stem/layer1/layer2; H/8 -> H/4 concat -> H/2 concat -> H.
No attention. All upsampling is nearest. Original SMALL weights are retained.
The added H/8 inputs to the H/4 fusion convolution start at zero, so the
initial function matches the original SMALL up to floating-point roundoff.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

import model_reference as M


class H8Head(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.conv_in = original.conv_in
        self.refine1 = original.refine1
        self.refine2 = original.refine2
        old_conv = self.conv_in[0]
        fusion = nn.Conv2d(256+64, 64, 3, padding=1)
        with torch.no_grad():
            fusion.weight.zero_()
            fusion.weight[:, :256].copy_(old_conv.weight)
            fusion.bias.copy_(old_conv.bias)
        self.conv_in[0] = fusion
        self.h8_in = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, f4, f8, f2):
        x = self.h8_in(f8)
        x = F.interpolate(x, size=f4.shape[-2:], mode='nearest')
        # Original H/4 channels come first, matching the copied old weights.
        x = self.conv_in(torch.cat([f4, x], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.refine1(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.refine2(x)


class SmallH8(M.BSENet):
    def __init__(self, weights_path):
        super().__init__(str(weights_path), verbose=False)
        # torchvision ResNet50-v1.5 layer2: four Bottleneck blocks, with
        # stride2 in the first block's 3x3 convolution and shortcut.
        downsample = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, stride=2, downsample=downsample),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
        )
        state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
        layer2_state = {k[len('layer2.'):]: v for k, v in state.items()
                        if k.startswith('layer2.')}
        self.layer2.load_state_dict(layer2_state, strict=True)
        self.head = H8Head(self.head)

    def backbone_parameters(self):
        modules = (self.conv1, self.bn1, self.layer1, self.layer2)
        return (p for module in modules for p in module.parameters())

    def forward(self, x):
        f2 = self.relu(self.bn1(self.conv1(x)))
        f4 = self.layer1(self.maxpool(f2))
        f8 = self.layer2(f4)
        return self.head(f4, f8, f2)


def original_state_dict(model):
    original = {}
    for key, value in model.state_dict().items():
        if key.startswith(('layer2.', 'head.h8_in.')):
            continue
        if key == 'head.conv_in.0.weight':
            value = value[:, :256]
        original[key] = value
    return original
