# archs/mobilenet_v3_deptheca.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import InvertedResidual


class DepthECAplus(nn.Module):
    """
    Depthwise Enhanced Channel Attention (DepthECA+):
      1) depthwise 3×3 conv → GroupNorm → SiLU
      2) dual‐branch channel MLP:
         • ECA‐style conv1d on pooled features
         • tiny SE‐MLP on pooled features
      3) fuse both attentions, sigmoid, then residual gate
    """
    def __init__(self, channels, gn_groups=16, se_reduction=4):
        super().__init__()
        # spatial context
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        g = gcd(channels, gn_groups)
        self.norm = nn.GroupNorm(g, channels)
        self.act  = nn.SiLU(inplace=True)

        # ECA branch (1D conv over channel axis)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # SE branch (tiny MLP)
        hidden = max(1, channels // se_reduction)
        self.se_fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        # x: [B,C,H,W]
        y = self.dw(x)
        y = self.norm(y)
        y = self.act(y)

        # global pooling → [B, C]
        gp = F.adaptive_avg_pool2d(y, 1).flatten(1)

        # --- ECA branch ---
        # [B, C, 1,1] → [B,1,C] → conv1d → [B,1,C] → back to [B,C]
        eca = gp.unsqueeze(1)              # [B,1,C]
        eca = self.eca_conv(eca)           # [B,1,C]
        eca = eca.squeeze(1)               # [B,C]

        # --- SE branch ---
        se  = self.se_fc(gp)               # [B,C]

        # fuse & gate
        attn = torch.sigmoid(eca + se)     # [B,C]
        attn = attn.view(x.size(0), x.size(1), 1, 1)

        # residual gating
        return x + x * attn


class MobileNetV3_DepthECA(nn.Module):
    """
    MobileNetV3-Small backbone with DepthECA+ after every residual block.
    For TSN, TSN._prepare_base_model will detect
      model.last_layer_name = 'classifier'
    and replace that layer with its own dropout+fc.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # load vanilla MobileNetV3-Small features
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base    = mobilenet_v3_small(weights=weights)

        # rebuild features + inject DepthECA+ after each InvertedResidual
        feats = []
        for layer in base.features:
            feats.append(layer)
            if isinstance(layer, InvertedResidual) and layer.use_res_connect:
                feats.append(DepthECAplus(layer.out_channels))
        self.features = nn.Sequential(*feats)

        # keep original global pooling
        self.pool = base.avgpool

        # TSN will replace this classifier, but we need its in_features
        self.last_channel = base.classifier[0].in_features  # typically 576
        self.classifier  = nn.Linear(self.last_channel, self.last_channel)

        # TSN metadata
        self.last_layer_name = 'classifier'
        self.input_size      = 224
        self.input_mean      = [0.485, 0.456, 0.406]
        self.input_std       = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)         # [B, C, H', W']
        x = self.pool(x)             # [B, C, 1, 1]
        x = torch.flatten(x, 1)      # [B, C]
        x = self.classifier(x)       # [B, C]
        return x


def mobilenet_v3_deptheca(pretrained: bool = True) -> MobileNetV3_DepthECA:
    """
    Factory: returns a MobileNetV3-Small + DepthECA+ backbone.
    TSN will pick up last_layer_name='classifier' and replace that layer.
    """
    return MobileNetV3_DepthECA(pretrained=pretrained)
