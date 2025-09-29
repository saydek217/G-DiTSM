import torch
import torch.nn as nn
import torch.nn.functional as F
from math import gcd
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import InvertedResidual
import math


class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super().__init__()
        # choose number of groups as gcd(channels, factor)
        self.groups = gcd(channels, factor)
        assert channels % self.groups == 0, "channels must be divisible by groups"
        per_ch = channels // self.groups

        self.agp     = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h  = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w  = nn.AdaptiveAvgPool2d((1, None))
        self.softmax = nn.Softmax(-1)
        self.gn      = nn.GroupNorm(self.groups, channels)

        # these convs act on each group slice of size per_ch
        self.conv1x1 = nn.Conv2d(per_ch, per_ch, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(per_ch, per_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        G = self.groups

        # reshape into G groups: [B·G, C/G, H, W]
        xg = x.view(B * G, C // G, H, W)

        # height & width summaries
        xh = self.pool_h(xg)                  # [B·G, C/G, H, 1]
        xw = self.pool_w(xg).permute(0,1,3,2) # [B·G, C/G, 1, W]
        hw = torch.cat([xh, xw], dim=2)       # [B·G, C/G, H+W, 1]
        hw = self.conv1x1(hw)                 # [B·G, C/G, H+W, 1]
        xh_new, xw_new = torch.split(hw, [H, W], dim=2)
        xw_new = xw_new.permute(0,1,3,2)      # back to [B·G, C/G, 1, W]

        # spatial gating branch
        x1 = xg * torch.sigmoid(xh_new) * torch.sigmoid(xw_new)
        x2 = self.conv3x3(xg)

        # channel‐wise pooling + softmax
        a1 = self.softmax(self.agp(x1).view(B*G, -1, 1).permute(0,2,1))  # [B·G,1,H·W]
        a2 = self.softmax(self.agp(x2).view(B*G, -1, 1).permute(0,2,1))

        w1 = a1 @ x2.view(B*G, C//G, -1)  # [B·G,1,H·W]
        w2 = a2 @ x1.view(B*G, C//G, -1)

        attn_map = (w1 + w2).view(B*G, 1, H, W)    # [B·G,1,H,W]
        out = xg * torch.sigmoid(attn_map)

        # restore original shape
        return out.view(B, C, H, W)


class DepthECAmega(nn.Module):
    """
    Depth‐enhanced ECA mega‐block:
      – DW‐3×3 → GN → SiLU
      – (opt) ECA‐1D conv channel branch
      – (opt) SE tiny MLP channel branch
      – fuse channel attentions, apply sigmoid
      – (opt) EMA spatial gate
      – residual gating: y = x + x*channel_attn, then spatial refinement
    """
    def __init__(
        self,
        channels,
        gn_groups: int = 16,
        se_reduction: int = 4,
        use_eca: bool = True,
        use_se: bool  = True,
        use_ema: bool = True,
        ema_groups:  int  = 32
    ):
        super().__init__()
        self.use_eca = use_eca
        self.use_se  = use_se
        self.use_ema = use_ema

        # spatial context
        self.dw   = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=False)
        self.norm = nn.GroupNorm(gcd(channels, gn_groups), channels)
        self.act  = nn.SiLU(inplace=True)

        # channel branches
        if self.use_eca:
            # e.g. k = nearest odd to |log2(C)/γ + b|
            t = max(1, int(abs(math.log2(channels)/2 + 1)))
            k = t if t % 2 else t+1
            self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)

        # residual scale
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        if self.use_se:
            hidden = max(1, channels // se_reduction)
            self.se_fc = nn.Sequential(
                nn.Linear(channels, hidden, bias=False),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, channels, bias=False),
            )

        # spatial branch
        if self.use_ema:
            self.ema = EMA(channels, factor=ema_groups)

    def forward(self, x):
        # 1) spatial‐DW context
        y = self.dw(x)
        y = self.norm(y)
        y = self.act(y)

        # 2) global pooling for channel branches
        gp = F.adaptive_avg_pool2d(y, 1).flatten(1)  # [B, C]

        # 3) accumulate ECA + SE
        attn = torch.zeros_like(gp)
        if self.use_eca:
            eca = self.eca_conv(gp.unsqueeze(1)).squeeze(1)  # [B,C]
            attn = attn + eca
        if self.use_se:
            se = self.se_fc(gp)                               # [B,C]
            attn = attn + se

        # 4) channel‐wise sigmoid
        attn = torch.sigmoid(attn).view(x.size(0), x.size(1), 1, 1)

        # 5) residual channel gating
        out = x + self.res_scale * x * attn

        # 6) spatial refinement
        if self.use_ema:
            out = self.ema(out)

        return out


class MobileNetV3_DepthECAmega(nn.Module):
    """
    MobileNetV3-Small backbone + DepthECAmega blocks after each InvertedResidual.
    """
    def __init__(
        self,
        pretrained: bool = True,
        use_eca:  bool  = True,
        use_se:   bool  = True,
        use_ema:  bool  = True
    ):
        super().__init__()
        # 1) load base MobileNetV3-Small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base    = mobilenet_v3_small(weights=weights)

        # 2) rebuild features + inject mega-block
        feats = []
        for layer in base.features:
            feats.append(layer)
            if isinstance(layer, InvertedResidual) and layer.use_res_connect:
                feats.append(
                    DepthECAmega(
                        channels    = layer.out_channels,
                        use_eca     = use_eca,
                        use_se      = use_se,
                        use_ema     = use_ema
                    )
                )
        self.features = nn.Sequential(*feats)

        # 3) preserve global pooling
        self.pool = base.avgpool

        # 4) classifier stub (TSN will overwrite)
        self.last_channel = base.classifier[0].in_features
        self.classifier   = nn.Linear(self.last_channel, self.last_channel)

        # TSN metadata
        self.last_layer_name = 'classifier'
        self.input_size      = 224
        self.input_mean      = [0.485, 0.456, 0.406]
        self.input_std       = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)             # [B, C, H', W']
        x = self.pool(x)                 # [B, C, 1, 1]
        x = torch.flatten(x, 1)          # [B, C]
        x = self.classifier(x)           # [B, C]
        return x


def mobilenet_v3_deptheca_mega(
    pretrained: bool = True,
    use_eca:   bool = True,
    use_se:    bool = True,
    use_ema:   bool = True
) -> MobileNetV3_DepthECAmega:
    return MobileNetV3_DepthECAmega(
        pretrained=pretrained,
        use_eca=use_eca,
        use_se=use_se,
        use_ema=use_ema
    )


# quick sanity check
if __name__ == "__main__":
    m = mobilenet_v3_deptheca_mega(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("✅ Forward OK, output shape:", y.shape)  # should be [2, m.last_channel]
