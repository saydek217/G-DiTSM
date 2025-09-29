import torch
import torch.nn as nn
from torch.nn.init import constant_
import torch.nn.functional as F

class GatedDTSM(nn.Module):
    """
    Gated Discriminative Temporal Shift Module (G-DTSM).
      1) Discriminative differences (D-TSM)
      2) Depthwise 3×1×1 temporal conv (initialized to zero)
      3) BatchNorm3d
      4) SE-style channel gating (gate bias = –3)
      5) Gated residual fusion
    Wraps a 2D module expecting (N*T, C, H, W) input.
    """
    def __init__(self,
                 net: nn.Module,
                 channels: int,
                 n_segment: int = 8,
                 n_div: int = 8,
                 reduction: int = 4):
        super().__init__()
        self.net       = net
        self.n_segment = n_segment
        self.fold_div  = n_div
        self.reduction = reduction

        # 1) Depthwise temporal conv
        self.dw_temporal = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3,1,1),
            padding=(1,0,0),
            groups=channels,
            bias=False
        )
        # init as zero → identity in time
        constant_(self.dw_temporal.weight, 0)

        # 2) Norm
        self.bn3d = nn.BatchNorm3d(channels)

        # 3) SE-style gating FCs
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        # gate starts nearly closed (sigmoid(-3)≈0.05)
        constant_(self.fc2.bias, -3.0)

    def forward(self, x):
        # x: (N*T, C, H, W)
        nt, c, h, w = x.shape
        n = nt // self.n_segment

        # reshape to (N, T, C, H, W)
        x = x.view(n, self.n_segment, c, h, w)

        # 1) discriminative shift
        fold = c // self.fold_div
        y = x.clone()
        y[:, :-1, :fold]      = x[:, 1:, :fold] - x[:, :-1, :fold]
        y[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold] - x[:, 1:, fold:2*fold]
        y[:, :, 2*fold:]      = x[:, :, 2*fold:]

        # 2) depthwise temporal conv + BN
        #    need (N, C, T, H, W)
        y = y.permute(0,2,1,3,4)        # → (N, C, T, H, W)
        y = self.dw_temporal(y)
        y = self.bn3d(y)
        y = y.permute(0,2,1,3,4)        # → (N, T, C, H, W)

        # 3) SE gating
        #    squeeze over (T, H, W) → (N, C)
        squeeze = y.mean(dim=(1,3,4))
        z = self.fc1(squeeze)
        z = F.relu(z, inplace=True)
        z = self.fc2(z)
        gate = torch.sigmoid(z).view(n, 1, c, 1, 1)

        # 4) gated fusion + flatten back
        out = x + y * gate
        out = out.view(nt, c, h, w)

        return self.net(out)


# Example sanity‐check
if __name__ == "__main__":
    C = 32
    backbone = nn.Identity()
    module = GatedDTSM(backbone, channels=C, n_segment=8, n_div=8, reduction=4)
    inp = torch.randn(4*8, C, 7, 7)  # 4 clips × 8 frames
    out = module(inp)
    print("In:", inp.shape, "Out:", out.shape)
