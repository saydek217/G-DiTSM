import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm

def get_sinusoid_encoding_table(n_pos: int, dim: int, device='cpu'):
    """ Sinusoidal positional encoding, like Vaswani et al. """
    def angle(pos, i):
        return pos / (10000 ** (2 * (i // 2) / dim))
    table = torch.zeros(n_pos, dim, device=device)
    for pos in range(n_pos):
        for i in range(dim):
            a = angle(pos, i)
            if i % 2 == 0:
                table[pos, i] = math.sin(a)
            else:
                table[pos, i] = math.cos(a)
    return table  # [n_pos, dim]

class discshift(nn.Module):
    """
    Enhanced D-TSM with:
      1) sinusoidal positional encoding
      2) discriminative forward/backward shifts
      3) temporal multi-head self-attention with LayerNorm + FFN
      4) gated residual fusion (fixed!)
    Wraps a backbone that expects (N*T, C, H, W) input.
    """
    def __init__(self,
                 net: nn.Module,
                 n_segment: int = 8,
                 n_div: int = 8,
                 attn_heads: int = 2):
        super().__init__()
        self.net        = net
        self.n_segment  = n_segment
        self.fold_div   = n_div
        self.attn_heads = attn_heads

        # lazy modules
        self._pos_table = None
        self._mha       = None
        self._norm      = None
        self._ffn       = None

    def forward(self, x):
        # x: (N*T, C, H, W)
        N_T, C, H, W = x.shape
        N = N_T // self.n_segment
        x = x.view(N, self.n_segment, C, H, W)

        # 1) positional encoding
        if (self._pos_table is None
            or self._pos_table.shape[0] != self.n_segment
            or self._pos_table.shape[1] != C):
            self._pos_table = get_sinusoid_encoding_table(
                self.n_segment, C, device=x.device)
        pos = self._pos_table.view(1, self.n_segment, C, 1, 1)
        x = x + pos

        # 2) discriminative shift
        fold = C // self.fold_div
        y = x.clone()
        # forward diff
        y[:, :-1, :fold]      = x[:, 1:, :fold] - x[:, :-1, :fold]
        # backward diff
        y[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold] - x[:, 1:, fold:2*fold]
        # rest untouched
        y[:, :, 2*fold:]      = x[:, :, 2*fold:]

        # 3) temporal self-attention + FFN
        # lazy init of MHA, LayerNorm, and FFN
        if self._mha is None:
            self._mha = nn.MultiheadAttention(
                embed_dim=C,
                num_heads=self.attn_heads,
                batch_first=True
            ).to(x.device)
        if self._norm is None:
            self._norm = LayerNorm(C).to(x.device)
        if self._ffn is None:
            self._ffn = nn.Sequential(
                nn.Linear(C, C * 4),
                nn.ReLU(inplace=True),
                nn.Linear(C * 4, C),
            ).to(x.device)

        # flatten spatial dims → tokens of shape (S, T, C)
        # where S = N * H * W
        y_tok = y.permute(0, 3, 4, 1, 2).reshape(N * H * W,
                                               self.n_segment,
                                               C)
        # normalize before attention
        y_tok = self._norm(y_tok)
        attn_out, _ = self._mha(y_tok, y_tok, y_tok)
        # small feed-forward + residual
        ffn_out = self._ffn(attn_out)
        attn_out = attn_out + ffn_out

        # reshape back to (N, T, C, H, W)
        y = attn_out.reshape(N, H, W, self.n_segment, C).permute(
            0, 3, 4, 1, 2)

        # 4) gated residual fusion (fixed!)
        gate = y.mean(dim=[3, 4], keepdim=True).sigmoid()  # [N,T,C,1,1]
        y = x + y * gate

        # flatten back to (N*T, C, H, W) and feed into backbone
        y = y.reshape(N * self.n_segment, C, H, W)
        return self.net(y)


if __name__ == "__main__":
    # toy test
    backbone = nn.Identity()
    block    = DiscShift(backbone, n_segment=3, n_div=8, attn_heads=2)

    inp = torch.randn(6, 16, 4, 4)   # 2 clips × 3 frames, C=16, H=W=4
    out = block(inp)
    assert out.shape == inp.shape
    print("✅ DiscShift forward OK:", out.shape)