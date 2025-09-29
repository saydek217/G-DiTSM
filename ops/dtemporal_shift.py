import torch
import torch.nn as nn

class DiscriminativeTemporalShift(nn.Module):
    """
    Discriminative Temporal Shift Module (D-TSM).

    This module introduces discriminative temporal differences between adjacent frames
    enhancing the extraction of temporal dynamics, as described in the provided paper.
    """
    def __init__(self, net, n_segment=8, n_div=8):
        """
        Args:
            net (nn.Module): The base CNN (e.g., MobileNetV3) to be wrapped.
            n_segment (int): Number of temporal segments (frames) per input clip.
            n_div (int): Number of divisions for temporal shift operation.
        """
        super(DiscriminativeTemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        print(f'=> Using Discriminative TSM (D-TSM) with fold division: {self.fold_div}')

    def forward(self, x):
        x = self.discriminative_shift(x, self.n_segment, self.fold_div)
        return self.net(x)

    @staticmethod
    def discriminative_shift(x, n_segment, fold_div):
        """
        Apply discriminative temporal shifting to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size*n_segment, channels, height, width)
            n_segment (int): Number of temporal segments.
            fold_div (int): Number of divisions along the channel dimension.

        Returns:
            torch.Tensor: Output tensor after discriminative temporal shifting.
        """
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = x.clone()

        # Forward discriminative shift
        out[:, :-1, :fold] = x[:, 1:, :fold] - x[:, :-1, :fold]

        # Backward discriminative shift
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold] - x[:, 1:, fold: 2 * fold]

        # Remaining channels are static (unchanged)
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)


if __name__ == '__main__':
    # Example usage and test
    backbone = nn.Identity()  # Placeholder backbone, replace with actual backbone CNN
    dtsm = DiscriminativeTemporalShift(backbone, n_segment=3, n_div=8)

    x = torch.rand(6, 16, 4, 4)  # (batch_size*n_segment, channels, H, W)
    y = dtsm(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    print("Discriminative TSM test passed.")
