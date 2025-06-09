"""
MagNet model implementation for earthquake magnitude estimation.

Reference:
    [1] Mousavi, S. M., & Beroza, G. C. (2020)
        A machine-learning approach for earthquake magnitude estimation.
        Geophysical Research Letters, 47, e2019GL085976.
        https://doi.org/10.1029/2019GL085976

Modules:
- _auto_pad_1d: Utility function for automatic padding in 1D convolutions.
- ConvBlock: 1D convolutional block with dropout and max pooling.
- MagNet: Complete model combining convolutional blocks, bidirectional LSTM, and linear output layer.
- magnet: Factory function to create MagNet instances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model


def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Automatically pads the input tensor on a specified dimension so that
    convolution with the given kernel size and stride produces an output
    with integer length.

    Args:
        x (torch.Tensor): Input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        dim (int, optional): Dimension along which to pad. Default is -1 (last dimension).
        padding_value (float, optional): Value to pad with. Default is 0.0.

    Returns:
        torch.Tensor: Padded input tensor.
    """
    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ConvBlock(nn.Module):
    """
    1D convolutional block with dropout and max pooling.

    Applies 1D convolution followed by dropout and max pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv_kernel_size (int): Kernel size of convolution.
        pool_kernel_size (int): Kernel size of max pooling.
        drop_rate (float): Dropout rate.

    Forward Input Shape:
        Tensor of shape (N, C_in, L)

    Forward Output Shape:
        Tensor of shape (N, C_out, L_out), where L_out depends on convolution and pooling.
    """

    def __init__(
        self, in_channels, out_channels, conv_kernel_size, pool_kernel_size, drop_rate
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
        )
        self.dropout = nn.Dropout(drop_rate)
        self.pool = nn.MaxPool1d(pool_kernel_size, ceil_mode=True)

    def forward(self, x):
        N, C, L = x.size()
        x = _auto_pad_1d(x, self.conv.kernel_size[0])
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class MagNet(nn.Module):
    """
    MagNet model for earthquake magnitude estimation combining convolutional layers
    and a bidirectional LSTM followed by a linear layer.

    Architecture:
        - Multiple ConvBlock layers.
        - Bidirectional LSTM layer.
        - Linear layer producing 2 output features.

    Args:
        in_channels (int): Number of input channels.
        conv_channels (list[int], optional): List specifying output channels for each ConvBlock.
            Default is [64, 32].
        lstm_dim (int, optional): Hidden size of the LSTM layer. Default is 100.
        drop_rate (float, optional): Dropout rate applied in ConvBlocks. Default is 0.2.

    Forward Input Shape:
        Tensor of shape (N, C_in, L)

    Forward Output Shape:
        Tensor of shape (N, 2), representing the output predictions.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: list = [64, 32],
        lstm_dim: int = 100,
        drop_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=inc,
                    out_channels=outc,
                    conv_kernel_size=3,
                    pool_kernel_size=4,
                    drop_rate=drop_rate,
                )
                for inc, outc in zip([in_channels] + conv_channels[:-1], conv_channels)
            ]
        )

        self.lstm = nn.LSTM(
            conv_channels[-1],
            lstm_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lin = nn.Linear(in_features=lstm_dim * 2, out_features=2)

    def forward(self, x):
        x = self.conv_layers(x)
        hs, (h, c) = self.lstm(x.transpose(-1, -2))
        h = h.transpose(0, 1).flatten(1)
        out = self.lin(h)

        return out


@register_model
def magnet(**kwargs):
    """
    Factory function to create a MagNet model instance.

    Args:
        kwargs: Keyword arguments passed to the MagNet constructor.

    Returns:
        MagNet: Instantiated MagNet model.
    """
    model = MagNet(**kwargs)
    return model
