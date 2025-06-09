"""
dist-PT network module.

This module implements the dist-PT model architecture for temporal convolutional
processing of 1D signals with causal convolutions, residual blocks, and dilation.

Reference:
    Mousavi, S. M. and Beroza, G. C. (2020).
    Bayesian-Deep-Learning Estimation of Earthquake Location From Single-Station Observations.
    IEEE Transactions on Geoscience and Remote Sensing, 58(11), 8211-8224.
    doi: 10.1109/TGRS.2020.2988770.

Classes:
    ResBlock: Residual block with two causal dilated convolutional layers.
    TemporalConvLayer: Stack of residual convolutional blocks with optional output sequence.
    DistPT_Network: Main dist-PT model combining temporal conv layers and linear outputs.

Functions:
    _causal_pad_1d: Applies causal padding to 1D inputs for causal convolutions.
    distpt_network: Factory function to instantiate DistPT_Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model


def _causal_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    dilation: int,
    dim: int = -1,
    padding_value: float = 0.0,
):
    """
    Apply causal padding to a 1D tensor for causal convolution.

    Args:
        x (torch.Tensor): Input tensor to pad.
        kernel_size (int): Kernel size of the convolution.
        stride (int): Stride of the convolution (must be 1).
        dilation (int): Dilation factor of the convolution.
        dim (int, optional): Dimension along which to pad. Defaults to -1.
        padding_value (float, optional): Constant padding value. Defaults to 0.0.

    Returns:
        torch.Tensor: Padded input tensor suitable for causal convolution.

    Raises:
        AssertionError: If stride is not equal to 1.
    """
    assert stride == 1

    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (kernel_size - 1) * dilation
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds, 0)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ResBlock(nn.Module):
    """
    Residual block containing two causal dilated convolutional layers with batch normalization,
    ReLU activation, dropout, and a 1x1 convolutional skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of convolutional layers.
        dilation (int): Dilation factor for convolutions.
        drop_rate (float): Dropout probability.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch, channels, length).

    Forward Output:
        Tuple[torch.Tensor, torch.Tensor]:
            - Output tensor after residual addition.
            - Shortcut tensor before addition (for skip connections).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, drop_rate):
        super().__init__()
        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn0 = nn.BatchNorm1d(out_channels)

        self.relu0 = nn.ReLU()

        self.dropout0 = nn.Dropout1d(drop_rate)

        self.conv1 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout1d(drop_rate)

        self.conv_out = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = _causal_pad_1d(
            x, self.conv0.kernel_size[0], self.conv0.stride[0], self.conv0.dilation[0]
        )
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.dropout0(x)

        x = _causal_pad_1d(
            x, self.conv1.kernel_size[0], self.conv1.stride[0], self.conv1.dilation[0]
        )
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x1 = x + self.conv_out(x)

        return x1, x


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer composed of an initial 1x1 convolution followed by
    multiple residual convolution blocks with specified dilations.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels for each conv block. Defaults to 64.
        kernel_size (int, optional): Kernel size of residual conv blocks. Defaults to 2.
        num_conv_blocks (int, optional): Number of convolution blocks to repeat. Defaults to 1.
        dilations (list, optional): List of dilation values to apply per block. Defaults to [1,2,4,8,16,32].
        drop_rate (float, optional): Dropout rate in residual blocks. Defaults to 0.0.
        return_sequences (bool, optional): Whether to return the full sequence or only last output. Defaults to False.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch, channels, length).

    Forward Output:
        torch.Tensor: Output tensor of shape
            (batch, out_channels, length) if return_sequences is True,
            else (batch, out_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_size: int = 2,
        num_conv_blocks: int = 1,
        dilations: list = [1, 2, 4, 8, 16, 32],
        drop_rate: float = 0.0,
        return_sequences: bool = False,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

        self.conv_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    drop_rate=drop_rate,
                )
                for dilation in dilations * num_conv_blocks
            ]
        )

        self.return_sequences = return_sequences

    def forward(self, x):
        x = self.conv_in(x)

        shortcuts = []
        for conv in self.conv_blocks:
            x, sc = conv(x)
            shortcuts.append(sc)

        x = sum(shortcuts)

        if not self.return_sequences:
            x = x[:, :, -1]

        return x


class DistPT_Network(nn.Module):
    """
    dist-PT network model that stacks temporal convolutional layers and outputs
    predictions for distance and probability.

    Args:
        in_channels (int): Number of input channels.
        tcn_channels (int, optional): Number of channels in temporal convolution layers. Defaults to 20.
        kernel_size (int, optional): Kernel size of temporal convolution layers. Defaults to 6.
        num_conv_blocks (int, optional): Number of convolution blocks per TCN layer. Defaults to 1.
        dilations (list, optional): List of dilation factors. Defaults to powers of two from 1 to 1024.
        drop_rate (float, optional): Dropout rate. Defaults to 0.1.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch, channels, length).

    Forward Output:
        Tuple[torch.Tensor, torch.Tensor]:
            - Distance output tensor of shape (batch, 2).
            - Probability output tensor of shape (batch, 2).
    """

    def __init__(
        self,
        in_channels: int,
        tcn_channels: int = 20,
        kernel_size: int = 6,
        num_conv_blocks: int = 1,
        dilations: list = [2**i for i in range(11)],
        drop_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.tcn = TemporalConvLayer(
            in_channels=in_channels,
            out_channels=tcn_channels,
            kernel_size=kernel_size,
            num_conv_blocks=num_conv_blocks,
            dilations=dilations,
            drop_rate=drop_rate,
        )

        self.lin_dist = nn.Linear(in_features=tcn_channels, out_features=2)
        self.lin_ptrvl = nn.Linear(in_features=tcn_channels, out_features=2)

    def forward(self, x):
        x = self.tcn(x)

        do = self.lin_dist(x)
        po = self.lin_ptrvl(x)

        return do, po


@register_model
def distpt_network(**kwargs):
    """
    Factory function to create a DistPT_Network model.

    Args:
        **kwargs: Arguments passed to DistPT_Network constructor.

    Returns:
        DistPT_Network: Instantiated dist-PT network model.
    """
    model = DistPT_Network(**kwargs)
    return model
