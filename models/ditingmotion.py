"""
DiTingMotion: Deep Learning Model for First-Motion Polarity Classification

This module provides an implementation of the DiTingMotion model introduced by Zhao et al. (2023),
a deep-learning approach for classifying first-motion polarity and clarity from 1D seismic signals.
The model architecture combines multiple convolutional layers with varied kernel sizes, auxiliary
side layers for intermediate predictions, and fusion layers for final classification outputs.

Main components include:
- CombConvLayer: Parallel 1D convolutional layers with different kernel sizes concatenated with input.
- BasicBlock: Sequential CombConvLayers followed by max pooling and residual concatenation.
- SideLayer: Auxiliary branches with convolution and fully connected layers for clarity and polarity classification.
- DiTingMotion: Complete model integrating blocks and side layers to predict both polarity and clarity.

This code is built using PyTorch and is intended for seismic signal classification tasks,
particularly in focal mechanism inversion applications.

Reference:
    Zhao M, Xiao Z, Zhang M, Yang Y, Tang L, Chen S. (2023)
    DiTingMotion: A deep-learning first-motion-polarity classifier and its application to focal mechanism inversion.
    Frontiers in Earth Science, 11:1103914.
    https://doi.org/10.3389/feart.2023.1103914
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
    Automatically pad a 1D tensor along a specified dimension for convolution.

    Pads the input tensor so that after convolution with given kernel size and stride,
    the output length remains consistent.

    Args:
        x (torch.Tensor): Input tensor.
        kernel_size (int): Convolution kernel size.
        stride (int, optional): Stride for convolution. Default is 1.
        dim (int, optional): Dimension to pad. Default is -1 (last dimension).
        padding_value (float, optional): Padding value. Default is 0.0.

    Returns:
        torch.Tensor: Padded tensor.

    Raises:
        AssertionError: If kernel_size is less than stride.
    """

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class CombConvLayer(nn.Module):
    """
    Combination convolution layer consisting of parallel 1D convolutions with different kernel sizes,
    concatenated with the input, followed by dropout, a final convolution, and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for each parallel convolution.
        kernel_sizes (list[int]): List of kernel sizes for parallel convolutions.
        out_kernel_size (int): Kernel size for the final output convolution.
        drop_rate (float): Dropout rate applied after concatenation.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C_in, L), where
            N is the batch size,
            C_in is the number of input channels,
            L is the length of the input sequence.

    Forward Output:
        torch.Tensor: Output tensor after combined convolutions,
            with shape (N, out_channels, L_out), where
            L_out depends on padding and kernel sizes.
    """

    def __init__(
        self, in_channels, out_channels, kernel_sizes, out_kernel_size, drop_rate
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kers,
                    ),
                    nn.ReLU(),
                )
                for kers in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(drop_rate)
        self.out_conv = nn.Conv1d(
            in_channels=(in_channels + len(kernel_sizes) * out_channels),
            out_channels=out_channels,
            kernel_size=out_kernel_size,
        )
        self.out_relu = nn.ReLU()

    def forward(self, x):
        outs = [x]
        for conv_relu in self.convs:
            xi = _auto_pad_1d(x, conv_relu[0].kernel_size[0])
            xi = conv_relu(xi)
            outs.append(xi)

        x = torch.cat(outs, dim=1)
        x = self.dropout(x)
        x = _auto_pad_1d(x, self.out_conv.kernel_size[0])
        x = self.out_conv(x)
        x = self.out_relu(x)
        return x


class BasicBlock(nn.Module):
    """
    Basic block composed of multiple CombConvLayers stacked sequentially followed by max pooling.

    Args:
        in_channels (int): Number of input channels.
        layer_channels (list[int]): List of output channels for each CombConvLayer.
        comb_kernel_sizes (list[int]): Kernel sizes for CombConvLayer convolutions.
        comb_out_kernel_size (int): Kernel size of the final convolution in each CombConvLayer.
        drop_rate (float): Dropout rate applied in CombConvLayers.
        pool_size (int): Kernel size for MaxPool1d.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C_in, L), where
            N is the batch size,
            C_in is the number of input channels,
            L is the length of the input sequence.

    Forward Output:
        torch.Tensor: Output tensor after convolutional layers and max pooling,
            with shape (N, C_out, L_out), where
            C_out is the number of channels after concatenation,
            L_out is the pooled length.
    """

    def __init__(
        self,
        in_channels: int,
        layer_channels: list,
        comb_kernel_sizes,
        comb_out_kernel_size,
        drop_rate,
        pool_size,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            *[
                CombConvLayer(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_sizes=comb_kernel_sizes,
                    out_kernel_size=comb_out_kernel_size,
                    drop_rate=drop_rate,
                )
                for inc, outc in zip(
                    [in_channels] + layer_channels[:-1], layer_channels
                )
            ]
        )
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x1 = self.conv_layers(x)
        x1 = torch.cat([x, x1], dim=1)
        x1 = self.pool(x1)
        return x1


class SideLayer(nn.Module):
    """
    Side layer for auxiliary classification consisting of a CombConvLayer followed by
    fully connected layers for classification.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the convolution layer.
        comb_kernel_sizes (list[int]): Kernel sizes for CombConvLayer convolutions.
        comb_out_kernel_size (int): Kernel size for the final convolution in CombConvLayer.
        drop_rate (float): Dropout rate in CombConvLayer.
        linear_in_dim (int): Input feature dimension for the first linear layer.
        linear_hidden_dim (int): Hidden layer size for the linear classifier.
        linear_out_dim (int): Output dimension of the linear classifier (number of classes).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C, L), where
            N is the batch size,
            C is the number of input channels,
            L is the length of the input sequence.

    Forward Output:
        tuple: A tuple containing:
            - x1 (torch.Tensor): Flattened output of the convolution layer, shape (N, C*L).
            - x2 (torch.Tensor): Output after the first linear layer and ReLU activation, shape (N, linear_hidden_dim).
            - x3 (torch.Tensor): Final output after sigmoid activation, shape (N, linear_out_dim).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: list,
        comb_kernel_sizes,
        comb_out_kernel_size,
        drop_rate,
        linear_in_dim,
        linear_hidden_dim,
        linear_out_dim,
    ):
        super().__init__()

        self.conv_layer = CombConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=comb_kernel_sizes,
            out_kernel_size=comb_out_kernel_size,
            drop_rate=drop_rate,
        )

        self.flatten = nn.Flatten(1)

        self.lin0 = nn.Linear(in_features=linear_in_dim, out_features=linear_hidden_dim)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(
            in_features=linear_hidden_dim, out_features=linear_out_dim
        )
        self.sigmoid = nn.Sigmoid()

        self.conv_out_channels = out_channels
        self.linear_in_dim = linear_in_dim

    def forward(self, x):
        x = self.conv_layer(x)
        N, C, L = x.size()

        if C * L != self.linear_in_dim:
            # The input shape of the official DiTingMotion-model is fixed to (2, 128).
            # In order to accommodate different shapes of input, interpolation is used here.
            tartget_size = self.linear_in_dim // self.conv_out_channels
            x = F.interpolate(x, tartget_size)

        x1 = self.flatten(x)
        x2 = self.lin0(x1)
        x2 = self.relu(x2)
        x3 = self.lin1(x2)
        x3 = self.sigmoid(x3)
        return x1, x2, x3


class DiTingMotion(nn.Module):
    """
    DiTingMotion main model combining multiple BasicBlocks and SideLayers
    to perform first-motion polarity and clarity classification.

    Args:
        in_channels (int): Number of input channels.
        blocks_layer_channels (list[list[int]], optional): List specifying output channels of each block's layers.
        side_layer_conv_channels (int, optional): Output channels for side layer convolution.
        blocks_sidelayer_linear_in_dims (list[int], optional): Input dims for side layer linear layers per block.
        blocks_sidelayer_linear_hidden_dims (list[int], optional): Hidden dims for side layer linear layers per block.
        comb_kernel_sizes (list[int], optional): Kernel sizes for CombConvLayers.
        comb_out_kernel_size (int, optional): Kernel size for final conv in CombConvLayer.
        pool_size (int, optional): Max pool kernel size.
        drop_rate (float, optional): Dropout rate.
        fuse_hidden_dim (int, optional): Hidden dimension for fusion layers.
        num_polarity_classes (int, optional): Number of polarity output classes.
        num_clarity_classes (int, optional): Number of clarity output classes.
        **kwargs: Additional keyword arguments.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C, L), where
            N is the batch size,
            C is the number of input channels,
            L is the length of the input sequence.

    Forward Output:
        tuple: A tuple containing:
            - final_clarity (torch.Tensor): Tensor of shape (N, num_clarity_classes) representing
              the averaged clarity classification outputs.
            - final_polarity (torch.Tensor): Tensor of shape (N, num_polarity_classes) representing
              the averaged polarity classification outputs.
    """

    def __init__(
        self,
        in_channels: int,
        blocks_layer_channels: list = [
            [8, 8],
            [8, 8],
            [8, 8, 8],
            [8, 8, 8],
            [8, 8, 8],
        ],
        side_layer_conv_channels: int = 2,
        blocks_sidelayer_linear_in_dims: list = [None, None, 32, 16, 16],
        blocks_sidelayer_linear_hidden_dims: list = [None, None, 8, 8, 8],
        comb_kernel_sizes: list = [3, 3, 5, 5],
        comb_out_kernel_size: int = 3,
        pool_size: int = 2,
        drop_rate: float = 0.2,
        fuse_hidden_dim: int = 8,
        num_polarity_classes: int = 2,
        num_clarity_classes: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.clarity_side_layers = nn.ModuleList()
        self.polarity_side_layers = nn.ModuleList()

        # Compute input channels for each block
        blocks_in_channels = [in_channels]
        for blc in blocks_layer_channels[:-1]:
            blocks_in_channels.append(blc[-1] + blocks_in_channels[-1])

        fuse_polarity_in_dim = fuse_clarity_in_dim = 0

        # Build blocks and side layers
        for inc, layer_channels, side_lin_in_dim, side_lin_hidden_dim in zip(
            blocks_in_channels,
            blocks_layer_channels,
            blocks_sidelayer_linear_in_dims,
            blocks_sidelayer_linear_hidden_dims,
        ):
            # Create a BasicBlock
            block = BasicBlock(
                in_channels=inc,
                layer_channels=layer_channels,
                comb_kernel_sizes=comb_kernel_sizes,
                comb_out_kernel_size=comb_out_kernel_size,
                drop_rate=drop_rate,
                pool_size=pool_size,
            )

            if side_lin_in_dim is not None:
                # Create clarity side layer
                clarity_side_layer = SideLayer(
                    in_channels=layer_channels[-1] + inc,
                    out_channels=side_layer_conv_channels,
                    comb_kernel_sizes=comb_kernel_sizes,
                    comb_out_kernel_size=comb_out_kernel_size,
                    drop_rate=drop_rate,
                    linear_in_dim=side_lin_in_dim,
                    linear_hidden_dim=side_lin_hidden_dim,
                    linear_out_dim=num_clarity_classes,
                )

                # Create polarity side layer
                polarity_side_layer = SideLayer(
                    in_channels=layer_channels[-1] + inc,
                    out_channels=side_layer_conv_channels,
                    comb_kernel_sizes=comb_kernel_sizes,
                    comb_out_kernel_size=comb_out_kernel_size,
                    drop_rate=drop_rate,
                    linear_in_dim=side_lin_in_dim,
                    linear_hidden_dim=side_lin_hidden_dim,
                    linear_out_dim=num_polarity_classes,
                )

                fuse_clarity_in_dim += side_lin_in_dim
                fuse_polarity_in_dim += side_lin_hidden_dim

            else:
                clarity_side_layer = polarity_side_layer = None

            self.blocks.append(block)
            self.clarity_side_layers.append(clarity_side_layer)
            self.polarity_side_layers.append(polarity_side_layer)

        # Fusion layers for polarity and clarity outputs
        self.fuse_polarity = nn.Sequential(
            *[
                nn.Linear(in_features=indim, out_features=outdim)
                for indim, outdim in zip(
                    [fuse_polarity_in_dim, fuse_hidden_dim],
                    [fuse_hidden_dim, num_polarity_classes],
                )
            ],
            nn.Sigmoid(),
        )

        self.fuse_clarity = nn.Sequential(
            *[
                nn.Linear(in_features=indim, out_features=outdim)
                for indim, outdim in zip(
                    [fuse_clarity_in_dim, fuse_hidden_dim],
                    [fuse_hidden_dim, num_clarity_classes],
                )
            ],
            nn.Sigmoid(),
        )

    def forward(self, x):
        clarity_to_fuse = list()
        polarity_to_fuse = list()
        clarity_outs = list()
        polarity_outs = list()

        for block, clarity_side_layer, polarity_side_layer in zip(
            self.blocks, self.clarity_side_layers, self.polarity_side_layers
        ):
            # Pass through BasicBlock
            x = block(x)

            # Auxiliary side layers if available
            if clarity_side_layer is not None and polarity_side_layer is not None:
                c0, _, c2 = clarity_side_layer(x)
                clarity_to_fuse.append(c0)
                clarity_outs.append(c2)

                _, p1, p2 = polarity_side_layer(x)
                polarity_to_fuse.append(p1)
                polarity_outs.append(p2)

        # Fuse side layer outputs
        x = torch.cat(clarity_to_fuse, dim=-1)
        x = self.fuse_clarity(x)
        clarity_outs.append(x)

        x = torch.cat(polarity_to_fuse, dim=-1)
        x = self.fuse_polarity(x)
        polarity_outs.append(x)

        # Compute final averaged outputs
        final_clarity = sum(clarity_outs) / len(clarity_outs)
        final_polarity = sum(polarity_outs) / len(polarity_outs)

        return final_clarity, final_polarity


@register_model
def ditingmotion(**kwargs):
    """
    Factory function to create a DiTingMotion model with default parameters.

    Args:
        **kwargs: Additional keyword arguments passed to DiTingMotion constructor.

    Returns:
        DiTingMotion: Initialized DiTingMotion model.
    """
    model = DiTingMotion(num_polarity_classes=2, num_clarity_classes=2, **kwargs)
    return model
