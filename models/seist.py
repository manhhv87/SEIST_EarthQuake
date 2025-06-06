import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from functools import partial
from ._factory import register_model


def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Automatically pads a 1D tensor along a specified dimension so that the output size
    of a convolution layer matches `ceil(input_size / stride)`.

    This function emulates 'same' padding behavior for 1D convolutions, which is not
    supported by `torch.jit` or `torch.onnx`.

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.
        kernel_size (int): Kernel size of the convolution.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dim (int, optional): Dimension along which to apply padding. Defaults to -1 (last dimension).
        padding_value (float, optional): Constant value to use for padding. Defaults to 0.0.

    Raises:
        AssertionError: If `kernel_size` is less than `stride`.

    Returns:
        torch.Tensor: Padded tensor with shape adjusted so that the convolution output length
            along the specified dimension is `ceil(original_length / stride)`.
    """

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


def _make_divisible(v: int, divisor: int) -> int:
    """
    Adjusts `v` to be the closest integer greater than or equal to `v` that is divisible by `divisor`.

    This function ensures that the returned value is at least `divisor` and does not decrease
    too much compared to the original `v`. It is commonly used in model design to ensure
    channel sizes are divisible by a certain number for hardware efficiency.

    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    Args:
        v (int): Original integer value to adjust.
        divisor (int): The number by which the result should be divisible.

    Returns:
        int: Adjusted integer divisible by `divisor`.
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ScaledActivation(nn.Module):
    """
    Applies an activation function followed by scaling the output by a given factor.

    This module wraps an activation layer and multiplies its output by a scale factor.

    Args:
        act_layer (nn.Module): Activation layer class (e.g., nn.ReLU, nn.Sigmoid). This will be instantiated inside.
        scale_factor (float): Factor to scale the output of the activation.

    Forward Input:
        x (torch.Tensor): Input tensor of any shape.

    Forward Output:
        torch.Tensor: Output tensor after applying activation and scaling. Same shape as input.
    """
    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer()

    def forward(self, x):
        return self.act(x) * self.scale_factor
    

class LocalAwareAggregationBlock(nn.Module):
    """
    Local Aware Aggregation block for 1D sequence data.

    This block aggregates local contextual information using average and max pooling 
    (if `kernel_size` > 1), then applies a 1D convolution and normalization.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels after projection.
        kernel_size (int): Kernel size for pooling operations. If 1, pooling is skipped.
        norm_layer (Callable): Normalization layer constructor (e.g., nn.BatchNorm1d or nn.LayerNorm).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, in_dim, sequence_length).

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, out_dim, new_sequence_length), 
                      where the length may be reduced depending on `kernel_size`.
    """

    def __init__(self, in_dim, out_dim, kernel_size,  norm_layer):
        super().__init__()

        if kernel_size > 1:
            self.avg_pool = nn.AvgPool1d(kernel_size, ceil_mode=True)
            self.max_pool = nn.MaxPool1d(kernel_size, ceil_mode=True)

        else:
            self.avg_pool = self.max_pool  = None
        
        self.proj = nn.Conv1d(
                in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=False
            )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        if self.avg_pool is not None:
            x = self.avg_pool(x) + self.max_pool(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) block implemented with 1D convolutions.

    This MLP module replaces standard linear layers with 1D convolutions (kernel size = 1)
    to avoid transpose operations commonly needed in transformer-like architectures.

    Args:
        in_dim (int): Number of input channels/features.
        out_dim (int): Number of output channels/features.
        mlp_ratio (float): Expansion ratio for hidden dimension, typically > 1.
        bias (bool): Whether to use bias in convolution layers.
        mlp_drop_rate (float): Dropout rate applied after the second layer.
        act_layer (Callable): Activation layer class (e.g., nn.ReLU, nn.GELU).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, in_dim, sequence_length).

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, out_dim, sequence_length).
    """

    def __init__(self, in_dim, out_dim, mlp_ratio, bias, mlp_drop_rate, act_layer):
        super().__init__()

        ffwd_dim = int(in_dim * mlp_ratio)
        # Use `nn.Conv1d` instead of `nn.Linear` to reduce the use of transpose operations.
        self.lin0 = nn.Conv1d(
            in_channels=in_dim, out_channels=ffwd_dim, kernel_size=1, bias=bias
        )
        self.act = act_layer()
        self.lin1 = nn.Conv1d(
            in_channels=ffwd_dim, out_channels=out_dim, kernel_size=1, bias=bias
        )
        self.dropout = nn.Dropout(mlp_drop_rate)

    def forward(self, x):
        x = self.lin0(x)
        x = self.act(x)
        x = self.lin1(x)
        x = self.dropout(x)
        return x


class DSConvNormAct(nn.Module):
    """Depthwise separable convolution with normalization and activation.

    This module implements a depthwise separable 1D convolution block consisting of:
    - Pointwise (1x1) convolution to project input features.
    - Depthwise convolution to process each channel independently.
    - Pointwise (1x1) convolution to mix channels.
    - Followed by normalization and activation layers.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        kernel_size (int): Kernel size for the depthwise convolution.
        stride (int): Stride for the depthwise convolution.
        act_layer (Callable): Constructor for the activation layer (e.g., nn.ReLU).
        norm_layer (Callable): Constructor for the normalization layer (e.g., nn.BatchNorm1d).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, in_dim, length), where
            batch_size is the number of samples,
            in_dim is the number of input channels,
            length is the sequence length.

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, out_dim, output_length),
            where output_length depends on the convolution parameters including kernel size and stride.
    """

    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()

        self.in_proj = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False
        )

        self.dconv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_dim,
            bias=False,
        )
        self.pconv = nn.Conv1d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.dconv.kernel_size[0], self.dconv.stride[0])
        x = self.dconv(x)
        x = self.pconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class StemBlock(nn.Module):
    """Stem layer consisting of multiple parallel depthwise separable convolution blocks.

    This module applies multiple DSConvNormAct blocks with varying kernel sizes in parallel,
    then concatenates their outputs along the channel dimension, followed by a pointwise
    convolution and normalization.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels for each DSConvNormAct block.
        kernel_size (int): Base kernel size for the depthwise separable convolutions.
        stride (int): Stride for the depthwise separable convolutions.
        act_layer (Callable): Constructor for the activation layer (e.g., nn.ReLU).
        norm_layer (Callable): Constructor for the normalization layer (e.g., nn.BatchNorm1d).
        npath (int, optional): Number of parallel convolutional paths with increasing kernel sizes. Default is 3.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, in_dim, length), where
            batch_size is the number of samples,
            in_dim is the number of input channels,
            length is the sequence length.

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, out_dim, output_length), where
            output_length depends on the convolution parameters including kernel size and stride.
    """

    def __init__(
        self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer, npath=3
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                DSConvNormAct(
                    in_dim,
                    out_dim,
                    kernel_size + int(4 * delta_k),
                    stride,
                    act_layer,
                    norm_layer,
                )
                for delta_k in range(npath)
            ]
        )

        self.out_proj = nn.Conv1d(
            in_channels=npath * out_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        outs = list()
        for conv in self.convs:
            xi = conv(x)
            outs.append(xi)
        x = torch.cat(outs, dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x


class GroupConvBlock(nn.Module):
    """Group convolution block with residual connections and MLP layers.

    This block performs a grouped convolution followed by normalization and activation,
    then projects the output with a 1x1 convolution and applies stochastic depth (DropPath).
    It adds a residual connection and then applies another normalization, an MLP block,
    and a second DropPath with a final residual addition.

    Args:
        io_dim (int): Number of input and output channels.
        groups (int): Number of groups for the grouped convolution.
        kernel_size (int): Kernel size for the grouped convolution.
        path_drop_rate (float): Drop path (stochastic depth) rate.
        mlp_drop_rate (float): Dropout rate within the MLP.
        mlp_ratio (float): Expansion ratio for the MLP hidden layer.
        mlp_bias (bool): Whether to use bias in MLP convolution layers.
        act_layer (Callable): Activation layer constructor (e.g., nn.ReLU).
        norm_layer (Callable): Normalization layer constructor (e.g., nn.BatchNorm1d).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, io_dim, length), where
            batch_size is the number of samples,
            io_dim is the number of channels,
            length is the sequence length.

    Forward Output:
        torch.Tensor: Output tensor of the same shape as input
            (batch_size, io_dim, length), after applying the group convolution block
            with residual connections.
    """

    def __init__(
        self,
        io_dim,
        groups,
        kernel_size,
        path_drop_rate,
        mlp_drop_rate,
        mlp_ratio,
        mlp_bias,
        act_layer,
        norm_layer,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=io_dim,
            out_channels=io_dim,
            kernel_size=kernel_size,
            stride=1,
            groups=groups,
            bias=False,
        )
        self.norm0 = norm_layer(io_dim)
        self.act = act_layer()
        self.proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=False
        )
        self.droppath0 = DropPath(path_drop_rate)

        self.norm1 = norm_layer(io_dim)
        self.mlp = MLP(
            in_dim=io_dim,
            out_dim=io_dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            mlp_drop_rate=mlp_drop_rate,
            act_layer=act_layer,
        )
        self.droppath1 = DropPath(path_drop_rate)

    def forward(self, x):
        x1 = _auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x1 = self.conv(x1)
        x1 = self.norm0(x1)
        x1 = self.act(x1)

        x1 = self.proj(x1)
        x1 = self.droppath0(x1)
        x = x + x1

        x1 = self.norm1(x)
        x1 = self.mlp(x1)
        x1 = self.droppath1(x1)
        x = x + x1

        return x


class MultiScaleMixedConv(nn.Module):
    """Multi-scale mixed group convolution block.

    This block splits the input channels into multiple groups processed with
    different kernel sizes. Each group applies a projection, normalization,
    and a GroupConvBlock. The outputs are concatenated and normalized again.

    Args:
        io_dim (int): Number of input and output channels.
        groups (int): Number of groups to split channels into.
        kernel_sizes (List[int]): List of kernel sizes for each group convolution.
        path_drop_rate (float): Drop path (stochastic depth) rate.
        mlp_drop_rate (float): Dropout rate within the MLP blocks.
        mlp_ratio (float): Expansion ratio for the MLP hidden layers.
        mlp_bias (bool): Whether to use bias in MLP convolution layers.
        act_layer (Callable): Activation layer constructor (e.g., nn.ReLU).
        norm_layer (Callable): Normalization layer constructor (e.g., nn.BatchNorm1d).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, io_dim, length), where
            batch_size is the number of samples,
            io_dim is the number of channels,
            length is the sequence length.

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, io_dim, length),
            representing the concatenated and normalized features from all groups.
    """

    def __init__(
        self,
        io_dim,
        groups,
        kernel_sizes,
        path_drop_rate,
        mlp_drop_rate,
        mlp_ratio,
        mlp_bias,
        act_layer,
        norm_layer,
    ):
        super().__init__()

        group_size = io_dim // groups
        dims_ = []
        self.projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            dim = _make_divisible(
                (io_dim - sum(dims_)) // (len(kernel_sizes) - len(dims_)), group_size
            )

            assert dim > 0
            dims_.append(dim)

            proj = nn.Conv1d(
                in_channels=io_dim, out_channels=dim, kernel_size=1, bias=False
            )
            norm = norm_layer(dim)
            conv = GroupConvBlock(
                io_dim=dim,
                groups=dim // group_size,
                kernel_size=kernel_size,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            self.projs.append(proj)
            self.norms.append(norm)
            self.convs.append(conv)

        self.out_norm = norm_layer(io_dim)

    def forward(self, x):
        outs = list()
        for proj, norm, conv in zip(self.projs, self.norms, self.convs):
            xi = norm(proj(x))
            xi = xi + conv(xi)
            outs.append(xi)

        x = torch.cat(outs, dim=1)
        x = self.out_norm(x)

        return x


class AttentionBlock(nn.Module):
    """Multi-head attention block with optional local aggregation.

    This module implements multi-head self-attention with an optional local
    aggregation step before computing key and value projections. It supports
    dropout for keys, attention weights, and output projections.

    Args:
        io_dim (int): Number of input and output channels.
        head_dim (int): Dimension of each attention head.
        qkv_bias (bool): Whether to include bias terms in query, key, value projections.
        attn_drop_rate (float): Dropout rate applied to the attention weights.
        key_drop_rate (float): Dropout rate applied to the key tensor.
        proj_drop_rate (float): Dropout rate applied after the output projection.
        attn_aggr_ratio (int): Kernel size for local aggregation. If > 1, applies
            a LocalAwareAggregationBlock; otherwise, no aggregation is applied.
        norm_layer (Callable): Normalization layer constructor (e.g., nn.BatchNorm1d).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, io_dim, length),
            where batch_size is the number of samples,
            io_dim is the number of channels,
            length is the sequence length.

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, io_dim, length),
            representing the attention-enhanced features.
    """

    def __init__(
        self,
        io_dim,
        head_dim,
        qkv_bias,
        attn_drop_rate,
        key_drop_rate,
        proj_drop_rate,
        attn_aggr_ratio,
        norm_layer,
    ):
        super().__init__()

        self.num_heads = io_dim // head_dim

        self.aggr = (
            LocalAwareAggregationBlock(
                in_dim=io_dim,
                out_dim=io_dim,
                kernel_size=attn_aggr_ratio,
                norm_layer=norm_layer,
            )
            if attn_aggr_ratio > 1
            else nn.Identity()
        )
        self.norm = norm_layer(io_dim) if attn_aggr_ratio > 1 else nn.Identity()

        self.q_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.k_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.v_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.k_dropout = nn.Dropout(key_drop_rate)
        self.attn_dropout = nn.Dropout(attn_drop_rate)

        self.out_proj = nn.Conv1d(
            in_channels=io_dim, out_channels=io_dim, kernel_size=1, bias=qkv_bias
        )
        self.proj_dropout = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        N, C, L = x.size()

        q = self.q_proj(x).view(N, self.num_heads, C // self.num_heads, L)

        x = self.aggr(x)
        x = self.norm(x)

        k = self.k_proj(x).view(N, self.num_heads, C // self.num_heads, -1)
        v = self.v_proj(x).view(N, self.num_heads, C // self.num_heads, -1)

        k = self.k_dropout(k)

        N, Nh, E, L = q.size()

        q_scaled = q / math.sqrt(E)

        attn = (q_scaled.transpose(-1, -2) @ k).softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(N, C, L)

        x = self.out_proj(x)
        x = self.proj_dropout(x)

        return x


class MultiPathTransformerLayer(nn.Module):
    """Transformer block with multi-path attention and convolution branches.

    This block splits the input channels into two paths: one for multi-head
    attention and another for group convolution. Both paths process the input
    in parallel, then their outputs are concatenated, normalized, and passed
    through an MLP block with residual connections.

    Args:
        io_dim (int): Number of input and output channels.
        path_drop_rate (float): Drop path rate for stochastic depth regularization.
        attn_aggr_ratio (int): Kernel size for local aggregation in attention.
        attn_ratio (float): Fraction of channels allocated to the attention path (0 to 1).
        head_dim (int): Dimension of each attention head.
        qkv_bias (bool): Whether to use bias in QKV projections.
        mlp_ratio (float): Expansion ratio for MLP hidden dimension.
        mlp_bias (bool): Whether to use bias in MLP layers.
        attn_drop_rate (float): Dropout rate applied to attention weights.
        key_drop_rate (float): Dropout rate applied to attention keys.
        attn_out_drop_rate (float): Dropout rate after attention output projection.
        mlp_drop_rate (float): Dropout rate in the MLP block.
        act_layer (Callable): Activation layer constructor (e.g., nn.GELU).
        norm_layer (Callable): Normalization layer constructor (e.g., nn.BatchNorm1d).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, io_dim, length),
            where batch_size is the batch size,
            io_dim is the number of channels,
            length is the sequence length.

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, io_dim, length),
            after multi-path processing and MLP with residual connections.
    """

    def __init__(
        self,
        io_dim,
        path_drop_rate,
        attn_aggr_ratio,
        attn_ratio,
        head_dim,
        qkv_bias,
        mlp_ratio,
        mlp_bias,
        attn_drop_rate,
        key_drop_rate,
        attn_out_drop_rate,
        mlp_drop_rate,
        act_layer,
        norm_layer,
    ):
        super().__init__()

        assert 0 <= attn_ratio <= 1

        self.attn_out_dim = (
            _make_divisible(int(io_dim * attn_ratio), head_dim) if attn_ratio > 0 else 0
        )
        self.conv_out_dim = max(io_dim - self.attn_out_dim, 0)

        self.has_attn = self.attn_out_dim > 0
        self.has_conv = self.conv_out_dim > 0

        if self.has_attn:
            self.attn_proj = nn.Conv1d(
                in_channels=io_dim,
                out_channels=self.attn_out_dim,
                kernel_size=1,
                bias=False,
            )
            self.norm0 = norm_layer(self.attn_out_dim)
            self.attention = AttentionBlock(
                io_dim=self.attn_out_dim,
                head_dim=head_dim,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                key_drop_rate=key_drop_rate,
                proj_drop_rate=attn_out_drop_rate,
                attn_aggr_ratio=attn_aggr_ratio,
                norm_layer=norm_layer,
            )
            self.attn_droppath = DropPath(path_drop_rate * attn_ratio)
        else:
            self.attn_proj = self.norm0 = self.attention = self.attn_droppath = None

        if self.has_conv:
            self.conv_proj = nn.Conv1d(
                in_channels=io_dim,
                out_channels=self.conv_out_dim,
                kernel_size=1,
                bias=False,
            )
            self.norm1 = norm_layer(self.conv_out_dim)

            self.gconv = GroupConvBlock(
                io_dim=self.conv_out_dim,
                groups=self.conv_out_dim // head_dim,
                kernel_size=3,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            self.gconv_droppath = DropPath(path_drop_rate * (1 - attn_ratio))
        else:
            self.conv_proj = self.norm1 = self.gconv = self.gconv_droppath = None

        self.norm2 = norm_layer(io_dim)

        self.mlp = MLP(
            in_dim=io_dim,
            out_dim=io_dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            mlp_drop_rate=mlp_drop_rate,
            act_layer=act_layer,
        )
        self.mlp_droppath = DropPath(path_drop_rate)

    def forward(self, x):
        # N,C,L = x.size()

        outs = list()
        if self.has_attn:
            x1 = self.norm0(self.attn_proj(x))
            x1 = x1 + self.attn_droppath(self.attention(x1))
            outs.append(x1)

        if self.has_conv:
            x2 = self.norm1(self.conv_proj(x))
            x2 = x2 + self.gconv_droppath(self.gconv(x2))
            outs.append(x2)

        x = torch.cat(outs, dim=1)
        x = self.norm2(x)
        x = x + self.mlp_droppath(self.mlp(x))

        return x


class HeadDetectionPicking(nn.Module):
    """Head module for detection and phase picking tasks.

    This module performs progressive upsampling through a series of convolutional
    blocks, followed by a final convolution and optional activation. It is designed
    to refine feature maps to the target resolution for detection and picking.

    Args:
        feature_channels (int): Number of input feature channels.
        layer_channels (List[int]): Number of output channels for each intermediate layer.
        layer_kernel_sizes (List[int]): Kernel sizes for each intermediate convolutional layer.
        act_layer (Callable): Activation layer constructor for intermediate layers (e.g., nn.ReLU).
        norm_layer (Callable): Normalization layer constructor for intermediate layers (e.g., nn.BatchNorm1d).
        out_act_layer (Callable, optional): Activation layer constructor for output layer.
            Defaults to nn.Identity.
        out_channels (int, optional): Number of output channels of the final output.
            Defaults to 1.
        **kwargs: Additional keyword arguments (currently unused).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C, L), where
            N is batch size,
            C is number of feature channels,
            L is sequence length.
        x0 (torch.Tensor): Reference tensor used to determine upsampling target length.

    Forward Output:
        torch.Tensor: Output tensor of shape (N, out_channels, L_out),
            where L_out is the length of x0 along the last dimension.
    """

    def __init__(
        self,
        feature_channels,
        layer_channels,
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels=1,
        **kwargs,
    ):
        super().__init__()

        assert len(layer_channels) == len(layer_kernel_sizes)

        self.depth = len(layer_channels)

        self.up_layers = nn.ModuleList()

        for i, (inc, outc, kers) in enumerate(
            zip(
                [feature_channels] + layer_channels[:-1],
                layer_channels[:-1] + [out_channels * 2],
                layer_kernel_sizes,
            )
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()

            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )

        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        N, C, L = x.size()
        up_sizes = self._upsampling_sizes(in_size=L, out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i]
            x = F.interpolate(x, size=upsize, mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class HeadClassification(nn.Module):
    """Classification head module.

    This module performs global average pooling followed by a linear layer and
    an optional activation function to produce class scores from feature maps.

    Args:
        feature_channels (int): Number of input feature channels.
        num_classes (int): Number of output classes.
        out_act_layer (Callable): Activation layer constructor for output (e.g., nn.Softmax, nn.Identity).
        **kwargs: Additional keyword arguments (currently unused).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C, L), where
            N is batch size,
            C is number of feature channels,
            L is sequence length.
        _ (torch.Tensor, optional): Unused tensor, included for interface compatibility.

    Forward Output:
        torch.Tensor: Output tensor of shape (N, num_classes), representing class scores or probabilities.
    """

    def __init__(self, feature_channels, num_classes, out_act_layer, **kwargs):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels , num_classes)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class HeadRegression(nn.Module):
    """Regression head module.

    This module applies global average pooling followed by a linear layer and
    an optional activation function to produce regression outputs from feature maps.

    Args:
        feature_channels (int): Number of input feature channels.
        out_act_layer (Callable): Activation layer constructor for output (e.g., nn.Identity).
        **kwargs: Additional keyword arguments (currently unused).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C, L), where
            N is batch size,
            C is number of feature channels,
            L is sequence length.
        _ (torch.Tensor, optional): Unused tensor, included for interface compatibility.

    Forward Output:
        torch.Tensor: Output tensor of shape (N, 1), representing regression predictions.
    """

    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels , 1)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class SeismogramTransformer(nn.Module):
    """
    Seismogram Transformer model.

    This model processes multi-channel seismogram signals through multiple
    convolutional and transformer-based layers, optionally with checkpointing
    to save memory, and outputs predictions through a configurable output head.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        stem_channels (list of int): Channels for each stem convolutional layer. Defaults to [16, 8, 16, 16].
        stem_kernel_sizes (list of int): Kernel sizes for each stem convolutional layer. Defaults to [11, 5, 5, 7].
        stem_strides (list of int): Strides for each stem convolutional layer. Defaults to [2, 1, 1, 2].
        layer_blocks (list of int): Number of blocks in each encoder stage. Defaults to [2, 3, 6, 2].
        layer_channels (list of int): Number of channels in each encoder stage. Defaults to [24, 32, 64, 96].
        attn_blocks (list of int): Number of attention blocks in each encoder stage. Defaults to [1, 1, 2, 1].
        stage_aggr_ratios (list of int): Aggregation ratios for each stage. Defaults to [2, 2, 2, 2].
        attn_aggr_ratios (list of int): Aggregation ratios inside attention blocks. Defaults to [8, 4, 2, 1].
        head_dims (list of int): Head dimensions for multi-head attention in each stage. Defaults to [8, 8, 16, 32].
        msmc_kernel_sizes (list of int): Kernel sizes for multi-scale mixed convolutions. Defaults to [3, 5].
        path_drop_rate (float): Drop path rate for stochastic depth regularization. Defaults to 0.2.
        attn_drop_rate (float): Dropout rate within attention blocks. Defaults to 0.1.
        key_drop_rate (float): Dropout rate applied to attention keys. Defaults to 0.1.
        mlp_drop_rate (float): Dropout rate for MLP layers. Defaults to 0.2.
        other_drop_rate (float): Dropout rate for other modules. Defaults to 0.1.
        attn_ratio (float): Ratio of attention output channels to total channels. Defaults to 0.6.
        mlp_ratio (int): Ratio of hidden dimension to input dimension in MLPs. Defaults to 2.
        qkv_bias (bool): Whether to add bias in QKV projections. Defaults to True.
        mlp_bias (bool): Whether to add bias in MLP layers. Defaults to True.
        act_layer (nn.Module): Activation layer constructor. Defaults to nn.GELU.
        norm_layer (nn.Module): Normalization layer constructor. Defaults to nn.BatchNorm1d.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Defaults to False.
        output_head (nn.Module): Output head module class (e.g., HeadDetectionPicking). Defaults to HeadDetectionPicking.
        **kwargs: Additional arguments for future extensions.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (N, C, L) where
            N is batch size,
            C is number of input channels,
            L is sequence length.

    Forward Output:
        torch.Tensor: Output tensor depends on the output_head module used,
            typically representing detection, classification, or regression results.
    """

    def __init__(
        self,
        in_channels=3,
        stem_channels=[16, 8, 16, 16],
        stem_kernel_sizes=[11, 5, 5, 7],
        stem_strides=[2, 1, 1, 2],
        layer_blocks=[2, 3, 6, 2],
        layer_channels=[24, 32, 64, 96],
        attn_blocks=[1, 1, 2, 1],
        stage_aggr_ratios=[2, 2, 2, 2],
        attn_aggr_ratios=[8, 4, 2, 1],
        head_dims=[8, 8, 16, 32],
        msmc_kernel_sizes=[3, 5],
        path_drop_rate=0.2,
        attn_drop_rate=0.1,
        key_drop_rate=0.1,
        mlp_drop_rate=0.2,
        other_drop_rate=0.1,
        attn_ratio=0.6,
        mlp_ratio=2,
        qkv_bias=True,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,
        output_head=HeadDetectionPicking,
        **kwargs,
    ):
        super().__init__()

        assert len(stem_channels) == len(stem_kernel_sizes) == len(stem_strides)
        assert (
            len(layer_blocks)
            == len(layer_channels)
            == len(stage_aggr_ratios)
            == len(attn_aggr_ratios)
            == len(attn_blocks)
            == len(head_dims)
        )

        self.use_checkpoint = use_checkpoint
        self.stem = nn.Sequential(
            *[
                StemBlock(
                    in_dim=inc,
                    out_dim=outc,
                    kernel_size=kers,
                    stride=strd,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for inc, outc, kers, strd in zip(
                    [in_channels] + stem_channels[:-1],
                    stem_channels,
                    stem_kernel_sizes,
                    stem_strides,
                )
            ]
        )

        pdprs = [x.item() for x in torch.linspace(0, path_drop_rate, sum(layer_blocks))]

        self.encoder_layers = nn.ModuleList()

        for i, (
            num_blocks,
            inc,
            lc,
            num_attns,
            aggr_ratio,
            attn_aggr_ratio,
            head_dim,
        ) in enumerate(
            zip(
                layer_blocks,
                stem_channels[-1:] + layer_channels,
                layer_channels,
                attn_blocks,
                stage_aggr_ratios,
                attn_aggr_ratios,
                head_dims,
            )
        ):
            layer_modules = []

            stage_aggr = LocalAwareAggregationBlock(
                in_dim=inc,
                out_dim=lc,
                kernel_size=aggr_ratio,
                norm_layer=norm_layer,
            )
            layer_modules.append(stage_aggr)

            for j in range(num_blocks):
                pdpr = pdprs[sum(layer_blocks[:i] + [j])]
                if j >= num_blocks - num_attns:
                    block = MultiPathTransformerLayer(
                        io_dim=lc,
                        path_drop_rate=pdpr,
                        attn_aggr_ratio=attn_aggr_ratio,
                        attn_ratio=attn_ratio,
                        head_dim=head_dim,
                        qkv_bias=qkv_bias,
                        mlp_ratio=mlp_ratio,
                        mlp_bias=mlp_bias,
                        attn_drop_rate=attn_drop_rate,
                        key_drop_rate=key_drop_rate,
                        attn_out_drop_rate=other_drop_rate,
                        mlp_drop_rate=mlp_drop_rate,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                else:
                    block = MultiScaleMixedConv(
                        io_dim=lc,
                        groups=lc // head_dim,  # * 2**i,
                        kernel_sizes=msmc_kernel_sizes,
                        path_drop_rate=pdpr,
                        mlp_drop_rate=mlp_drop_rate,
                        mlp_ratio=mlp_ratio,
                        mlp_bias=mlp_bias,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                layer_modules.append(block)

            self.encoder_layers.append(nn.Sequential(*layer_modules))

        if (output_head in [HeadDetectionPicking]) or (
            isinstance(output_head, partial)
            and (output_head.func in [HeadDetectionPicking])
        ):
            out_layer_channels = []
            out_layer_kernel_sizes = []
            for channel, kernel, stride in zip(
                [in_channels] + stem_channels + layer_channels[:-1],
                stem_kernel_sizes
                + [max(msmc_kernel_sizes)] * len(layer_channels),
                stem_strides + stage_aggr_ratios,
            ):
                if stride > 1:
                    out_layer_channels.insert(0, channel)
                    out_layer_kernel_sizes.insert(0, kernel)

            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=layer_channels[-1],
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias,
                # out_channels=,
                # out_act_layer=,
                # channel_factor=,
                # msmc_kernel_sizes=,
                # num_classes=,
            )

        else:
            self.out_head = output_head(
                feature_channels=layer_channels[-1],
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(
            m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d)
        ):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        N, C, L = x.size()
        x_input = x

        # Stem
        x = self.stem(x)

        # Basic layers
        for layer in self.encoder_layers:
            if self.use_checkpoint and not (
                torch.jit.is_tracing() or torch.jit.is_scripting()
            ):
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

        # Output head
        x = self.out_head(x, x_input)

        return x


def SeismogramTransformer_S(**kwargs):
    """
    Constructs a small variant of the SeismogramTransformer model.

    This variant uses a reduced number of blocks and channels to create
    a lightweight transformer suitable for smaller datasets or faster inference.

    Args:
        **kwargs: Additional keyword arguments to override the default configuration
            of the SeismogramTransformer.

    Returns:
        SeismogramTransformer: An instance of the SeismogramTransformer model
        configured with the small variant parameters.
    """

    _args = dict(
        stem_channels=[16, 8, 16, 16],
        stem_kernel_sizes=[11, 5, 5, 7],
        stem_strides=[2, 1, 1, 2],
        layer_blocks=[2, 2, 3, 2],
        layer_channels=[16, 24, 32, 64],
        attn_blocks=[1, 1, 1, 1],
        stage_aggr_ratios=[2, 2, 2, 2],
        attn_aggr_ratios=[8, 4, 2, 1],
        head_dims=[8, 8, 8, 16],
        msmc_kernel_sizes=[5, 7],
        path_drop_rate=0.1,
        attn_drop_rate=0.1,
        key_drop_rate=0.1,
        mlp_drop_rate=0.1,
        other_drop_rate=0.1,
        attn_ratio=0.6,
        mlp_ratio=2,
    )
    _args.update(**kwargs)
    model = SeismogramTransformer(**_args)
    return model


def SeismogramTransformer_M(**kwargs):
    """
    Constructs a medium-sized variant of the SeismogramTransformer model.

    This configuration balances complexity and performance,
    suitable for general-purpose seismogram analysis tasks.

    Args:
        **kwargs: Additional keyword arguments to override the default configuration
            of the SeismogramTransformer.

    Returns:
        SeismogramTransformer: An instance of the SeismogramTransformer model
        configured with the medium variant parameters.
    """

    _args = dict(
        stem_channels=[16, 8, 16, 16],
        stem_kernel_sizes=[11, 5, 5, 7],
        stem_strides=[2, 1, 1, 2],
        layer_blocks=[2, 3, 6, 2],
        layer_channels=[24, 32, 64, 96],
        attn_blocks=[1, 1, 1, 1],  
        stage_aggr_ratios=[2, 2, 2, 2],
        attn_aggr_ratios=[8, 4, 2, 1],
        head_dims=[8, 8, 16, 32],
        msmc_kernel_sizes=[5, 7],
        path_drop_rate=0.1,
        attn_drop_rate=0.1,
        key_drop_rate=0.1,
        mlp_drop_rate=0.1,
        other_drop_rate=0.1,
        attn_ratio=0.6,
        mlp_ratio=2,
    )
    _args.update(**kwargs)
    model = SeismogramTransformer(**_args)
    return model


def SeismogramTransformer_L(**kwargs):
    """
    Constructs a large variant of the SeismogramTransformer model.

    This configuration is designed for higher capacity and accuracy,
    suitable for more complex seismogram analysis tasks requiring deeper networks.

    Args:
        **kwargs: Additional keyword arguments to override the default configuration
            of the SeismogramTransformer.

    Returns:
        SeismogramTransformer: An instance of the SeismogramTransformer model
        configured with the large variant parameters.
    """

    _args = dict(
        stem_channels=[16, 8, 16, 16],
        stem_kernel_sizes=[11, 5, 5, 7],
        stem_strides=[2, 1, 1, 2],
        layer_blocks=[2, 3, 6, 3],
        layer_channels=[32, 32, 64, 128],
        attn_blocks=[1, 1, 2, 1],
        stage_aggr_ratios=[2, 2, 2, 2],
        attn_aggr_ratios=[8, 4, 2, 1],
        head_dims=[8, 8, 16, 32],
        msmc_kernel_sizes=[3, 5, 7, 11],
        path_drop_rate=0.2,
        attn_drop_rate=0.2,
        key_drop_rate=0.1,
        mlp_drop_rate=0.2,
        other_drop_rate=0.1,
        attn_ratio=0.6,
        mlp_ratio=3,
    )
    _args.update(**kwargs)
    model = SeismogramTransformer(**_args)
    return model


@register_model
def seist_s_dpk(**kwargs):
    """
    Constructs a small Seismogram Transformer model tailored for 
    detection and phase-picking tasks.

    This model uses the small configuration of SeismogramTransformer_S 
    and the HeadDetectionPicking output head configured with a Sigmoid 
    activation and 3 output channels.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_S constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for detection and phase-picking.
    """
    model = SeismogramTransformer_S(
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),
        **kwargs,
    )
    return model


@register_model
def seist_m_dpk(**kwargs):
    """
    Constructs a medium-sized Seismogram Transformer model for 
    detection and phase-picking tasks.

    This model uses the medium configuration of SeismogramTransformer_M 
    with increased dropout rates (0.2) for path, attention, key, MLP, 
    and other modules. The output head is set to HeadDetectionPicking 
    with Sigmoid activation and 3 output channels.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_M constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for detection and phase-picking.
    """
    model = SeismogramTransformer_M(
        path_drop_rate=0.2,
        attn_drop_rate=0.2,
        key_drop_rate=0.2,
        mlp_drop_rate=0.2,
        other_drop_rate=0.2,
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),
        **kwargs,
    )
    return model


@register_model
def seist_l_dpk(**kwargs):
    """
    Constructs a large Seismogram Transformer model for detection 
    and phase-picking tasks.

    This model uses the large configuration of SeismogramTransformer_L 
    with elevated dropout rates (0.3) for path, attention, key, MLP, 
    and other modules. The output head is set to HeadDetectionPicking 
    with Sigmoid activation and 3 output channels.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_L constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for detection and phase-picking.
    """
    model = SeismogramTransformer_L(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),
        **kwargs,
    )
    return model


@register_model
def seist_s_pmp(**kwargs):
    """
    Constructs a small Seismogram Transformer model for P-wave motion 
    polarity classification.

    This model uses the small configuration of SeismogramTransformer_S 
    with dropout rates of 0.2 for path, attention, key, MLP, and other modules. 
    The output head is set to HeadClassification with Softmax activation 
    over 2 classes.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_S constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for P-motion polarity classification.
    """
    model = SeismogramTransformer_S(
        path_drop_rate=0.2,
        attn_drop_rate=0.2,
        key_drop_rate=0.2,
        mlp_drop_rate=0.2,
        other_drop_rate=0.2,
        output_head=partial(
            HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=2
        ),
        **kwargs,
    )
    return model


@register_model
def seist_m_pmp(**kwargs):
    """
    Constructs a medium-sized Seismogram Transformer model for P-wave motion 
    polarity classification.

    This model uses the medium configuration of SeismogramTransformer_M 
    with dropout rates set to 0.25 for path, attention, key, MLP, and other modules. 
    The output head is configured as HeadClassification with Softmax activation 
    over 2 classes.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_M constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for P-motion polarity classification.
    """
    model = SeismogramTransformer_M(
        path_drop_rate=0.25,
        attn_drop_rate=0.25,
        key_drop_rate=0.25,
        mlp_drop_rate=0.25,
        other_drop_rate=0.25,
        output_head=partial(
            HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=2
        ),
        **kwargs,
    )
    return model


@register_model
def seist_l_pmp(**kwargs):
    """
    Constructs a large Seismogram Transformer model for P-wave motion 
    polarity classification.

    This model uses the large configuration of SeismogramTransformer_L 
    with dropout rates set to 0.3 for path, attention, key, MLP, and other modules. 
    The output head is configured as HeadClassification with Softmax activation 
    over 2 classes.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_L constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for P-motion polarity classification.
    """
    model = SeismogramTransformer_L(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=2
        ),
        **kwargs,
    )
    return model


@register_model
def seist_s_emg(**kwargs):
    """
    Constructs a small Seismogram Transformer model for magnitude estimation.

    This model uses the small configuration of SeismogramTransformer_S.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 8.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_S constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for magnitude regression.
    """
    model = SeismogramTransformer_S(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_m_emg(**kwargs):
    """
    Constructs a medium Seismogram Transformer model for magnitude estimation.

    This model uses the medium configuration of SeismogramTransformer_M.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 8.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_M constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for magnitude regression.
    """
    model = SeismogramTransformer_M(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_l_emg(**kwargs):
    """
    Constructs a large Seismogram Transformer model for magnitude estimation.

    This model uses the large configuration of SeismogramTransformer_L.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 8.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_L constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for magnitude regression.
    """
    model = SeismogramTransformer_L(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_s_baz(**kwargs):
    """
    Constructs a small Seismogram Transformer model for azimuth estimation.

    This model uses the small configuration of SeismogramTransformer_S.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 360 
    to map predictions to azimuth angles in degrees.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_S constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for azimuth regression.
    """
    model = SeismogramTransformer_S(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=360
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_m_baz(**kwargs):
    """
    Constructs a medium Seismogram Transformer model for azimuth estimation.

    This model uses the medium configuration of SeismogramTransformer_M.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 360 
    to map predictions to azimuth angles in degrees.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_M constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for azimuth regression.
    """
    model = SeismogramTransformer_M(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=360
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_l_baz(**kwargs):
    """
    Constructs a large Seismogram Transformer model for azimuth estimation.

    This model uses the large configuration of SeismogramTransformer_L.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 360 
    to map predictions to azimuth angles in degrees.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_L constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for azimuth regression.
    """
    model = SeismogramTransformer_L(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=360
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_s_dis(**kwargs):
    """
    Constructs a small Seismogram Transformer model for epicentral distance estimation.

    This model uses the small configuration of SeismogramTransformer_S.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 500 
    to map predictions to epicentral distance values.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_S constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for epicentral distance regression.
    """
    model = SeismogramTransformer_S(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_m_dis(**kwargs):
    """
    Constructs a medium Seismogram Transformer model for epicentral distance estimation.

    This model uses the medium configuration of SeismogramTransformer_M.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 500 
    to map predictions to epicentral distance values.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_M constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for epicentral distance regression.
    """
    model = SeismogramTransformer_M(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def seist_l_dis(**kwargs):
    """
    Constructs a large Seismogram Transformer model for epicentral distance estimation.

    This model uses the large configuration of SeismogramTransformer_L.
    The output head is configured as HeadRegression with a scaled Sigmoid 
    activation function, where the sigmoid output is scaled by a factor of 500 
    to map predictions to epicentral distance values.

    Args:
        **kwargs: Additional keyword arguments passed to the 
                  SeismogramTransformer_L constructor.

    Returns:
        nn.Module: An instance of the SeismogramTransformer model 
                   configured for epicentral distance regression.
    """
    model = SeismogramTransformer_L(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500
            ),
        ),
        **kwargs,
    )
    return model