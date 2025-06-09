"""
EQTransformer: An attentive deep learning model for simultaneous earthquake detection and phase picking.

This module implements the EQTransformer architecture as described in Mousavi et al. (2020).
The model combines convolutional blocks, residual convolutional blocks, bidirectional LSTMs, and transformer layers
with local attention mechanisms to detect earthquakes and pick seismic phases from multichannel time-series data.

Main components:
- ConvBlock: 1D convolution + ReLU + max pooling with optional L1 regularization on weights and biases.
- ResConvBlock: Residual convolutional block with batch normalization, ReLU, dropout, and skip connections.
- BiLSTMBlock: Bidirectional LSTM followed by convolution and batch normalization.
- AttentionLayer: Single-head attention with optional local attention window.
- FeedForward: Position-wise feedforward neural network.
- TransformerLayer: Combines attention and feedforward layers with residual connections and layer normalization.
- Encoder: Stacked convolutional, residual convolutional, BiLSTM, and transformer layers.
- Decoder: Optional LSTM and transformer layers, followed by upsampling blocks and output convolution.
- EQTransformer: The full model assembling encoder and decoder components.

Reference:
    Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., and Beroza, G.C.
    Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking.
    Nature Communications, 11, 3952 (2020). https://doi.org/10.1038/s41467-020-17591-w
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model


class ConvBlock(nn.Module):
    """
    1D convolutional block with ReLU activation and max pooling, including optional L1 regularization.

    This block applies 1D convolution with 'same' padding, followed by ReLU activation,
    and max pooling with kernel size 2. It supports optional L1 regularization on convolution
    weights and biases via backward hooks.

    Attributes:
        conv_padding_same (Tuple[int, int]): Padding to preserve input length after convolution.
        conv (nn.Conv1d): 1D convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        pool (nn.MaxPool1d): Max pooling layer with kernel size 2.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        kernel_l1_alpha (float): L1 regularization coefficient for convolution weights. Must be non-negative.
        bias_l1_alpha (float): L1 regularization coefficient for convolution biases. Must be non-negative.

    Raises:
        AssertionError: If `kernel_l1_alpha` or `bias_l1_alpha` is negative.

    Forward input shape:
        Tensor of shape (N, C_in, L), where
        N is batch size,
        C_in is number of input channels,
        L is sequence length.

    Forward output shape:
        Tensor of shape (N, C_out, L_out), where
        C_out is number of output channels,
        L_out is sequence length after convolution and pooling.
    """

    _epsilon = 1e-6

    def __init__(
        self, in_channels, out_channels, kernel_size, kernel_l1_alpha, bias_l1_alpha
    ):
        super().__init__()

        assert kernel_l1_alpha >= 0.0
        assert bias_l1_alpha >= 0.0

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, padding=0)

        if kernel_l1_alpha > 0.0:
            self.conv.weight.register_hook(
                lambda grad: grad.data
                + kernel_l1_alpha * torch.sign(self.conv.weight.data)
            )
        if bias_l1_alpha > 0.0:
            self.conv.bias.register_hook(
                lambda grad: grad.data + bias_l1_alpha * torch.sign(self.conv.bias.data)
            )

    def forward(self, x):
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv(x)
        x = self.relu(x)
        x = F.pad(x, (0, x.size(-1) % 2), "constant", -1 / self._epsilon)
        x = self.pool(x)
        return x


class ResConvBlock(nn.Module):
    """
    Residual 1D convolutional block with batch normalization, ReLU activation, dropout, and skip connections.

    Args:
        io_channels (int): Number of input and output channels.
        kernel_size (int): Size of convolution kernel.
        drop_rate (float): Dropout rate.

    Forward Input Shape:
        Tensor of shape (N, C, L)

    Forward Output Shape:
        Tensor of shape (N, C, L), same size as input (residual connection).
    """

    def __init__(self, io_channels, kernel_size, drop_rate):
        super().__init__()

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.bn0 = nn.BatchNorm1d(num_features=io_channels)
        self.relu0 = nn.ReLU()
        self.dropout0 = nn.Dropout1d(p=drop_rate)
        self.conv0 = nn.Conv1d(
            in_channels=io_channels, out_channels=io_channels, kernel_size=kernel_size
        )

        self.bn1 = nn.BatchNorm1d(io_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(p=drop_rate)
        self.conv1 = nn.Conv1d(
            in_channels=io_channels, out_channels=io_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        x1 = self.bn0(x)
        x1 = self.relu0(x1)
        x1 = self.dropout0(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv0(x1)

        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv1(x1)
        out = x + x1
        return out


class BiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM block followed by 1D convolution and batch normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        drop_rate (float): Dropout rate.

    Forward Input Shape:
        Tensor of shape (N, C, L)

    Forward Output Shape:
        Tensor of shape (N, out_channels, L)
    """

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=drop_rate)
        self.conv = nn.Conv1d(
            in_channels=2 * out_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        return x


class AttentionLayer(nn.Module):
    """
    Single-head self-attention layer with optional local attention window.

    Args:
        in_channels (int): Number of input channels.
        d_model (int): Dimensionality of the attention mechanism.
        attn_width (int, optional): Width of local attention window. If None, full attention is used.

    Forward Input Shape:
        Tensor of shape (N, C, L)

    Forward Output:
        Tuple containing:
            - Tensor (N, C, L): Attention output.
            - Tensor (N, L, L): Attention weights.
    """

    _epsilon = 1e-6

    def __init__(self, in_channels, d_model, attn_width=None):
        super().__init__()
        self.attn_width = attn_width
        self.Wx = nn.Parameter(torch.empty((in_channels, d_model)))
        self.Wt = nn.Parameter(torch.empty((in_channels, d_model)))
        self.bh = nn.Parameter(torch.empty(d_model))
        self.Wa = nn.Parameter(torch.empty((d_model, 1)))
        self.ba = nn.Parameter(torch.empty(1))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.Wt)
        nn.init.xavier_uniform_(self.Wa)
        nn.init.zeros_(self.bh)
        nn.init.zeros_(self.ba)

    def forward(self, x):
        # (N,C,L) -> (N,L,C)
        x = x.permute(0, 2, 1)

        # (N,L,C),(C,d) -> (N,L,1,d)
        q = torch.matmul(x, self.Wt).unsqueeze(2)

        # (N,L,C),(C,d) -> (N,1,L,d)
        k = torch.matmul(x, self.Wx).unsqueeze(1)

        # (N,L,1,d),(N,1,L,d),(d,) -> (N,L,L,d)
        h = torch.tanh(q + k + self.bh)

        # (N,L,d),(d,1) -> (N,L,L,1) -> (N,L,L)
        e = (torch.matmul(h, self.Wa) + self.ba).squeeze(-1)

        # (N,L,L)
        e = torch.exp(e - torch.max(e, dim=-1, keepdim=True).values)

        # Masked attention
        if self.attn_width is not None:
            mask = (
                torch.ones(e.shape[-2:], dtype=torch.bool, device=e.device)
                .tril(self.attn_width // 2 - 1)
                .triu(-self.attn_width // 2)
            )
            e = e.where(mask, 0)

        # (N,L,L)
        s = torch.sum(e, dim=-1, keepdim=True)
        a = e / (s + self._epsilon)

        # (N,L,L),(N,L,C) -> (N,L,C)
        v = torch.matmul(a, x)

        # (N,L,C) -> (N,C,L)
        v = v.permute(0, 2, 1)

        return v, a


class FeedForward(nn.Module):
    """
    Position-wise feedforward neural network with ReLU activation and dropout.

    Args:
        io_channels (int): Number of input and output channels.
        feedforward_dim (int): Hidden layer dimensionality.
        drop_rate (float): Dropout rate.

    Forward Input Shape:
        Tensor of shape (N, L, C)

    Forward Output Shape:
        Tensor of shape (N, L, C)
    """

    def __init__(self, io_channels, feedforward_dim, drop_rate):
        super().__init__()

        self.lin0 = nn.Linear(in_features=io_channels, out_features=feedforward_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.lin1 = nn.Linear(in_features=feedforward_dim, out_features=io_channels)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin0.weight)
        nn.init.zeros_(self.lin0.bias)

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)

    def forward(self, x):
        x = self.lin0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin1(x)
        return x


class TransformerLayer(nn.Module):
    """
    Transformer block combining attention layer and feedforward network with residual connections
    and layer normalization.

    Args:
        io_channels (int): Number of input and output channels.
        d_model (int): Dimensionality of attention.
        feedforward_dim (int): Feedforward hidden layer size.
        drop_rate (float): Dropout rate.
        attn_width (int, optional): Local attention window size.

    Forward Input Shape:
        Tensor of shape (N, C, L)

    Forward Output:
        Tuple containing:
            - Tensor (N, C, L): Output features.
            - Tensor (N, L, L): Attention weights.
    """

    def __init__(
        self, io_channels, d_model, feedforward_dim, drop_rate, attn_width=None
    ):
        super().__init__()

        self.attn = AttentionLayer(
            in_channels=io_channels, d_model=d_model, attn_width=attn_width
        )
        self.ln0 = nn.LayerNorm(normalized_shape=io_channels)

        self.ff = FeedForward(
            io_channels=io_channels,
            feedforward_dim=feedforward_dim,
            drop_rate=drop_rate,
        )
        self.ln1 = nn.LayerNorm(normalized_shape=io_channels)

    def forward(self, x):
        x1, w = self.attn(x)
        x2 = x1 + x
        # (N,C,L) -> (N,L,C)
        x2 = x2.permute(0, 2, 1)
        x2 = self.ln0(x2)
        x3 = self.ff(x2)
        x4 = x3 + x2
        x4 = self.ln1(x4)
        # (N,L,C) -> (N,C,L)
        x4 = x4.permute(0, 2, 1)
        return x4, w


class Encoder(nn.Module):
    """
    Encoder module consisting of multiple convolutional blocks, residual convolutional blocks,
    bidirectional LSTMs, and transformer layers.

    Args:
        in_channels (int): Number of input channels.
        conv_channels (list of int): Output channels for convolutional layers.
        conv_kernels (list of int): Kernel sizes for convolutional layers.
        resconv_kernels (list of int): Kernel sizes for residual convolutional layers.
        num_lstm_blocks (int): Number of BiLSTM blocks.
        num_transformer_layers (int): Number of transformer layers.
        transformer_io_channels (int): Input/output channels of transformer layers.
        transformer_d_model (int): Dimensionality of transformer attention.
        feedforward_dim (int): Feedforward network hidden size.
        drop_rate (float): Dropout rate.
        conv_kernel_l1_regularization (float): L1 regularization coefficient for conv weights.
        conv_bias_l1_regularization (float): L1 regularization coefficient for conv biases.

    Forward Input Shape:
        Tensor of shape (N, in_channels, L)

    Forward Output:
        Tuple of:
            - Tensor (N, transformer_io_channels, L_out): Encoded features.
            - Tensor (N, L_out, L_out): Attention weights from last transformer layer.

    Note:
        L1 regularization was only applied to the convolution blocks of the first stage.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: list,
        conv_kernels: list,
        resconv_kernels: list,
        num_lstm_blocks: int,
        num_transformer_layers: int,
        transformer_io_channels: int,
        transformer_d_model: int,
        feedforward_dim: int,
        drop_rate: float,
        conv_kernel_l1_regularization: float = 0.0,
        conv_bias_l1_regularization: float = 0.0,
    ):
        super().__init__()

        # Conv 1D
        self.convs = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=kers,
                    kernel_l1_alpha=conv_kernel_l1_regularization,
                    bias_l1_alpha=conv_bias_l1_regularization,
                )
                for inc, outc, kers in zip(
                    [in_channels] + conv_channels[:-1], conv_channels, conv_kernels
                )
            ]
        )

        # Res CNN
        self.res_convs = nn.Sequential(
            *[
                ResConvBlock(
                    io_channels=conv_channels[-1], kernel_size=kers, drop_rate=drop_rate
                )
                for kers in resconv_kernels
            ]
        )

        # Bi-LSTM
        self.bilstms = nn.Sequential(
            *[
                BiLSTMBlock(in_channels=inc, out_channels=outc, drop_rate=drop_rate)
                for inc, outc in zip(
                    [conv_channels[-1]]
                    + [transformer_io_channels] * (num_lstm_blocks - 1),
                    [transformer_io_channels] * num_lstm_blocks,
                )
            ]
        )

        # Transformer
        self.transformers = nn.ModuleList(
            [
                TransformerLayer(
                    io_channels=transformer_io_channels,
                    d_model=transformer_d_model,
                    feedforward_dim=feedforward_dim,
                    drop_rate=drop_rate,
                )
                for _ in range(num_transformer_layers)
            ]
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.res_convs(x)
        x = self.bilstms(x)

        for transformer_ in self.transformers:
            x, w = transformer_(x)

        return x, w


class UpSamplingBlock(nn.Module):
    """
    1D upsampling block with padding, convolution, ReLU activation, and optional L1 regularization.

    This block upsamples the input sequence by a factor of 2 using nearest-neighbor interpolation,
    crops or pads the sequence to a desired length, applies padding to maintain 'same' length after
    convolution, and finally applies a 1D convolution followed by ReLU activation.

    L1 regularization can be optionally applied to convolution weights and biases through
    backward hooks.

    Attributes:
        out_samples (int): Target number of output samples after upsampling and cropping.
        conv_padding_same (Tuple[int, int]): Padding sizes to maintain 'same' output length after convolution.
        upsampling (nn.Upsample): Upsampling layer with scale factor 2.
        conv (nn.Conv1d): 1D convolutional layer.
        relu (nn.ReLU): ReLU activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_samples (int): Desired output sequence length after upsampling and cropping.
        kernel_size (int): Size of the convolution kernel.
        kernel_l1_alpha (float): Non-negative coefficient for L1 regularization on convolution weights.
        bias_l1_alpha (float): Non-negative coefficient for L1 regularization on convolution biases.

    Raises:
        AssertionError: If `kernel_l1_alpha` or `bias_l1_alpha` is negative.

    Forward input shape:
        Tensor of shape (N, C_in, L), where
        N is batch size,
        C_in is number of input channels,
        L is input sequence length.

    Forward output shape:
        Tensor of shape (N, C_out, out_samples), where
        C_out is number of output channels,
        out_samples is the specified output sequence length.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        out_samples,
        kernel_size,
        kernel_l1_alpha,
        bias_l1_alpha,
    ):
        super().__init__()

        assert kernel_l1_alpha >= 0.0
        assert bias_l1_alpha >= 0.0

        self.out_samples = out_samples

        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()

        if kernel_l1_alpha > 0.0:
            self.conv.weight.register_hook(
                lambda grad: grad.data
                + kernel_l1_alpha * torch.sign(self.conv.weight.data)
            )
        if bias_l1_alpha > 0.0:
            self.conv.bias.register_hook(
                lambda grad: grad.data + bias_l1_alpha * torch.sign(self.conv.bias.data)
            )

    def forward(self, x):
        x = self.upsampling(x)
        x = x[:, :, : self.out_samples]
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv(x)
        x = self.relu(x)

        return x


class IdentityNTuple(nn.Identity):
    """
    Identity module that returns the input as-is or repeated in a tuple multiple times.

    This module behaves like `nn.Identity` when `ntuple` is 1, returning the input tensor unchanged.
    When `ntuple` > 1, it returns a tuple containing the input tensor repeated `ntuple` times.
    This is useful for network components that expect tuple outputs, such as multi-headed
    architectures or models that unpack multiple outputs.

    Args:
        ntuple (int): Number of times to repeat the input in the output tuple. Must be >= 1.
            Defaults to 1.
        *args: Additional positional arguments forwarded to `nn.Identity`.
        **kwargs: Additional keyword arguments forwarded to `nn.Identity`.

    Raises:
        AssertionError: If `ntuple` is less than 1.

    Forward input shape:
        Any shape tensor.

    Forward output shape:
        - If `ntuple == 1`: returns a tensor with the same shape as input.
        - If `ntuple > 1`: returns a tuple of length `ntuple` where each element is the input tensor.
    """

    def __init__(self, *args, ntuple: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ntuple >= 1
        self.ntuple = ntuple

    def forward(self, input: torch.Tensor):
        if self.ntuple > 1:
            return (super().forward(input),) * self.ntuple
        else:
            return super().forward(input)


class Decoder(nn.Module):
    """
    Decoder module with optional LSTM and transformer layers, followed by upsampling convolution blocks
    and a final output convolution.

    Args:
        conv_channels (list of int): Channels for upsampling convolution layers.
        conv_kernels (list of int): Kernel sizes for upsampling convolution layers.
        transformer_io_channels (int): Input/output channels of transformer layers.
        transformer_d_model (int): Dimensionality of transformer attention.
        feedforward_dim (int): Feedforward network hidden size.
        drop_rate (float): Dropout rate.
        out_samples (int): Number of output samples after upsampling.
        has_lstm (bool): Whether to include an LSTM layer in decoder.
        has_local_attn (bool): Whether to include local attention transformer layer.
        local_attn_width (int): Local attention window size.
        conv_kernel_l1_regularization (float): L1 regularization coefficient for conv weights.
        conv_bias_l1_regularization (float): L1 regularization coefficient for conv biases.

    Forward Input Shape:
        Tensor of shape (N, transformer_io_channels, L)

    Forward Output Shape:
        Tensor of shape (N, 1, out_samples), with sigmoid activation.
    """

    def __init__(
        self,
        conv_channels: list,
        conv_kernels: list,
        transformer_io_channels: int,
        transformer_d_model: int,
        feedforward_dim: int,
        drop_rate: float,
        out_samples,
        has_lstm: bool = True,
        has_local_attn: bool = True,
        local_attn_width: int = 3,
        conv_kernel_l1_regularization: float = 0.0,
        conv_bias_l1_regularization: float = 0.0,
    ):
        super().__init__()

        self.lstm = (
            nn.LSTM(
                input_size=transformer_io_channels,
                hidden_size=transformer_io_channels,
                batch_first=True,
                bidirectional=False,
            )
            if has_lstm
            else IdentityNTuple(ntuple=2)
        )

        self.lstm_dropout = nn.Dropout(p=drop_rate) if has_lstm else nn.Identity()

        self.transformer = (
            TransformerLayer(
                io_channels=transformer_io_channels,
                d_model=transformer_d_model,
                feedforward_dim=feedforward_dim,
                drop_rate=drop_rate,
                attn_width=local_attn_width,
            )
            if has_local_attn
            else IdentityNTuple(ntuple=2)
        )

        crop_sizes = [out_samples]
        for _ in range(len(conv_kernels) - 1):
            crop_sizes.insert(0, math.ceil(crop_sizes[0] / 2))

        self.upsamplings = nn.Sequential(
            *[
                UpSamplingBlock(
                    in_channels=inc,
                    out_channels=outc,
                    out_samples=crop,
                    kernel_size=kers,
                    kernel_l1_alpha=conv_kernel_l1_regularization,
                    bias_l1_alpha=conv_bias_l1_regularization,
                )
                for inc, outc, crop, kers in zip(
                    [transformer_io_channels] + conv_channels[:-1],
                    conv_channels,
                    crop_sizes,
                    conv_kernels,
                )
            ]
        )

        self.conv_out = nn.Conv1d(
            in_channels=conv_channels[-1], out_channels=1, kernel_size=11, padding=5
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.transformer(x)
        x = self.upsamplings(x)
        x = self.conv_out(x)
        x = x.sigmoid()
        return x


class EQTransformer(nn.Module):
    """
    EQTransformer model for earthquake detection and seismic phase picking.

    This model combines convolutional, BiLSTM, and transformer components to analyze
    seismic waveform data. It outputs detection probabilities for earthquakes and
    picks P and S seismic phases.

    Attributes:
        encoder (Encoder): Feature extractor combining convolutions, BiLSTM, and transformer layers.
        decoders (nn.ModuleList): List of decoder branches for event detection, P-pick, and S-pick.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        in_samples (int): Length of the input waveform sequence. Default is 8192.
        conv_channels (List[int]): Number of output channels for each convolutional layer.
        conv_kernels (List[int]): Kernel sizes for each convolutional layer.
        resconv_kernels (List[int]): Kernel sizes for residual convolutional layers.
        num_lstm_blocks (int): Number of BiLSTM blocks in the encoder.
        num_transformer_layers (int): Number of transformer encoder layers.
        transformer_io_channels (int): Number of channels used within transformer layers.
        transformer_d_model (int): Dimensionality of the transformer attention mechanism.
        feedforward_dim (int): Hidden size of feedforward layers in transformer blocks.
        local_attention_width (int): Width of the local attention window used in decoders.
        drop_rate (float): Dropout rate applied across the network.
        decoder_with_attn_lstm (List[bool]): Flags for each decoder branch indicating usage of attention and LSTM.
        conv_kernel_l1_regularization (float): L1 regularization coefficient for convolution weights.
        conv_bias_l1_regularization (float): L1 regularization coefficient for convolution biases.
        **kwargs: Additional keyword arguments (currently unused).

    Raises:
        AssertionError: If the length of `conv_channels` does not match `conv_kernels`.

    Forward Input Shape:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_samples).

    Forward Output Shape:
        torch.Tensor: Output tensor of shape (batch_size, 3, in_samples) with:
            - channel 0: earthquake detection probability
            - channel 1: P-phase pick probability
            - channel 2: S-phase pick probability
    """

    _epsilon = 1e-6

    def __init__(
        self,
        in_channels=3,
        in_samples=8192,
        conv_channels=[8, 16, 16, 32, 32, 64, 64],
        conv_kernels=[11, 9, 7, 7, 5, 5, 3],
        resconv_kernels=[3, 3, 3, 2, 2],
        num_lstm_blocks=3,
        num_transformer_layers=2,
        transformer_io_channels=16,
        transformer_d_model=32,
        feedforward_dim=128,
        local_attention_width=3,
        drop_rate=0.1,
        decoder_with_attn_lstm=[False, True, True],
        conv_kernel_l1_regularization: float = 0.0,
        conv_bias_l1_regularization: float = 0.0,
        **kwargs
    ):
        super().__init__()

        assert len(conv_channels) == len(conv_kernels)

        self.in_channels = in_channels
        self.in_samples = in_samples
        self.drop_rate = drop_rate
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.resconv_kernels = resconv_kernels
        self.num_lstm_blocks = num_lstm_blocks
        self.num_transformer_layers = num_transformer_layers
        self.transformer_io_channels = transformer_io_channels
        self.transformer_d_model = transformer_d_model
        self.feedforward_dim = feedforward_dim
        self.decoder_with_attn_lstm = decoder_with_attn_lstm

        self.encoder = Encoder(
            in_channels=self.in_channels,
            conv_channels=self.conv_channels,
            conv_kernels=self.conv_kernels,
            resconv_kernels=self.resconv_kernels,
            num_lstm_blocks=self.num_lstm_blocks,
            num_transformer_layers=self.num_transformer_layers,
            transformer_io_channels=self.transformer_io_channels,
            transformer_d_model=self.transformer_d_model,
            feedforward_dim=self.feedforward_dim,
            drop_rate=self.drop_rate,
            conv_kernel_l1_regularization=conv_kernel_l1_regularization,
            conv_bias_l1_regularization=conv_bias_l1_regularization,
        )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    conv_channels=self.conv_channels[::-1],
                    conv_kernels=self.conv_kernels[::-1],
                    transformer_io_channels=self.transformer_io_channels,
                    transformer_d_model=self.transformer_d_model,
                    feedforward_dim=self.feedforward_dim,
                    drop_rate=self.drop_rate,
                    out_samples=self.in_samples,
                    has_lstm=has_attn_lstm,
                    has_local_attn=has_attn_lstm,
                    local_attn_width=local_attention_width,
                    conv_kernel_l1_regularization=conv_kernel_l1_regularization,
                    conv_bias_l1_regularization=conv_bias_l1_regularization,
                )
                for has_attn_lstm in self.decoder_with_attn_lstm
            ]
        )

    def forward(self, x):
        feature, _ = self.encoder(x)
        outputs = [decoder(feature) for decoder in self.decoders]
        return torch.cat(outputs, dim=1)


@register_model
def eqtransformer(**kwargs):
    """
    Constructs and returns an instance of the EQTransformer model for earthquake detection and phase picking.

    This function wraps the EQTransformer class and passes all keyword arguments directly to its constructor.
    It is intended to be used in model registry systems where models are created dynamically by name.

    Keyword Args:
        in_channels (int): Number of input channels. Defaults to 3.
        in_samples (int): Number of input samples. Defaults to 8192.
        conv_channels (List[int]): Output channels for each convolutional layer.
        conv_kernels (List[int]): Kernel sizes for each convolutional layer.
        resconv_kernels (List[int]): Kernel sizes for residual convolutional layers.
        num_lstm_blocks (int): Number of BiLSTM blocks in the encoder.
        num_transformer_layers (int): Number of transformer layers in the encoder.
        transformer_io_channels (int): Number of channels in transformer layers.
        transformer_d_model (int): Dimensionality of the transformer attention mechanism.
        feedforward_dim (int): Hidden size of the transformer feedforward network.
        local_attention_width (int): Width of local attention window.
        drop_rate (float): Dropout rate applied throughout the model.
        decoder_with_attn_lstm (List[bool]): Indicates whether each decoder branch uses LSTM and attention.
        conv_kernel_l1_regularization (float): L1 regularization coefficient for convolutional weights.
        conv_bias_l1_regularization (float): L1 regularization coefficient for convolutional biases.
        **kwargs: Additional keyword arguments passed to the EQTransformer constructor.

    Returns:
        EQTransformer: An initialized instance of the EQTransformer model.
    """
    model = EQTransformer(**kwargs)
    return model
