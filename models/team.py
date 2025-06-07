import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class MLP(nn.Module):
    def __init__(
        self, input_shape, dims=(100, 50), activation="relu", last_activation=None
    ):
        super().__init__()
        act_fn = self._get_activation(activation)
        last_act_fn = self._get_activation(
            last_activation if last_activation else activation
        )

        layers = []
        in_dim = input_shape if isinstance(input_shape, int) else input_shape[0]

        if len(dims) == 1:
            layers.append(nn.Linear(in_dim, dims[0]))
            if last_act_fn:
                layers.append(last_act_fn)
        else:
            # First layer
            layers.append(nn.Linear(in_dim, dims[0]))
            layers.append(act_fn)
            # Hidden layers
            for i in range(1, len(dims) - 1):
                layers.append(nn.Linear(dims[i - 1], dims[i]))
                layers.append(act_fn)
            # Last layer
            layers.append(nn.Linear(dims[-2], dims[-1]))
            if last_act_fn:
                layers.append(last_act_fn)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _get_activation(self, name):
        if name is None:
            return None
        name = name.lower()
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
        }.get(name, nn.ReLU())


class MixtureOutput(nn.Module):
    def __init__(
        self,
        input_shape,
        n,
        d=1,
        activation="relu",
        eps=1e-4,
        bias_mu=1.8,
        bias_sigma=0.2,
        name=None,
    ):
        super().__init__()
        self.name = name
        self.n = n
        self.d = d
        self.eps = eps

        self.act_fn = self._get_activation(activation)

        self.alpha_layer = nn.Sequential(nn.Linear(input_shape, n), nn.Softmax(dim=-1))
        self.mu_layer = nn.Linear(input_shape, n * d)
        self.sigma_layer = nn.Linear(input_shape, n * d)

        nn.init.constant_(self.mu_layer.bias, bias_mu)
        nn.init.constant_(self.sigma_layer.bias, bias_sigma)

    def forward(self, x):
        alpha = self.alpha_layer(x).unsqueeze(-1)  # (B, n, 1)
        mu = self.act_fn(self.mu_layer(x)).view(-1, self.n, self.d)  # (B, n, d)
        sigma = F.relu(self.sigma_layer(x)).view(-1, self.n, self.d)  # (B, n, d)
        sigma = sigma + self.eps
        return torch.cat([alpha, mu, sigma], dim=2)

    def _get_activation(self, name):
        if name is None:
            return nn.Identity()
        return {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
        }.get(name.lower(), nn.ReLU())


class NormalizedScaleEmbedding(nn.Module):
    def __init__(
        self,
        input_shape,
        activation="relu",
        downsample=1,
        mlp_dims=(500, 300, 200, 150),
        eps=1e-8,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        self.downsample = downsample
        self.eps = eps

        T, C = input_shape
        self.conv2d_1 = nn.Conv2d(
            1, 8, kernel_size=(downsample, 1), stride=(downsample, 1)
        )
        self.conv2d_2 = nn.Conv2d(8, 32, kernel_size=(16, 3), stride=(1, 3))

        time_out = T // downsample - 15
        feature_out = 32 * (C // 3)

        self.conv1d_stack = nn.Sequential(
            nn.Conv1d(feature_out, 64, 16),
            self.activation,
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 16),
            self.activation,
            nn.MaxPool1d(2),
            nn.Conv1d(128, 32, 8),
            self.activation,
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 8),
            self.activation,
            nn.Conv1d(32, 16, 4),
            self.activation,
        )

        self.flatten = nn.Flatten()
        self.mlp = MLP(input_shape=865, dims=mlp_dims, activation=activation)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        max_val = torch.amax(torch.abs(x), dim=(1, 2), keepdim=True) + self.eps
        x_norm = x / max_val

        scale = torch.log(torch.amax(torch.abs(x), dim=(1, 2)) + self.eps) / 100.0
        scale = scale.unsqueeze(1)  # (B, 1)

        x = x_norm.unsqueeze(1)  # (B, 1, T, C)
        x = self.activation(self.conv2d_1(x))
        x = self.activation(self.conv2d_2(x))

        B, C2, T2, W2 = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T2, C2 * W2).permute(0, 2, 1)
        x = self.conv1d_stack(x)
        x = self.flatten(x)
        x = torch.cat([x, scale], dim=1)
        x = self.mlp(x)
        return x

    def _get_activation(self, name):
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
        }.get(name.lower(), nn.ReLU())


class Transformer(nn.Module):
    def __init__(
        self,
        max_stations=32,
        emb_dim=500,
        layers=6,
        att_masking=False,
        hidden_dropout=0.0,
        mad_params={},
        ffn_params={},
        norm_params={},
    ):
        super().__init__()
        self.max_stations = max_stations  # để tương thích với bản gốc
        self.att_masking = att_masking
        self.hidden_dropout = hidden_dropout

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            attn = MultiHeadSelfAttention(**mad_params)
            ffn = PointwiseFeedForward(emb_dim, **ffn_params)
            norm1 = LayerNormalization(**norm_params)
            norm2 = LayerNormalization(**norm_params)
            self.blocks.append(nn.ModuleList([attn, ffn, norm1, norm2]))

        self.dropout = (
            nn.Dropout(hidden_dropout) if hidden_dropout > 0 else nn.Identity()
        )

    def forward(self, x, att_mask=None):
        for attn_layer, ffn_layer, norm1, norm2 in self.blocks:
            residual = x
            if self.att_masking and att_mask is not None:
                attn_out = attn_layer([x, att_mask])
            else:
                attn_out = attn_layer(x)
            attn_out = self.dropout(attn_out)
            x = norm1(x + attn_out)

            residual = x
            ffn_out = ffn_layer(x)
            ffn_out = self.dropout(ffn_out)
            x = norm2(x + ffn_out)

        return x


class PositionEmbedding(nn.Module):
    def __init__(
        self, wavelengths, emb_dim, borehole=False, rotation=None, rotation_anchor=None
    ):
        super().__init__()
        self.wavelengths = wavelengths
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

        if rotation is not None and rotation_anchor is None:
            raise ValueError(
                "Rotations in the positional embedding require a rotation anchor"
            )

        if rotation is not None:
            c, s = np.cos(rotation), np.sin(rotation)
            self.rotation_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        else:
            self.rotation_matrix = None

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]
        min_depth, max_depth = wavelengths[2]

        assert emb_dim % 10 == 0
        if borehole:
            assert emb_dim % 20 == 0

        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10
        if borehole:
            depth_dim = emb_dim // 20

        self.lat_coeff = (
            2
            * math.pi
            / min_lat
            * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        )
        self.lon_coeff = (
            2
            * math.pi
            / min_lon
            * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        )
        self.depth_coeff = (
            2
            * math.pi
            / min_depth
            * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))
        )

        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9

        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        if borehole:
            depth_dim *= 2
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = (
            2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        )
        self.mask = torch.tensor(self.mask, dtype=torch.long)
        self.fake_borehole = False

    def forward(self, x, mask=None):
        B, S, C = x.shape
        device = x.device

        if C == 3:
            self.fake_borehole = True

        lat = x[:, :, 0:1]
        lon = x[:, :, 1:2]
        depth = x[:, :, 2:3]

        lat_coeff = torch.tensor(self.lat_coeff, device=device).view(1, 1, -1)
        lon_coeff = torch.tensor(self.lon_coeff, device=device).view(1, 1, -1)
        depth_coeff = torch.tensor(self.depth_coeff, device=device).view(1, 1, -1)

        if self.rotation is not None:
            lon = lon * torch.cos(lat * math.pi / 180)
            lat_r = lat - self.rotation_anchor[0]
            lon_r = lon - self.rotation_anchor[1] * math.cos(
                self.rotation_anchor[0] * math.pi / 180
            )
            latlon = torch.cat([lat_r, lon_r], dim=-1)  # (B, S, 2)
            rot_mat = self.rotation_matrix.to(device)
            rotated = torch.matmul(latlon, rot_mat)  # (B, S, 2)
            lat_base = rotated[:, :, 0:1] * lat_coeff
            lon_base = rotated[:, :, 1:2] * lon_coeff
        else:
            lat_base = lat * lat_coeff
            lon_base = lon * lon_coeff

        if self.borehole:
            if self.fake_borehole:
                depth_base = depth * depth_coeff * 0
                depth2_base = depth * depth_coeff
            else:
                depth2_base = x[:, :, 3:4] * depth_coeff
                depth_base = depth * depth_coeff

            output = torch.cat(
                [
                    torch.sin(lat_base),
                    torch.cos(lat_base),
                    torch.sin(lon_base),
                    torch.cos(lon_base),
                    torch.sin(depth_base),
                    torch.cos(depth_base),
                    torch.sin(depth2_base),
                    torch.cos(depth2_base),
                ],
                dim=-1,
            )
        else:
            depth_base = depth * depth_coeff
            output = torch.cat(
                [
                    torch.sin(lat_base),
                    torch.cos(lat_base),
                    torch.sin(lon_base),
                    torch.cos(lon_base),
                    torch.sin(depth_base),
                    torch.cos(depth_base),
                ],
                dim=-1,
            )

        output = torch.index_select(output, dim=-1, index=self.mask.to(device))

        if mask is not None:
            output = output * mask.unsqueeze(-1).float()

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, n_heads, emb_dim=500, att_masking=False, att_dropout=0.0, infinity=1e6
    ):
        super().__init__()
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.att_masking = att_masking
        self.att_dropout = att_dropout
        self.infinity = infinity

        assert emb_dim % n_heads == 0
        self.d_key = emb_dim // n_heads

        # Linear projections
        self.WQ = nn.Linear(emb_dim, emb_dim)
        self.WK = nn.Linear(emb_dim, emb_dim)
        self.WV = nn.Linear(emb_dim, emb_dim)
        self.WO = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(att_dropout) if att_dropout > 0 else nn.Identity()

    def forward(self, x, mask=None):
        if self.att_masking and isinstance(x, (list, tuple)):
            x, att_mask = x
            if mask is not None:
                mask = mask[0]
        else:
            att_mask = None

        B, S, E = x.shape
        H = self.n_heads
        D = self.d_key

        # Linear projection and reshape
        q = self.WQ(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        k = self.WK(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        v = self.WV(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)

        # Attention score
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, S, S)

        if mask is not None:
            inv_mask = (~mask).unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, S)
            score = score - inv_mask * self.infinity

        if att_mask is not None:
            inv_att_mask = (~att_mask).unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, S)
            score = score - inv_att_mask * self.infinity

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to V
        o = torch.matmul(attn, v)  # (B, H, S, D)
        o = o.transpose(1, 2).contiguous().view(B, S, H * D)  # (B, S, E)
        o = self.WO(o)

        if mask is not None:
            o = torch.abs(o * mask.unsqueeze(-1).float())

        return o


class PointwiseFeedForward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Lớp tuyến tính 1: input → hidden
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # Lớp tuyến tính 2: hidden → output (cùng chiều input)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

        # Khởi tạo trọng số và bias
        self._init_weights(kernel_initializer, bias_initializer)

    def forward(self, x, mask=None):
        x = gelu(self.linear1(x))  # GELU giữa 2 lớp tuyến tính
        x = self.linear2(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()  # Áp dụng mask (B, S) → (B, S, 1)
        return x

    def _init_weights(self, kernel_initializer, bias_initializer):
        # Khởi tạo trọng số
        if kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        elif kernel_initializer == "zeros":
            nn.init.zeros_(self.linear1.weight)
            nn.init.zeros_(self.linear2.weight)

        # Khởi tạo bias
        if bias_initializer == "zeros":
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = None  # sẽ khởi tạo trong forward
        self.beta = None

    def forward(self, x, mask=None):
        if self.gamma is None or self.beta is None:
            # Khởi tạo gamma và beta nếu chưa có (lazy init theo input shape)
            feature_dim = x.shape[-1]
            self.gamma = nn.Parameter(torch.ones(feature_dim))
            self.beta = nn.Parameter(torch.zeros(feature_dim))

        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * normed + self.beta

        if mask is not None:
            out = out * mask.unsqueeze(-1).float()  # Mask shape (B, S) → (B, S, 1)

        return out


class AddEventToken(nn.Module):
    def __init__(self, fixed=True, init_range=None):
        super().__init__()
        self.fixed = fixed
        self.init_range = init_range
        self.emb = None  # Lazy initialization

    def forward(self, x, mask=None):
        B, S, E = x.shape
        device = x.device

        # Lazy init embedding vector if not fixed
        if self.emb is None and not self.fixed:
            if self.init_range is None:
                self.emb = nn.Parameter(torch.ones(E, device=device))
            else:
                bound = self.init_range
                self.emb = nn.Parameter(
                    torch.empty(E, device=device).uniform_(-bound, bound)
                )

        # Prepare token to add at beginning
        if self.fixed:
            pad = torch.ones((B, 1, E), device=device)
        else:
            pad = self.emb.expand(B, 1, E)

        # Concatenate the event token at the front
        out = torch.cat([pad, x], dim=1)

        # Also update the mask if provided
        if mask is not None:
            mask_pad = torch.ones((B, 1), dtype=torch.bool, device=device)
            mask = torch.cat([mask_pad, mask], dim=1)
            return out, mask

        return out


class AddConstantToMixture(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask=None):
        mix, const = inputs  # mix: (B, N, 3), const: (B, N)
        const = const.unsqueeze(-1)  # (B, N, 1)

        alpha = mix[..., 0]
        mu = mix[..., 1] + const.squeeze(-1)
        sigma = mix[..., 2]

        out = torch.stack([alpha, mu, sigma], dim=-1)  # (B, N, 3)

        # Mask propagation
        if mask is not None:
            mask1, mask2 = mask
            if mask1 is None:
                mask = mask2
            elif mask2 is None:
                mask = mask1
            else:
                mask = mask1 & mask2

            mask = mask.float()
            while mask.ndim < out.ndim:
                mask = mask.unsqueeze(-1)
            out = out * mask

        return out


class Masking_nd(nn.Module):
    def __init__(self, mask_value=0.0, axis=-1, nodim=False):
        super().__init__()
        self.mask_value = mask_value
        self.axis = axis
        self.nodim = nodim

    def forward(self, inputs):
        if self.nodim:
            # Trả về mask toàn bộ các phần tử != mask_value
            output_mask = inputs != self.mask_value  # shape = inputs.shape
        else:
            # Trả về mask true nếu có ít nhất một giá trị khác mask_value trên trục axis
            output_mask = (inputs != self.mask_value).any(
                dim=self.axis, keepdim=True
            )  # shape broadcastable

        masked_inputs = inputs * output_mask.to(dtype=inputs.dtype)
        return masked_inputs, output_mask


class GetMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return mask


class StripMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class GlobalMaxPooling1DMasked(nn.Module):
    def __init__(self):
        super().__init__()
        self.pseudo_infty = 1e6  # Giá trị lớn để loại bỏ masked elements

    def forward(self, x, mask=None):
        if mask is not None:
            # mask: (B, S) → (B, S, 1) để broadcast
            mask = mask.unsqueeze(-1).float()
            x = x - (1.0 - mask) * self.pseudo_infty
        return torch.max(
            x, dim=1
        ).values  # Global Max Pooling along time/sequence dimension


class SingleStationModel(nn.Module):
    def __init__(
        self,
        input_shape,
        waveform_model_dims,
        output_mlp_dims,
        activation="relu",
        bias_mag_mu=1.8,
        bias_mag_sigma=0.2,
    ):
        super().__init__()
        self.waveform_model = NormalizedScaleEmbedding(
            input_shape, mlp_dims=waveform_model_dims, activation=activation
        )
        self.mlp = MLP(
            (waveform_model_dims[-1],), output_mlp_dims, activation=activation
        )
        self.out_layer = MixtureOutput(
            (output_mlp_dims[-1],),
            5,
            name="magnitude",
            bias_mu=bias_mag_mu,
            bias_sigma=bias_mag_sigma,
        )

    def forward(self, x):
        x = self.waveform_model(x)
        x = self.mlp(x)
        x = self.out_layer(x)
        return x


class TEAM(nn.Module):
    def __init__(
        self,
        max_stations,
        waveform_model_dims=(500, 500, 500),
        output_mlp_dims=(150, 100, 50, 30, 10),
        output_location_dims=(150, 100, 50, 50, 50),
        wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
        mad_params={"n_heads": 10, "att_dropout": 0.0},
        ffn_params={"hidden_dim": 1000},
        transformer_layers=6,
        hidden_dropout=0.0,
        activation="relu",
        n_pga_targets=0,
        location_mixture=5,
        pga_mixture=5,
        magnitude_mixture=5,
        borehole=False,
        bias_mag_mu=1.8,
        bias_mag_sigma=0.2,
        bias_loc_mu=0,
        bias_loc_sigma=1,
        event_token_init_range=None,
        dataset_bias=False,
        n_datasets=None,
        no_event_token=False,
        trace_length=3000,
        downsample=5,
        rotation=None,
        rotation_anchor=None,
        skip_transformer=False,
        alternative_coords_embedding=False,
    ):
        super().__init__()

        self.n_pga_targets = n_pga_targets
        self.no_event_token = no_event_token
        self.skip_transformer = skip_transformer
        self.dataset_bias = dataset_bias

        self.emb_dim = waveform_model_dims[-1]

        # Submodules
        in_channels = 6 if borehole else 3
        metadata_channels = 4 if borehole else 3

        self.waveform_model = NormalizedScaleEmbedding(
            input_shape=(trace_length, in_channels),
            downsample=downsample,
            activation=activation,
            mlp_dims=waveform_model_dims,
        )

        self.mlp_mag = MLP((self.emb_dim,), output_mlp_dims, activation=activation)
        self.output_mag = MixtureOutput(
            (output_mlp_dims[-1],),
            magnitude_mixture,
            bias_mu=bias_mag_mu,
            bias_sigma=bias_mag_sigma,
        )

        self.mlp_loc = MLP((self.emb_dim,), output_location_dims, activation=activation)
        self.output_loc = MixtureOutput(
            (output_location_dims[-1],),
            location_mixture,
            d=3,
            bias_mu=bias_loc_mu,
            bias_sigma=bias_loc_sigma,
            activation="linear",
        )

        self.mlp_pga = MLP((self.emb_dim,), output_mlp_dims, activation=activation)
        self.output_pga = MixtureOutput(
            (output_mlp_dims[-1],),
            pga_mixture,
            activation="linear",
            bias_mu=-5,
            bias_sigma=1,
        )

        self.mask_waveform = Masking_nd(mask_value=0.0, axis=(2, 3))
        self.mask_coords = Masking_nd(mask_value=0.0, axis=-1)
        self.ln = LayerNormalization()
        self.position_embedding = PositionEmbedding(
            wavelengths=wavelength,
            emb_dim=self.emb_dim,
            borehole=borehole,
            rotation=rotation,
            rotation_anchor=rotation_anchor,
        )

        if not self.no_event_token:
            self.add_event_token = AddEventToken(
                fixed=False, init_range=event_token_init_range
            )

        if not skip_transformer:
            transformer_max_stations = (
                max_stations + (0 if no_event_token else 1) + n_pga_targets
            )
            mad_params = mad_params.copy()
            mad_params["att_masking"] = n_pga_targets > 0
            self.transformer = Transformer(
                max_stations=transformer_max_stations,
                emb_dim=self.emb_dim,
                layers=transformer_layers,
                hidden_dropout=hidden_dropout,
                mad_params=mad_params,
                ffn_params=ffn_params,
            )

        if dataset_bias:
            assert n_datasets is not None
            self.dataset_embedding = nn.Embedding(n_datasets, 1)
            self.bias_adder = AddConstantToMixture()

        if skip_transformer:
            self.global_pool = GlobalMaxPooling1DMasked()
            self.mlp_pre = MLP(
                (
                    self.emb_dim
                    + (metadata_channels if alternative_coords_embedding else 0),
                ),
                [self.emb_dim, self.emb_dim],
                activation=activation,
            )

    def forward(
        self, waveform, metadata, pga_targets=None, att_mask=None, dataset=None
    ):
        x = self.mask_waveform(waveform)[0]  # (B, S, T, C)
        m = self.mask_coords(metadata)[0]  # (B, S, D)

        x = self.waveform_model(x)  # (B, S, D_emb)
        x = self.ln(x)

        if hasattr(self, "position_embedding"):
            coords = self.position_embedding(m)
            x = x + coords
        else:
            x = torch.cat([x, m], dim=-1)

        if not self.no_event_token:
            x = self.add_event_token(x)

        if self.n_pga_targets:
            pga_coords = self.mask_coords(pga_targets)[0]
            pga_emb = self.position_embedding(pga_coords)
            x = torch.cat([x, pga_emb], dim=1)
            x = self.transformer([x, att_mask])
        elif self.skip_transformer:
            x = self.mlp_pre(x)
            x = self.global_pool(x)
        else:
            x = self.transformer(x)

        if not self.no_event_token:
            event_emb = x if self.skip_transformer else x[:, 0, :]
            out_mag = self.output_mag(self.mlp_mag(event_emb))
            out_loc = self.output_loc(self.mlp_loc(event_emb))
        else:
            out_mag = out_loc = None

        out_pga = None
        if self.n_pga_targets:
            pga_emb = x[:, -self.n_pga_targets :, :]
            out_pga = self.output_pga(self.mlp_pga(pga_emb))

        if self.dataset_bias:
            bias = self.dataset_embedding(dataset).squeeze(-1)
            out_mag = self.bias_adder([out_mag, bias])

        return out_mag, out_loc, out_pga
