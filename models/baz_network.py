"""
BAZ network module.

This module implements the BAZ_Network architecture for Bayesian deep learning estimation of
earthquake location from single-station observations as described in:

Reference:
    [1] S. M. Mousavi and G. C. Beroza. (2020)
        Bayesian-Deep-Learning Estimation of Earthquake Location From Single-Station Observations.
        IEEE Transactions on Geoscience and Remote Sensing, 58, 11, 8211-8224.
        doi: 10.1109/TGRS.2020.2988770.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model


class BAZ_Network(nn.Module):
    """
    BAZ network model for earthquake location estimation from single-station seismic data.

    The model applies a sequence of 1D convolutional layers with ReLU activation, dropout, and
    max pooling to extract features from the input seismic signals. It also computes covariance
    matrices and eigen decomposition of the input for enhanced feature representation.

    The final output layer produces two outputs corresponding to the predicted parameters.

    Args:
        in_channels (int): Number of input channels of the seismic signal.
        in_samples (int): Number of samples per input signal.
        in_matrix_dim (int, optional): Dimension of additional matrix features concatenated after convolutions. Defaults to 7.
        conv_channels (list of int, optional): Number of output channels for each convolutional layer. Defaults to [20, 32, 64, 20].
        kernel_size (int, optional): Kernel size for convolutional layers. Defaults to 3.
        pool_size (int, optional): Kernel size for max pooling layers. Defaults to 2.
        lin_hidden_dim (int, optional): Number of hidden units in the fully connected layer. Defaults to 100.
        drop_rate (float, optional): Dropout probability applied after convolution and in fully connected layers. Defaults to 0.3.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_samples), representing seismic signals.

    Forward Output:
        tuple(torch.Tensor, torch.Tensor): Two tensors each of shape (batch_size, 1),
            representing the model's two predicted outputs.

    Methods:
        _cov(x): Compute batch covariance matrices for input tensor x.
        _eig(cov, dtype): Compute eigenvalues and eigenvectors of covariance matrices.
        _compute_cov_and_eig(x): Compute covariance matrices, eigenvalues, and eigenvectors, and concatenate them.
    """

    def __init__(
        self,
        in_channels: int,
        in_samples: int,
        in_matrix_dim: int = 7,
        conv_channels: list = [20, 32, 64, 20],
        kernel_size: int = 3,
        pool_size: int = 2,
        lin_hidden_dim: int = 100,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        dim = in_samples
        for inc, outc in zip([in_channels] + conv_channels[:-1], conv_channels):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=inc,
                        out_channels=outc,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.MaxPool1d(pool_size, ceil_mode=True),
                )
            )
            dim = (dim + (pool_size - (dim % pool_size)) % pool_size) // pool_size
        dim = (dim + in_matrix_dim) * conv_channels[-1]

        self.flatten0 = nn.Flatten()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=conv_channels[-1], kernel_size=1
        )
        self.relu0 = nn.ReLU()
        self.flatten1 = nn.Flatten()

        self.lin0 = nn.Linear(in_features=dim, out_features=lin_hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self.lin1 = nn.Linear(in_features=lin_hidden_dim, out_features=2)

    @torch.no_grad()
    def _cov(self, x: torch.Tensor):
        """
        Compute batch covariance matrices for input tensor x.

        torch.cov does not support batch processing, so this method computes covariance matrices
        independently for each sample in the batch.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Covariance matrices of shape (batch_size, channels, channels).
        """
        N, C, L = x.size()

        # Center the data
        diff = (x - x.mean(-1, keepdim=True)).transpose(-1, -2).reshape(N * L, C)

        # Compute outer product and reshape
        cov = torch.bmm(diff.unsqueeze(-1), diff.unsqueeze(-2)).view(N, L, C, C).sum(
            dim=1
        ) / (L - 1)
        return cov

    @torch.no_grad()
    def _eig(self, cov: torch.Tensor, dtype: torch.dtype = torch.float32):
        """
        Compute eigenvalues and eigenvectors of covariance matrices.

        Returns only the real part of eigenvalues and eigenvectors.

        Args:
            cov (torch.Tensor): Covariance matrices of shape (batch_size, channels, channels).
            dtype (torch.dtype, optional): Output dtype. Defaults to torch.float32.

        Returns:
            tuple: eigenvalues (batch_size, channels, 1), eigenvectors (batch_size, channels, channels).
        """
        eig_values, eig_vectors = torch.linalg.eig(cov)
        eig_values, eig_vectors = eig_values.unsqueeze(-1).type(
            dtype
        ), eig_vectors.type(dtype)
        return eig_values, eig_vectors

    @torch.no_grad()
    def _compute_cov_and_eig(self, x):
        """
        Compute covariance matrix, eigenvalues, and eigenvectors of input tensor and normalize them.

        Concatenate normalized covariance, eigenvalues, and eigenvectors along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Concatenated tensor of shape (batch_size, channels, channels + 1 + channels).
        """
        cov_mat = self._cov(x)

        eig_values, eig_vectors = self._eig(cov_mat, x.dtype)

        eig_values /= eig_values.max()
        cov_mat /= cov_mat.abs().max()
        x = torch.cat([cov_mat, eig_values, eig_vectors], dim=-1)

        return x

    def forward(self, x):
        x1 = self._compute_cov_and_eig(x)

        for layer in self.layers:
            x = layer(x)

        x = self.flatten0(x)

        x1 = self.conv1(x1)
        x1 = self.relu0(x1)
        x1 = self.flatten1(x1)

        x = torch.cat([x, x1], dim=1)
        x = self.lin0(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.lin1(x)

        return x[:, :1], x[:, 1:]


@register_model
def baz_network(**kwargs):
    """
    Factory function to create a BAZ_Network model instance.

    Args:
        **kwargs: Arbitrary keyword arguments passed to BAZ_Network constructor.

    Returns:
        BAZ_Network: Instantiated BAZ_Network model.
    """
    model = BAZ_Network(**kwargs)
    return model
