"""SpectralNHITS: dual-head classifier with hierarchical NHITS stacks."""

from __future__ import annotations

from math import e
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from neuralforecast.models.nhits import NHITSBlock, _IdentityBasis


def _default_global_stacks(hidden_dim: int, n_stacks: int = 4) -> dict:
    """Default stack config for global view (L ~ 2001)."""
    if n_stacks == 3:
        pool = [16, 8, 4]
        freq = [4, 2, 1]
    elif n_stacks == 4:
        pool = [32, 16, 8, 4]
        freq = [8, 4, 2, 1]
        
    elif n_stacks == 6:
        pool = [32, 16, 8, 4, 2, 1]
        freq = [8, 4, 2, 2, 1, 1]
    else:
        raise ValueError("n_stacks must be 3 or 4 for defaults")
    return {
        "n_blocks": [1] * n_stacks,
        "n_pool_kernel_size": pool,
        "n_freq_downsample": freq,
        "mlp_units": [[[hidden_dim, hidden_dim]]] * n_stacks,
    }


def _default_local_stacks(hidden_dim: int, n_stacks: int = 4) -> dict:
    """Default stack config for local view (L ~ 201)."""
    if n_stacks == 3:
        pool = [8, 4, 2]
        freq = [4, 2, 1]
    elif n_stacks == 4:
        pool = [16, 8, 4, 2]
        freq = [4, 2, 2, 1]
    else:
        raise ValueError("n_stacks must be 3 or 4 for defaults")
    return {
        "n_blocks": [1] * n_stacks,
        "n_pool_kernel_size": pool,
        "n_freq_downsample": freq,
        "mlp_units": [[[hidden_dim, hidden_dim]]] * n_stacks,
    }


def _validate_stack_config(
    n_blocks: Sequence[int],
    n_pool_kernel_size: Sequence[int],
    n_freq_downsample: Sequence[int],
    mlp_units: Sequence[Sequence[Sequence[int]]],
    label: str,
) -> int:
    n_stacks = len(n_blocks)
    if not (
        len(n_pool_kernel_size) == n_stacks
        and len(n_freq_downsample) == n_stacks
        and len(mlp_units) == n_stacks
    ):
        raise ValueError(
            f"{label}: len(n_blocks), n_pool_kernel_size, n_freq_downsample and "
            f"mlp_units must match (got {len(n_blocks)}, {len(n_pool_kernel_size)}, "
            f"{len(n_freq_downsample)}, {len(mlp_units)})"
        )
    if any(b < 1 for b in n_blocks):
        raise ValueError(f"{label}: each stack needs at least one block")
    return int(sum(n_blocks))


class SpectralNHITS(nn.Module):
    """Dual-head (global/local) classifier with NHITS multi-stack decomposition.

    Each head receives two folded curves (odd/even). NHITS stacks analyse each
    curve with doubly-residual blocks at increasing frequency resolution,
    analogous to a learned wavelet decomposition inside the network.

    Input: binned global/local flux from ``.npz`` exports, shape ``[B, 2, L]``.
  """

    N_CURVES = 2

    def __init__(
        self,
        input_size: Tuple[int, int],
        n_classes: int,
        *,
        n_stacks: int = 4,
        global_n_blocks: Optional[List[int]] = None,
        global_n_pool_kernel_size: Optional[List[int]] = None,
        global_n_freq_downsample: Optional[List[int]] = None,
        global_mlp_units: Optional[List[List[List[int]]]] = None,
        local_n_blocks: Optional[List[int]] = None,
        local_n_pool_kernel_size: Optional[List[int]] = None,
        local_n_freq_downsample: Optional[List[int]] = None,
        local_mlp_units: Optional[List[List[List[int]]]] = None,
        hidden_dim: int = 64,
        h: int = 1,
        feature_mode: str = "per_block",
        dropout_prob: float = 0.35,
        feature_dropout: float = 0.2,
        classifier_width: Tuple[int, ...] = (64, 32),
        interpolation_mode: str = "linear",
        pooling_mode: str = "MaxPool1d",
        activation: str = "ReLU",
        # Legacy: one block per kernel size (maps to n_pool_kernel_size)
        kernel_sizes: Optional[List[int]] = None,
        global_kernel_sizes: Optional[List[int]] = None,
        local_kernel_sizes: Optional[List[int]] = None,
        n_freq_downsample: Optional[List[int]] = None,
        mlp_units: Optional[List[List[int]]] = None,
    ):
        super().__init__()
        if feature_mode not in ("per_block", "final"):
            raise ValueError("feature_mode must be 'per_block' or 'final'")

        self.global_size, self.local_size = input_size
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.h = h
        self.feature_mode = feature_mode
        self.classifier_dropout = dropout_prob
        self.feature_dropout_p = feature_dropout

        global_cfg = _default_global_stacks(hidden_dim, n_stacks)
        local_cfg = _default_local_stacks(hidden_dim, n_stacks)

        if global_kernel_sizes is not None or kernel_sizes is not None:
            kernels = global_kernel_sizes or kernel_sizes
            global_cfg["n_pool_kernel_size"] = list(kernels)
            global_cfg["n_blocks"] = [1] * len(kernels)
            freq = global_n_freq_downsample or n_freq_downsample
            if freq is not None:
                global_cfg["n_freq_downsample"] = list(freq)[: len(kernels)]
        if local_kernel_sizes is not None or kernel_sizes is not None:
            kernels = local_kernel_sizes or kernel_sizes
            local_cfg["n_pool_kernel_size"] = list(kernels)
            local_cfg["n_blocks"] = [1] * len(kernels)
            freq = local_n_freq_downsample or n_freq_downsample
            if freq is not None:
                local_cfg["n_freq_downsample"] = list(freq)[: len(kernels)]

        if global_n_blocks is not None:
            global_cfg["n_blocks"] = list(global_n_blocks)
        if global_n_pool_kernel_size is not None:
            global_cfg["n_pool_kernel_size"] = list(global_n_pool_kernel_size)
        if global_n_freq_downsample is not None:
            global_cfg["n_freq_downsample"] = list(global_n_freq_downsample)
        if global_mlp_units is not None:
            global_cfg["mlp_units"] = list(global_mlp_units)

        if local_n_blocks is not None:
            local_cfg["n_blocks"] = list(local_n_blocks)
        if local_n_pool_kernel_size is not None:
            local_cfg["n_pool_kernel_size"] = list(local_n_pool_kernel_size)
        if local_n_freq_downsample is not None:
            local_cfg["n_freq_downsample"] = list(local_n_freq_downsample)
        if local_mlp_units is not None:
            local_cfg["mlp_units"] = list(local_mlp_units)

        if mlp_units is not None and global_mlp_units is None:
            layer_cfg = [list(mlp_units)]
            global_cfg["mlp_units"] = [layer_cfg] * len(global_cfg["n_blocks"])
            local_cfg["mlp_units"] = [layer_cfg] * len(local_cfg["n_blocks"])

        self.n_global_blocks = _validate_stack_config(
            global_cfg["n_blocks"],
            global_cfg["n_pool_kernel_size"],
            global_cfg["n_freq_downsample"],
            global_cfg["mlp_units"],
            "global",
        )
        self.n_local_blocks = _validate_stack_config(
            local_cfg["n_blocks"],
            local_cfg["n_pool_kernel_size"],
            local_cfg["n_freq_downsample"],
            local_cfg["mlp_units"],
            "local",
        )
        self.n_global_stacks = len(global_cfg["n_blocks"])
        self.n_local_stacks = len(local_cfg["n_blocks"])

        self.global_blocks = self._build_stacks(
            input_size=self.global_size,
            h=h,
            n_blocks=global_cfg["n_blocks"],
            n_pool_kernel_size=global_cfg["n_pool_kernel_size"],
            n_freq_downsample=global_cfg["n_freq_downsample"],
            mlp_units=global_cfg["mlp_units"],
            interpolation_mode=interpolation_mode,
            pooling_mode=pooling_mode,
            activation=activation,
        )
        self.local_blocks = self._build_stacks(
            input_size=self.local_size,
            h=h,
            n_blocks=local_cfg["n_blocks"],
            n_pool_kernel_size=local_cfg["n_pool_kernel_size"],
            n_freq_downsample=local_cfg["n_freq_downsample"],
            mlp_units=local_cfg["mlp_units"],
            interpolation_mode=interpolation_mode,
            pooling_mode=pooling_mode,
            activation=activation,
        )

        blocks_per_head = (
            self.n_global_blocks + self.n_local_blocks
            if feature_mode == "per_block"
            else 2
        )
        n_features = self.N_CURVES * h * blocks_per_head
        self.feature_norm = nn.LayerNorm(n_features)
        self.feature_dropout = nn.Dropout(self.feature_dropout_p)
        self.classifier = self._build_classifier(
            n_features, n_classes, classifier_width, dropout_prob
        )

    @staticmethod
    def _build_classifier(
        n_features: int,
        n_classes: int,
        widths: Tuple[int, ...],
        dropout_prob: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = n_features
        for width in widths:
            layers.extend(
                [
                    nn.Linear(in_features, width),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
            )
            in_features = width
        layers.append(nn.Linear(in_features, n_classes))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_stacks(
        input_size: int,
        h: int,
        n_blocks: List[int],
        n_pool_kernel_size: List[int],
        n_freq_downsample: List[int],
        mlp_units: List[List[List[int]]],
        interpolation_mode: str,
        pooling_mode: str,
        activation: str,
    ) -> nn.ModuleList:
        blocks: list[NHITSBlock] = []
        for stack_i, n_stack_blocks in enumerate(n_blocks):
            for _ in range(n_stack_blocks):
                n_theta = input_size + max(h // max(n_freq_downsample[stack_i], 1), 1)
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    interpolation_mode=interpolation_mode,
                    out_features=1,
                )
                blocks.append(
                    NHITSBlock(
                        input_size=input_size,
                        h=h,
                        n_theta=n_theta,
                        mlp_units=mlp_units[stack_i],
                        basis=basis,
                        futr_input_size=0,
                        hist_input_size=0,
                        stat_input_size=0,
                        n_pool_kernel_size=n_pool_kernel_size[stack_i],
                        pooling_mode=pooling_mode,
                        dropout_prob=0.0,
                        activation=activation,
                    )
                )
        return nn.ModuleList(blocks)

    @staticmethod
    def _as_curves(x: torch.Tensor, n_curves: int) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != n_curves:
            raise ValueError(
                f"Expected {n_curves} curves per head, got shape {tuple(x.shape)}"
            )
        return x

    def _run_blocks(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        b, seq_len = x.shape
        device, dtype = x.device, x.dtype
        futr_exog = torch.zeros(b, 0, seq_len + self.h, device=device, dtype=dtype)
        hist_exog = torch.zeros(b, 0, seq_len, device=device, dtype=dtype)
        stat_exog = torch.zeros(b, 0, device=device, dtype=dtype)

        insample_mask = torch.ones_like(x)
        residuals = torch.flip(x, dims=(-1,))
        forecast = x[:, -1:, None]
        per_block: list[torch.Tensor] = []

        for block in blocks:
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            if self.feature_mode == "per_block":
                per_block.append(block_forecast.squeeze(-1))

        if self.feature_mode == "per_block":
            return torch.cat(per_block, dim=1)
        return forecast.squeeze(-1)

    def _run_head(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        x = self._as_curves(x, self.N_CURVES)
        forecasts = [self._run_blocks(x[:, c, :], blocks) for c in range(self.N_CURVES)]
        return torch.cat(forecasts, dim=1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_global, x_local = inputs
        global_feat = self._run_head(x_global, self.global_blocks)
        local_feat = self._run_head(x_local, self.local_blocks)
        features = torch.cat([global_feat, local_feat], dim=1)
        features = self.feature_dropout(self.feature_norm(features))
        return self.classifier(features)

    def stack_summary(self) -> str:
        """Human-readable description of stack layout."""
        lines = [
            f"feature_mode={self.feature_mode}, h={self.h}",
            f"global: {self.n_global_stacks} stacks, {self.n_global_blocks} blocks",
            f"local:  {self.n_local_stacks} stacks, {self.n_local_blocks} blocks",
            f"classifier in_features={self.classifier[0].in_features}",
        ]
        return "\n".join(lines)
