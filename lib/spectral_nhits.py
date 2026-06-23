"""SpectralNHITS: dual-head classifier with hierarchical NHITS stacks + Shallue-style conv."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralforecast.models.nhits import NHITSBlock, _IdentityBasis


def _default_global_stacks(hidden_dim: int, n_stacks: int = 4) -> dict:
    """Default stack config for global view (L ~ 2001)."""
    if n_stacks == 3:
        pool, freq = [16, 8, 4], [4, 2, 1]
    elif n_stacks == 4:
        pool, freq = [32, 16, 8, 4], [8, 4, 2, 1]
    elif n_stacks == 6:
        pool, freq = [32, 16, 8, 4, 2, 1], [8, 4, 2, 2, 1, 1]
    else:
        raise ValueError("n_stacks must be 3, 4 or 6 for global defaults")
    return {
        "n_blocks": [1] * n_stacks,
        "n_pool_kernel_size": pool,
        "n_freq_downsample": freq,
        "mlp_units": [[[hidden_dim, hidden_dim]]] * n_stacks,
    }


def _default_local_stacks(hidden_dim: int, n_stacks: int = 4) -> dict:
    """Default stack config for local view (L ~ 201)."""
    if n_stacks == 3:
        pool, freq = [8, 4, 2], [4, 2, 1]
    elif n_stacks == 4:
        pool, freq = [16, 8, 4, 2], [4, 2, 2, 1]
    elif n_stacks == 6:
        pool, freq = [16, 8, 4, 2, 2, 1], [4, 2, 2, 2, 1, 1]
    else:
        raise ValueError("n_stacks must be 3, 4 or 6 for local defaults")
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


class SkipLayer2d(nn.Module):
    """Residual conv block aligned with ShallueModel2DSkip (no BatchNorm for ROCm)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (1, 5),
        stride: int = 1,
        padding: Tuple[int, int] = (0, 2),
        conv_size: int = 2,
        pool_kernel: Tuple[int, int] = (1, 5),
        pool_stride: Tuple[int, int] = (1, 2),
    ):
        super().__init__()
        conv_layers: list[nn.Module] = []
        ch_in = in_channels
        for i in range(conv_size):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        ch_in,
                        out_channels,
                        kernel_size=kernel_size if i == 0 else (1, kernel_size[1]),
                        stride=stride,
                        padding=padding if i == 0 else (0, padding[1]),
                    ),
                    nn.ReLU(),
                ]
            )
            ch_in = out_channels
        conv_layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
        self.conv_branch = nn.Sequential(*conv_layers)

        self.id_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv_branch(x) + self.id_branch(x))


def _build_global_skip_tower(n_bands: int) -> nn.Sequential:
    """Five skip blocks, kernel/pool (1,5) — mirrors Shallue global branch."""
    kw = dict(kernel_size=(1, 5), padding=(0, 2), conv_size=2, pool_kernel=(1, 5), pool_stride=(1, 2))
    return nn.Sequential(
        SkipLayer2d(2, 16, kernel_size=(n_bands, 5), padding=(0, 2), conv_size=2, pool_kernel=(1, 5), pool_stride=(1, 2)),
        SkipLayer2d(16, 32, **kw),
        SkipLayer2d(32, 64, **kw),
        SkipLayer2d(64, 128, **kw),
        SkipLayer2d(128, 256, **kw),
    )


def _build_local_skip_tower(n_bands: int) -> nn.Sequential:
    """Two skip blocks, kernel/pool (1,7) — mirrors Shallue local branch."""
    kw = dict(kernel_size=(1, 5), padding=(0, 2), conv_size=2, pool_kernel=(1, 7), pool_stride=(1, 2))
    return nn.Sequential(
        SkipLayer2d(2, 16, kernel_size=(n_bands, 5), padding=(0, 2), conv_size=2, pool_kernel=(1, 7), pool_stride=(1, 2)),
        SkipLayer2d(16, 32, **kw),
    )


class ShallueSkipConvHead(nn.Module):
    """Conv tower over NHITS bands shaped as [B, 2, n_bands, L] (odd/even × bands)."""

    def __init__(
        self,
        n_bands: int,
        seq_len: int,
        view: str = "global",
        adaptive_pool: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        if view == "global":
            self.tower = _build_global_skip_tower(n_bands)
            adaptive_pool = adaptive_pool or (1, 32)
        elif view == "local":
            self.tower = _build_local_skip_tower(n_bands)
            adaptive_pool = adaptive_pool or (1, 8)
        else:
            raise ValueError("view must be 'global' or 'local'")
        self.view = view
        self.pool_size = adaptive_pool
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool)
        with torch.no_grad():
            dummy = torch.zeros(1, 2, n_bands, seq_len)
            self.out_dim = int(self._encode(dummy).size(1))

    def _encode(self, bands: torch.Tensor) -> torch.Tensor:
        return self.adaptive_pool(self.tower(bands)).view(bands.size(0), -1)

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        return self._encode(bands)


class SpectralNHITS(nn.Module):
    """Dual-head classifier: NHITS frequency bands → Shallue-style skip conv → FC.

    NHITS backcasts per band are stacked with odd/even curves as
    ``[B, 2, n_bands, L]``, then processed by skip-conv towers matching
    Shallue global (deep, pool 5) and local (shallow, pool 7).
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
        feature_mode: str = "backcast_conv",
        global_adaptive_pool: Tuple[int, int] = (1, 32),
        local_adaptive_pool: Tuple[int, int] = (1, 8),
        dropout_prob: float = 0.2,
        feature_dropout: float = 0.1,
        classifier_width: Tuple[int, ...] = (512, 256, 64),
        interpolation_mode: str = "linear",
        pooling_mode: str = "MaxPool1d",
        activation: str = "ReLU",
        kernel_sizes: Optional[List[int]] = None,
        global_kernel_sizes: Optional[List[int]] = None,
        local_kernel_sizes: Optional[List[int]] = None,
        n_freq_downsample: Optional[List[int]] = None,
        mlp_units: Optional[List[List[int]]] = None,
    ):
        super().__init__()
        if feature_mode not in ("backcast_conv", "per_block", "final"):
            raise ValueError(
                "feature_mode must be 'backcast_conv', 'per_block' or 'final'"
            )

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

        if feature_mode == "backcast_conv":
            self.global_conv = ShallueSkipConvHead(
                n_bands=self.n_global_blocks,
                seq_len=self.global_size,
                view="global",
                adaptive_pool=global_adaptive_pool,
            )
            self.local_conv = ShallueSkipConvHead(
                n_bands=self.n_local_blocks,
                seq_len=self.local_size,
                view="local",
                adaptive_pool=local_adaptive_pool,
            )
            n_features = self.global_conv.out_dim + self.local_conv.out_dim
        else:
            self.global_conv = None
            self.local_conv = None
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
        per_block_forecast: list[torch.Tensor] = []
        per_block_backcast: list[torch.Tensor] = []

        for block in blocks:
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            per_block_backcast.append(backcast)
            per_block_forecast.append(block_forecast.squeeze(-1))

        bands = torch.stack(per_block_backcast, dim=1)
        if self.feature_mode == "backcast_conv":
            return bands
        if self.feature_mode == "per_block":
            return torch.cat(per_block_forecast, dim=1)
        return forecast.squeeze(-1)

    def _bands_from_head(
        self, x: torch.Tensor, blocks: nn.ModuleList
    ) -> torch.Tensor:
        """NHITS backcasts for odd/even → [B, 2, n_bands, L]."""
        x = self._as_curves(x, self.N_CURVES)
        curve_bands = [self._run_blocks(x[:, c, :], blocks) for c in range(self.N_CURVES)]
        return torch.stack(curve_bands, dim=1)

    def _run_head(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        conv_head: Optional[ShallueSkipConvHead],
    ) -> torch.Tensor:
        if self.feature_mode == "backcast_conv":
            assert conv_head is not None
            return conv_head(self._bands_from_head(x, blocks))

        x_curves = self._as_curves(x, self.N_CURVES)
        curve_feats = [
            self._run_blocks(x_curves[:, c, :], blocks) for c in range(self.N_CURVES)
        ]
        return torch.cat(curve_feats, dim=1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_global, x_local = inputs
        global_feat = self._run_head(x_global, self.global_blocks, self.global_conv)
        local_feat = self._run_head(x_local, self.local_blocks, self.local_conv)
        features = torch.cat([global_feat, local_feat], dim=1)
        features = self.feature_dropout(self.feature_norm(features))
        return self.classifier(features)

    def stack_summary(self) -> str:
        lines = [
            f"feature_mode={self.feature_mode}, h={self.h}",
            f"global: {self.n_global_stacks} stacks, {self.n_global_blocks} NHITS blocks",
            f"local:  {self.n_local_stacks} stacks, {self.n_local_blocks} NHITS blocks",
        ]
        if self.feature_mode == "backcast_conv" and self.global_conv is not None:
            lines.append(
                f"global skip-conv out: {self.global_conv.out_dim} "
                f"(5×SkipLayer + pool {self.global_conv.pool_size})"
            )
            lines.append(
                f"local skip-conv out:  {self.local_conv.out_dim} "
                f"(2×SkipLayer + pool {self.local_conv.pool_size})"
            )
        lines.append(f"classifier in_features={self.classifier[0].in_features}")
        return "\n".join(lines)
