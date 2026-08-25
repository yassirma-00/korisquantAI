"""Deep-learning architectures for financial time-series forecasting.

All models share the same contract:

    input  : (batch, lookback, n_features)
    output : (batch, n_outputs)   -- default 1 (next-horizon return)

Implemented: LSTM, GRU, 1D-CNN, TCN (dilated causal convolutions),
Transformer encoder, and a hybrid CNN-LSTM.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ------------------------------------------------------------------- LSTM
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, n_outputs: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# -------------------------------------------------------------------- GRU
class GRUForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, n_outputs: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# -------------------------------------------------------------------- TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_in, n_out, kernel, padding=padding, dilation=dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_out, n_out, kernel, padding=padding, dilation=dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    def __init__(self, n_features: int, channels: tuple[int, ...] = (48, 48, 48),
                 kernel: int = 3, dropout: float = 0.2, n_outputs: int = 1) -> None:
        super().__init__()
        layers, n_in = [], n_features
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(n_in, ch, kernel, dilation=2 ** i, dropout=dropout))
            n_in = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(channels[-1]), nn.Linear(channels[-1], n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn(x.transpose(1, 2))
        return self.head(out[:, :, -1])


# ------------------------------------------------------------ Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, layers: int = 2,
                 dim_ff: int = 128, dropout: float = 0.15, n_outputs: int = 1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        # enable_nested_tensor=False: nested tensors are incompatible with
        # norm_first=True and emit a warning while silently disabling the fast path.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.input_proj(x))
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


# --------------------------------------------------------------- CNN-LSTM
class CNNLSTMForecaster(nn.Module):
    def __init__(self, n_features: int, conv_channels: int = 32, hidden: int = 64,
                 dropout: float = 0.2, n_outputs: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :])


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "tcn": TCNForecaster,
    "transformer": TransformerForecaster,
    "cnn_lstm": CNNLSTMForecaster,
}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](n_features=n_features, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
