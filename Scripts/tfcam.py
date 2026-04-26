"""
tfcam.py  –  Temporal-Feature Cross Attention Mechanism (TFCAM)
================================================================
Paper: "No Black Box Anymore: Demystifying Clinical Predictive Modeling
        with Temporal-Feature Cross Attention Mechanism"
        Li, Yao, Padman — AMIA 2025  (arXiv:2503.19285)

Usage (from a notebook)
-----------------------
    from tfcam import build_tfcam, TFCAMConfig

    cfg = TFCAMConfig(n_features=30, n_timesteps=8)
    model = build_tfcam(cfg)

    # Or with custom settings:
    cfg = TFCAMConfig(
        n_features   = 30,   # F  — number of clinical features
        n_timesteps  = 8,    # T  — number of time intervals
        d_model      = 64,   # D  — embedding dimension
        n_heads      = 4,    # transformer heads
        n_layers     = 2,    # transformer encoder layers
        lstm_hidden  = 64,   # hidden size of each BiLSTM
        dropout      = 0.1,
        num_classes  = 1,    # 1 → binary (sigmoid), >1 → softmax
    )
    model = build_tfcam(cfg)

    # Forward pass returns (logits, interpretability_dict)
    import torch
    x = torch.randn(16, 8, 30)          # (batch, T, F)
    logits, interp = model(x)

    # interp keys:
    #   "alpha"     – temporal attention weights  (B, T)
    #   "beta"      – feature-level weights       (B, T, F)
    #   "C"         – local contribution matrix   (B, T, F)
    #   "A"         – cross-attention matrix      (B, T, T)
    #   "influence" – chained influence tensor    (B, T, F, T, F)
    #                 index as [b, t, i, t', j]
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFCAMConfig:
    """All hyper-parameters for TFCAM in one place."""
    n_features:  int = 30          # F  – number of input clinical features
    n_timesteps: int = 8           # T  – number of time steps / intervals
    d_model:     int = 64          # D  – projection embedding dimension
    n_heads:     int = 4           # H  – transformer multi-head attention heads
    n_layers:    int = 2           # L  – number of transformer encoder layers
    lstm_hidden: int = 64          # hidden size of each BiLSTM branch
    dropout:     float = 0.1
    num_classes: int = 1           # 1 = binary (BCEWithLogitsLoss)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding (fixed sinusoidal)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the embedding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)               # (max_len, D)
        pos = torch.arange(0, max_len).unsqueeze(1)       # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))       # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Dual BiLSTM Attention  (RETAIN-style dual attention)
# ─────────────────────────────────────────────────────────────────────────────

class DualBiLSTMAttention(nn.Module):
    """
    Two parallel BiLSTM branches:
      Branch-α : produces scalar temporal attention weights  α ∈ (B, T)
      Branch-β : produces feature-level attention weights    β ∈ (B, T, F)

    Local contribution matrix:
      C[b, t, i] = α[b, t] * β[b, t, i] * x[b, t, i]

    Follows §2.3.2 of the paper (inspired by RETAIN but uses BiLSTM).
    """

    def __init__(self, d_model: int, n_features: int, lstm_hidden: int, dropout: float):
        super().__init__()
        # Branch α — temporal attention (scalar per time step)
        self.lstm_alpha = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            batch_first=True, bidirectional=True
        )
        self.fc_alpha = nn.Linear(lstm_hidden * 2, 1)   # → (B, T, 1)

        # Branch β — feature-level attention (one weight per feature per time step)
        self.lstm_beta = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            batch_first=True, bidirectional=True
        )
        self.fc_beta = nn.Linear(lstm_hidden * 2, n_features)  # → (B, T, F)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, E: torch.Tensor, x_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        E      : positionally-encoded embedding  (B, T, D)
        x_raw  : original input features         (B, T, F)

        Returns
        -------
        alpha  : temporal attention weights       (B, T)        softmax over T
        beta   : feature-level attention weights  (B, T, F)     tanh-normalised
        C      : local contribution matrix        (B, T, F)
        """
        # α branch
        out_a, _ = self.lstm_alpha(self.dropout(E))      # (B, T, 2*H)
        alpha = torch.softmax(
            self.fc_alpha(out_a).squeeze(-1), dim=1
        )                                                 # (B, T)

        # β branch
        out_b, _ = self.lstm_beta(self.dropout(E))       # (B, T, 2*H)
        beta = torch.tanh(self.fc_beta(out_b))            # (B, T, F)

        # Local contribution  C[t,i] = α[t] · β[t,i] · x[t,i]
        C = alpha.unsqueeze(-1) * beta * x_raw            # (B, T, F)

        return alpha, beta, C


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Feature Attention  (Transformer encoder)
# ─────────────────────────────────────────────────────────────────────────────

class CrossFeatureAttention(nn.Module):
    """
    Standard Transformer encoder applied over the time dimension.
    Aggregates attention weights across all layers and heads to produce
    the unified cross-attention matrix A ∈ (B, T, T).

    A[b, t, t'] = mean over L layers and H heads of A^{(l,h)}[t, t']
    (§2.3.3 of the paper)
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.n_heads = n_heads
        self.n_layers = n_layers

    def forward(self, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        E : positionally-encoded embeddings  (B, T, D)

        Returns
        -------
        out : transformer output             (B, T, D)
        A   : aggregated cross-attn matrix   (B, T, T)
        """
        # We need raw attention weights, so we call each layer manually.
        attn_matrices = []
        x = E
        for layer in self.encoder.layers:
            # MultiheadAttention returns (output, attn_weights)
            # attn_weights shape: (B, T, T)  averaged over heads internally
            x2, attn = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
            attn_matrices.append(attn)                    # (B, T, T)
            # Manually apply the rest of the encoder layer
            x = layer.norm1(x + layer.dropout1(x2))
            x = layer.norm2(x + layer.dropout2(layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            )))

        # Aggregate: mean over all layers
        A = torch.stack(attn_matrices, dim=0).mean(dim=0)  # (B, T, T)
        return x, A


# ─────────────────────────────────────────────────────────────────────────────
# TFCAM — full model
# ─────────────────────────────────────────────────────────────────────────────

class TFCAM(nn.Module):
    """
    Temporal-Feature Cross Attention Mechanism (TFCAM)

    Forward pass returns
    --------------------
    logits : (B, 1) for binary or (B, num_classes) for multi-class
    interp : dict with keys
        "alpha"      (B, T)
        "beta"       (B, T, F)
        "C"          (B, T, F)   – local contribution matrix
        "A"          (B, T, T)   – aggregated cross-attention matrix
        "influence"  (B, T, F, T, F)  – chained influence I(t,i;t',j)
    """

    def __init__(self, cfg: TFCAMConfig):
        super().__init__()
        self.cfg = cfg
        F = cfg.n_features

        # 1. Input projection  X → E
        self.input_proj = nn.Linear(F, cfg.d_model)

        # 2. Positional encoding
        self.pos_enc = PositionalEncoding(
            cfg.d_model, max_len=cfg.n_timesteps + 16, dropout=cfg.dropout
        )

        # 3. Dual BiLSTM attention branches
        self.dual_attn = DualBiLSTMAttention(
            d_model=cfg.d_model,
            n_features=F,
            lstm_hidden=cfg.lstm_hidden,
            dropout=cfg.dropout,
        )

        # 4. Cross-feature (transformer) attention
        self.cross_attn = CrossFeatureAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )

        # 5. Classifier head
        # Context vector = sum over time of attended representation
        # We pool the transformer output weighted by alpha
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _chained_influence(
        C: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """
        I(t, i; t', j) = C[t, i] * A[t, t'] * C[t', j]

        Args
        ----
        C : (B, T, F)
        A : (B, T, T)

        Returns
        -------
        I : (B, T, F, T, F)
        """
        B, T, F = C.shape
        # Expand dims for broadcasting
        Ct  = C.unsqueeze(3).unsqueeze(4)          # (B, T, F, 1,  1)
        Ctp = C.unsqueeze(1).unsqueeze(2)           # (B, 1,  1, T, F)
        At  = A.unsqueeze(2).unsqueeze(4)           # (B, T,  1, T, 1)
        I   = Ct * At * Ctp                         # (B, T, F, T, F)
        return I

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args
        ----
        x : raw input tensor  (B, T, F)

        Returns
        -------
        logits : (B, num_classes)
        interp : interpretability dict
        """
        # 1. Embed + positional encoding
        E = self.pos_enc(self.input_proj(x))          # (B, T, D)

        # 2. Dual BiLSTM attention
        alpha, beta, C = self.dual_attn(E, x)          # α:(B,T) β:(B,T,F) C:(B,T,F)

        # 3. Cross-feature transformer attention
        trans_out, A = self.cross_attn(E)               # out:(B,T,D) A:(B,T,T)

        # 4. Weighted context vector  (alpha-weighted sum over time)
        context = (trans_out * alpha.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # 5. Classification
        logits = self.classifier(context)               # (B, num_classes)

        # 6. Chained influence (detached — only used for interpretation)
        with torch.no_grad():
            influence = self._chained_influence(C.detach(), A.detach())

        interp = {
            "alpha":     alpha,      # (B, T)
            "beta":      beta,       # (B, T, F)
            "C":         C,          # (B, T, F)
            "A":         A,          # (B, T, T)
            "influence": influence,  # (B, T, F, T, F)
        }

        return logits, interp


# ─────────────────────────────────────────────────────────────────────────────
# Public factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_tfcam(cfg: TFCAMConfig | None = None, **kwargs) -> TFCAM:
    """
    Build and return a TFCAM model.

    Parameters
    ----------
    cfg : TFCAMConfig, optional
        Pass a pre-built config object.  If None, kwargs are forwarded
        to TFCAMConfig (e.g. build_tfcam(n_features=30, n_timesteps=8)).

    Returns
    -------
    model : TFCAM  (not yet on any device — call .to(device) yourself)

    Examples
    --------
    >>> from tfcam import build_tfcam, TFCAMConfig
    >>> model = build_tfcam(n_features=30, n_timesteps=8)
    >>> model = build_tfcam(TFCAMConfig(n_features=30, n_timesteps=8, d_model=128))
    """
    if cfg is None:
        cfg = TFCAMConfig(**kwargs)
    return TFCAM(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check (run this file directly: python tfcam.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = TFCAMConfig(n_features=30, n_timesteps=8, d_model=64, n_heads=4, n_layers=2)
    model = TFCAM(cfg)

    B, T, F = 4, cfg.n_timesteps, cfg.n_features
    x = torch.randn(B, T, F)

    logits, interp = model(x)

    print("=== TFCAM Sanity Check ===")
    print(f"Input shape       : {x.shape}")
    print(f"Logits shape      : {logits.shape}")
    print(f"alpha shape       : {interp['alpha'].shape}")
    print(f"beta  shape       : {interp['beta'].shape}")
    print(f"C     shape       : {interp['C'].shape}")
    print(f"A     shape       : {interp['A'].shape}")
    print(f"influence shape   : {interp['influence'].shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters  : {total:,}")
    print("All shapes OK ✓")
