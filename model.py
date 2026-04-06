import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import json
import os
from typing import Dict, List, Tuple, Optional


class DyT(nn.Module):
    """
    Dynamic Tanh normalization.
    """
    def __init__(self, num_features: int, alpha_init: float = 1.0,
                 beta_init: float = 1.0, gamma_init: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * alpha_init)
        self.beta = nn.Parameter(torch.ones(num_features) * beta_init)
        self.gamma = nn.Parameter(torch.ones(num_features) * gamma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.tanh(self.beta * x + self.gamma)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class DyTTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer using DyT instead of LayerNorm.
    Returns per-head attention weights when requested.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = DyT(d_model, alpha_init=0.8)
        self.norm2 = DyT(d_model, alpha_init=0.8)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,   # IMPORTANT: keep per-head weights
        )

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        if return_attention:
            return src, attn_weights  # (B, heads, T, T)
        return src, None



class ChainOfInfluence(nn.Module):
    """
    Binary mortality classifier with:
    - input projection
    - positional encoding
    - dual BiLSTM branches for temporal and feature attention
    - DyT Transformer encoder stack
    - single-logit classifier head
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        lstm_hidden: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 48,
        use_mask_as_padding: bool = False,   # set True only if mask means true padding
    ):
        super().__init__()

        self.d_model = d_model
        self.n_features = n_features
        self.max_len = max_len
        self.use_mask_as_padding = use_mask_as_padding

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.temporal_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.feature_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.temporal_attn_proj = nn.Linear(2 * lstm_hidden, 1)      # alpha
        self.feature_attn_proj = nn.Linear(2 * lstm_hidden, d_model)  # beta

        self.transformer_layers = nn.ModuleList([
            DyTTransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # single logit
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        x:    (B, T, F)
        mask: (B, T) boolean, True = real/valid timestep
              If your mask is NOT true padding, leave use_mask_as_padding=False.
        """
        B, T, _ = x.shape

        # Input projection + position encoding
        emb = self.input_proj(x)      # (B, T, d_model)
        emb = self.pos_enc(emb)

        # LSTM branches on full sequence
        temporal_out, _ = self.temporal_lstm(emb)  # (B, T, 2*lstm_hidden)
        feat_out, _ = self.feature_lstm(emb)       # (B, T, 2*lstm_hidden)

        # Attention weights
        alpha = torch.sigmoid(self.temporal_attn_proj(temporal_out))  # (B, T, 1)
        beta = torch.sigmoid(self.feature_attn_proj(feat_out))        # (B, T, d_model)

        # If mask is true padding, you can suppress padded positions here.
        # For forward-filled irregular ICU windows, usually keep this False.
        if mask is not None and self.use_mask_as_padding:
            m = mask.unsqueeze(-1).float()  # (B, T, 1)
            alpha = alpha * m
            beta = beta * m

        weighted_emb = emb * beta  # (B, T, d_model)

        # Transformer
        key_padding_mask = None
        if mask is not None and self.use_mask_as_padding:
            key_padding_mask = ~mask.bool()  # True = padded positions

        transformer_attentions = [] if return_attentions else None
        z = weighted_emb
        for layer in self.transformer_layers:
            z, attn = layer(
                z,
                src_key_padding_mask=key_padding_mask,
                return_attention=return_attentions
            )
            if return_attentions and attn is not None:
                transformer_attentions.append(attn)

        # Temporal attention weighting
        z = z * alpha.expand(-1, -1, self.d_model)
            
        # Global pooling
        if mask is not None:
            w = mask.unsqueeze(-1).float()
            pooled = (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        else:
            pooled = z.mean(dim=1)

        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        if return_attentions:
            info = {
                "alpha": alpha,                         # (B, T, 1)
                "beta": beta,                           # (B, T, d_model)
                "transformer_attentions": transformer_attentions,
                "embedding_weight": self.input_proj.weight,  # (d_model, n_features)
                "input": x,
                "mask": mask,
            }
            return logits, info

        return logits, None

    def get_local_contributions(self, info: Dict) -> torch.Tensor:
        alpha = info["alpha"]                  # (B, T, 1)
        beta = info["beta"]                    # (B, T, d_model)
        W_emb = info["embedding_weight"]       # (d_model, n_features)
        X = info["input"]                      # (B, T, n_features)
        mask = info["mask"]

        beta_W = torch.matmul(beta, W_emb)     # (B, T, n_features)
        contrib = alpha * (beta_W * X)         # (B, T, n_features)

        if mask is not None:
            contrib = contrib * mask.unsqueeze(-1).float()

        return contrib

    def get_cross_attention_matrix(self, info: Dict) -> torch.Tensor:
        attn_list = info["transformer_attentions"]
        if not attn_list:
            raise ValueError("No attention stored. Call forward(return_attentions=True).")

        # Each layer: (B, heads, T, T)
        stacked = torch.stack(attn_list, dim=0)  # (L, B, heads, T, T)
        return stacked.mean(dim=(0, 2))          # (B, T, T)

    def get_chain_of_influence(self, info: Dict) -> torch.Tensor:
        C = self.get_local_contributions(info)    # (B, T, F)
        A = self.get_cross_attention_matrix(info) # (B, T, T)

        C1 = C.unsqueeze(-1).unsqueeze(-1)        # (B, T, F, 1, 1)
        A2 = A.unsqueeze(2).unsqueeze(-1)         # (B, T, 1, T, 1)
        C2 = C.unsqueeze(1).unsqueeze(2)          # (B, 1, 1, T, F)

        I = C1 * A2 * C2                          # (B, T, F, T, F)
        return I





def build_training_components(
    X_train, y_train, device,
    d_model=64,
    lstm_hidden=32,
    n_heads=4,
    n_layers=2,
    dim_feedforward=256,
    dropout=0.2,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=128,
):
    n_features = X_train.shape[-1]

    model = ChainOfInfluence(
        n_features=n_features,
        d_model=d_model,
        lstm_hidden=lstm_hidden,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=X_train.shape[1],
        use_mask_as_padding=False,   # keep False for forward-filled 48h windows
    ).to(device)

    num_pos = float((y_train == 1).sum())
    num_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, criterion, optimizer