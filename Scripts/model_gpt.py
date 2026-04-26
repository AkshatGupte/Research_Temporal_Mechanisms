import math
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Normalization
# -----------------------------
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


# -----------------------------
# Positional encoding
# -----------------------------
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


# -----------------------------
# Transformer encoder layer with DyT
# -----------------------------
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

        self.norm1 = DyT(d_model)
        self.norm2 = DyT(d_model)

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


# -----------------------------
# Main model
# -----------------------------
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
        """
        B, T, _ = x.shape

        # Input projection + positional encoding
        emb = self.input_proj(x)          # (B, T, d_model)
        emb = self.pos_enc(emb)

        # LSTM branches
        temporal_out, _ = self.temporal_lstm(emb)  # (B, T, 2*lstm_hidden)
        feat_out, _ = self.feature_lstm(emb)       # (B, T, 2*lstm_hidden)

        # Attention weights
        alpha_logits = self.temporal_attn_proj(temporal_out)   # (B, T, 1)
        beta = torch.sigmoid(self.feature_attn_proj(feat_out)) # (B, T, d_model)

        # If mask represents real timesteps, suppress invalid positions
        if mask is not None and self.use_mask_as_padding:
            m = mask.unsqueeze(-1).float()  # (B, T, 1)
            alpha_logits = alpha_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
            beta = beta * m

        alpha = torch.softmax(alpha_logits, dim=1)  # (B, T, 1)

        # Feature gating
        weighted_emb = emb * beta

        # Transformer stack
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

        # Temporal gating
        z = z * alpha.expand(-1, -1, self.d_model)

        # Pooling
        if mask is not None:
            w = mask.unsqueeze(-1).float()
            pooled = (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        else:
            pooled = z.mean(dim=1)

        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        if return_attentions:
            info = {
                "alpha": alpha,                       # (B, T, 1)
                "beta": beta,                         # (B, T, d_model)
                "transformer_attentions": transformer_attentions,
                "embedding_weight": self.input_proj.weight,
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


# -----------------------------
# Dataset
# -----------------------------
class ICUDataset(Dataset):
    def __init__(self, X, mask, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)  # IMPORTANT for BCEWithLogitsLoss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


# -----------------------------
# Metrics / threshold search
# -----------------------------
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import f1_score

    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_y = []

    for xb, mb, yb in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        yb = yb.to(device)

        logits, _ = model(xb, mask=mb, return_attentions=False)
        loss = criterion(logits, yb)

        probs = torch.sigmoid(logits)

        total_loss += loss.item() * xb.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_y = np.concatenate(all_y)

    best_t, best_f1 = find_best_threshold(all_y.astype(int), all_probs)

    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

    y_pred = (all_probs >= best_t).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "f1": f1_score(all_y, y_pred, zero_division=0),
        "precision": precision_score(all_y, y_pred, zero_division=0),
        "recall": recall_score(all_y, y_pred, zero_division=0),
        "accuracy": accuracy_score(all_y, y_pred),
        "auc": roc_auc_score(all_y, all_probs) if len(np.unique(all_y)) > 1 else float("nan"),
        "best_threshold": best_t,
        "best_threshold_f1": best_f1,
    }
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0

    for xb, mb, yb in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(xb, mask=mb, return_attentions=False)
        loss = criterion(logits, yb)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


# -----------------------------
# Build model + training setup
# -----------------------------
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


# -----------------------------
# Example usage
# -----------------------------
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = ICUDataset(X_train, mask_train, y_train)
val_ds   = ICUDataset(X_val, mask_val, y_val)
test_ds  = ICUDataset(X_test, mask_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

model, criterion, optimizer = build_training_components(
    X_train=X_train,
    y_train=y_train,
    device=device,
    d_model=64,
    lstm_hidden=32,
    n_heads=4,
    n_layers=2,
    dim_feedforward=256,
    dropout=0.2,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=128,
)

best_val_f1 = -1.0
best_state = None

for epoch in range(1, 51):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_f1={val_metrics['f1']:.4f} | "
        f"val_auc={val_metrics['auc']:.4f} | "
        f"thr={val_metrics['best_threshold']:.3f}"
    )

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

test_metrics = evaluate(model, test_loader, criterion, device)
print("TEST:", test_metrics)
"""