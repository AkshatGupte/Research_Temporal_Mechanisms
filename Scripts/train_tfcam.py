"""
train_tfcam.py  —  training + evaluation only
assumes train_loader, val_loader, test_loader already exist in notebook
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def train_tfcam(model, train_loader, val_loader, device, epochs=50, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_auroc, best_state, patience_cnt = 0.0, None, 0
    history = {"train_loss": [], "val_auroc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for X_b, _, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_b)
            loss = criterion(logits, y_b.unsqueeze(-1).float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_auroc = evaluate_tfcam(model, val_loader, device, verbose=False)["AUROC"]
        avg_loss  = np.mean(losses)
        history["train_loss"].append(avg_loss)
        history["val_auroc"].append(val_auroc)
        scheduler.step(val_auroc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_auroc={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            print(f"Early stop at epoch {epoch} | best AUROC={best_auroc:.4f}")
            break

    model.load_state_dict(best_state)
    return model, history


def evaluate_tfcam(model, loader, device, threshold=0.3, verbose=True):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, _, y_b in loader:
            logits, _ = model(X_b.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().squeeze(-1))
            all_labels.append(y_b.cpu())

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = (probs >= threshold).astype(int)

    metrics = {
        "AUROC":     roc_auc_score(labels, probs),
        "F1":        f1_score(labels, preds, zero_division=0),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall":    recall_score(labels, preds, zero_division=0),
        "Accuracy":  accuracy_score(labels, preds),
    }

    if verbose:
        for k, v in metrics.items():
            print(f"{k:<12}: {v:.4f}")

    return metrics
