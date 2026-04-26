import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for X_b, mask_b, y_b in loader:
        X_b = X_b.to(device)
        mask_b = mask_b.to(device)
        y_b = y_b.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(X_b, mask=mask_b, return_attentions=False)
        loss = criterion(logits.unsqueeze(1), y_b.unsqueeze(1).float())
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * X_b.size(0)

    return total_loss / len(loader.dataset)


def _validate(model, loader, criterion, device):
    model.eval()
    losses, probs, labels = [], [], []
    with torch.no_grad():
        for X_b, mask_b, y_b in loader:
            X_b = X_b.to(device)
            mask_b = mask_b.to(device)
            y_b = y_b.to(device)

            logits, _ = model(X_b, mask=mask_b, return_attentions=False)
            loss = criterion(logits.unsqueeze(1), y_b.unsqueeze(1).float())
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits).cpu().squeeze(-1))
            labels.append(y_b.cpu())

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    auroc = roc_auc_score(labels, probs) if labels.sum() > 0 else 0.0
    return np.mean(losses), auroc


def train_coi(model, train_loader, val_loader, device, epochs=50, lr=1e-3, patience=10, pos_weight=None):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss, best_state, patience_cnt = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc = _validate(model, val_loader, criterion, device)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auroc={val_auroc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            print(f"Early stop at epoch {epoch} | best val_loss={best_val_loss:.4f}")
            break

    model.load_state_dict(best_state)
    return model


def evaluate_coi(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, mask_b, y_b in loader:
            logits, _ = model(X_b.to(device), mask=mask_b.to(device), return_attentions=False)
            all_probs.append(torch.sigmoid(logits).cpu().squeeze(-1))
            all_labels.append(y_b)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "AUROC": roc_auc_score(labels, probs),
        "F1": f1_score(labels, preds, zero_division=0),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "Accuracy": accuracy_score(labels, preds),
    }
    for k, v in metrics.items():
        print(f"{k:<12}: {v:.4f}")
    return metrics