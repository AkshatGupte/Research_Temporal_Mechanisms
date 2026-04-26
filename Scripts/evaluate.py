from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
import torch
import numpy as np

def evaluate_test(model, test_loader, device, threshold=0.5):
    """
    Evaluate model on test set and return comprehensive metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: torch device
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for X_b, mask_b, y_b in test_loader:
            X_b, mask_b, y_b = X_b.to(device), mask_b.to(device), y_b.to(device)
            
            # Forward pass
            logits, _ = model(X_b, mask=mask_b, return_attentions=False)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= threshold).astype(int)
            
            all_probs.extend(probs)
            all_labels.extend(y_b.cpu().numpy())
            all_preds.extend(preds)
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    metrics = {
        'AUROC': roc_auc_score(all_labels, all_probs),
        'F1': f1_score(all_labels, all_preds, zero_division=0),
        'Precision': precision_score(all_labels, all_preds, zero_division=0),
        'Recall': recall_score(all_labels, all_preds, zero_division=0),
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Threshold': threshold
    }
    
    return metrics, all_probs, all_labels, all_preds


def find_optimal_threshold(model, val_loader, device):
    """
    Find optimal threshold using validation set (maximizes F1).
    
    Returns:
        float: Optimal threshold value
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_b, mask_b, y_b in val_loader:
            X_b, mask_b, y_b = X_b.to(device), mask_b.to(device), y_b.to(device)
            logits, _ = model(X_b, mask=mask_b, return_attentions=False)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_b.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Try different thresholds
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    return best_threshold


def print_detailed_metrics(metrics, all_labels, all_preds, all_probs):
    """Print detailed metrics including confusion matrix and classification report."""
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    print(f"\n📊 Core Metrics:")
    print(f"   AUROC:      {metrics['AUROC']:.4f}")
    print(f"   Accuracy:   {metrics['Accuracy']:.4f}")
    print(f"   F1 Score:   {metrics['F1']:.4f}")
    print(f"   Precision:  {metrics['Precision']:.4f}")
    print(f"   Recall:     {metrics['Recall']:.4f}")
    print(f"   Threshold:  {metrics['Threshold']:.3f}")
    
    print(f"\n📈 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"              Predicted")
    print(f"              Neg   Pos")
    print(f"   Actual Neg  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"          Pos  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n📉 Additional Metrics:")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   NPV:         {npv:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Survived', 'Mortality'], zero_division=0))
    
    # Class distribution in test set
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    print(f"\n📊 Test Set Distribution:")
    print(f"   Mortality:  {n_pos} ({n_pos/len(all_labels)*100:.1f}%)")
    print(f"   Survived:   {n_neg} ({n_neg/len(all_labels)*100:.1f}%)")


def get_predictions(model, loader, device):
    """Return all predictions, probabilities, and labels for further analysis."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_b, mask_b, y_b in loader:
            X_b, mask_b, y_b = X_b.to(device), mask_b.to(device), y_b.to(device)
            logits, _ = model(X_b, mask=mask_b, return_attentions=False)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_b.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)
