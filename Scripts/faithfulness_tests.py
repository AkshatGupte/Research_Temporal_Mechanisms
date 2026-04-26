"""
faithfulness_testing.py
-----------------------
Faithfulness experiments for ICU mortality using ChainOfInfluence model.
Tests whether temporal attention is truly important for predictions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wilcoxon


def get_attention_and_prediction(model, X, mask, device, batch_size=64):
    """
    Get predictions and normalised temporal attention for a dataset.
    
    Returns:
        probs: (N,) predicted mortality probabilities
        attn: (N, T) normalised attention weights (sum=1 over time)
    """
    model.eval()
    
    # Convert to tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool)
    
    dataset = TensorDataset(X, mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_attns = []
    
    with torch.no_grad():
        for xb, mb in dataloader:
            xb = xb.to(device)
            mb = mb.to(device)
            
            # Forward pass with attention
            logits, info = model(xb, mask=mb, return_attentions=True)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Extract temporal attention (alpha)
            # FIXED: alpha is ALREADY softmaxed from the model
            # Don't apply softmax again!
            alpha = info['alpha'].squeeze(-1)  # (B, T) - already normalized by model
            alpha_np = alpha.cpu().numpy()
            
            all_probs.extend(probs)
            all_attns.extend(alpha_np)
    
    return np.array(all_probs), np.array(all_attns)


def erasure_experiment(model, X, mask, device, top_k=3, batch_size=64):
    """
    Erase top-k attended time steps and measure prediction change.
    
    For faithful attention: erasing top-k steps should cause a LARGER drop
    in mortality probability than erasing random steps.
    """
    print(f"\n{'='*50}")
    print(f"ERASURE EXPERIMENT (top_k={top_k})")
    print(f"{'='*50}")
    
    # Get baseline predictions and attention
    baseline_probs, attn = get_attention_and_prediction(model, X, mask, device, batch_size)
    N, T = attn.shape
    
    print(f"Patients: {N}, Time steps: {T}")
    print(f"Baseline mortality probability: {baseline_probs.mean():.4f}")
    
    # Find top-k attended time steps for each patient
    top_k_indices = np.argsort(attn, axis=1)[:, -top_k:]  # (N, top_k)
    
    # FIXED: Proper erasure using masking, not zero-setting
    # Create modified mask that excludes top-k timesteps
    mask_erased_attn = mask.copy() if isinstance(mask, np.ndarray) else mask.numpy().copy()
    X_erased_attn = X.copy() if isinstance(X, np.ndarray) else X.numpy().copy()
    
    for i in range(N):
        # Set both mask and features to zero for top-k attended steps
        # This prevents the LSTM from processing these timesteps
        mask_erased_attn[i, top_k_indices[i]] = 0
        X_erased_attn[i, top_k_indices[i], :] = 0.0
    
    # Get predictions after attention-guided erasure
    erased_attn_probs, _ = get_attention_and_prediction(
        model, X_erased_attn, mask_erased_attn, device, batch_size
    )
    
    # Random erasure control - ALSO erase from mask
    rng = np.random.default_rng(42)
    mask_erased_rand = mask.copy() if isinstance(mask, np.ndarray) else mask.numpy().copy()
    X_erased_rand = X.copy() if isinstance(X, np.ndarray) else X.numpy().copy()
    
    for i in range(N):
        rand_indices = rng.choice(T, size=top_k, replace=False)
        mask_erased_rand[i, rand_indices] = 0
        X_erased_rand[i, rand_indices, :] = 0.0
    
    erased_rand_probs, _ = get_attention_and_prediction(
        model, X_erased_rand, mask_erased_rand, device, batch_size
    )
    
    # Calculate changes
    delta_attn = baseline_probs - erased_attn_probs
    delta_rand = baseline_probs - erased_rand_probs
    
    # Statistical test (one-sided: attention-guided drop > random drop)
    stat, p_value = wilcoxon(delta_attn, delta_rand, alternative='greater')
    
    results = {
        'baseline_probs': baseline_probs,
        'erased_attn_probs': erased_attn_probs,
        'erased_rand_probs': erased_rand_probs,
        'delta_attn': delta_attn,
        'delta_rand': delta_rand,
        'mean_delta_attn': float(delta_attn.mean()),
        'mean_delta_rand': float(delta_rand.mean()),
        'ratio': float(delta_attn.mean() / (delta_rand.mean() + 1e-9)),
        'wilcoxon_stat': float(stat),
        'wilcoxon_p': float(p_value),
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  Mean Δ (attention-guided erase): {results['mean_delta_attn']:.4f}")
    print(f"  Mean Δ (random erase):           {results['mean_delta_rand']:.4f}")
    print(f"  Ratio (attn/random):             {results['ratio']:.3f}")
    print(f"  Wilcoxon p (greater):            {results['wilcoxon_p']:.4f}")
    
    if results['ratio'] > 1 and results['wilcoxon_p'] < 0.05:
        print("  ✅ Attention is FAITHFUL (erasing attended steps hurts more)")
    else:
        print("  ❌ Attention is NOT faithful (or model doesn't rely on attended steps)")
    
    return results


def peak_erasure_experiment(model, X, mask, device, batch_size=64):
    """
    Simplified: Erase ONLY the single peak attention timestep.
    
    This is a more direct test: if the model truly focuses on one time step,
    erasing it should significantly change the prediction.
    """
    print(f"\n{'='*50}")
    print(f"PEAK ERASURE EXPERIMENT (single timestep)")
    print(f"{'='*50}")
    
    # Get baseline predictions and attention
    baseline_probs, attn = get_attention_and_prediction(model, X, mask, device, batch_size)
    N, T = attn.shape
    
    # Find peak timestep for each patient
    peak_indices = np.argmax(attn, axis=1)  # (N,)
    
    # Create copy and erase only the peak timestep
    X_erased = X.copy() if isinstance(X, np.ndarray) else X.numpy().copy()
    
    for i in range(N):
        X_erased[i, peak_indices[i], :] = 0.0
    
    # Get predictions after erasure
    erased_probs, _ = get_attention_and_prediction(model, X_erased, mask, device, batch_size)
    
    # Calculate change
    delta = baseline_probs - erased_probs
    
    results = {
        'baseline_probs': baseline_probs,
        'erased_probs': erased_probs,
        'delta': delta,
        'mean_delta': float(delta.mean()),
        'std_delta': float(delta.std()),
        'frac_positive': float((delta > 0).mean()),
        'frac_negative': float((delta < 0).mean()),
    }
    
    print(f"\nResults:")
    print(f"  Mean Δ after erasing peak timestep: {results['mean_delta']:.4f}")
    print(f"  Std Δ: {results['std_delta']:.4f}")
    print(f"  Fraction with decreased risk: {results['frac_positive']*100:.1f}%")
    print(f"  Fraction with increased risk: {results['frac_negative']*100:.1f}%")
    
    if results['mean_delta'] > 0:
        print("  ✅ Erasing peak timestep decreases mortality prediction (expected)")
    else:
        print("  ⚠️ Erasing peak timestep INCREASES mortality prediction (counterintuitive)")
    
    return results


def print_attention_summary(model, test_loader, device, num_patients=100):
    """
    Print summary statistics of attention distribution.
    """
    model.eval()
    
    all_attn_entropy = []
    all_peak_attn = []
    
    with torch.no_grad():
        for X_b, mask_b, y_b in test_loader:
            X_b = X_b[:num_patients].to(device)
            mask_b = mask_b[:num_patients].to(device)
            
            logits, info = model(X_b, mask=mask_b, return_attentions=True)
            alpha = info['alpha'].squeeze(-1)  # (B, T)
            # FIXED: Don't re-normalize - it's already softmaxed
            alpha_norm = alpha.cpu().numpy()  # Already normalized [0,1]
            
            # Entropy (lower = more focused)
            entropy = -np.sum(alpha_norm * np.log(alpha_norm + 1e-10), axis=1)
            all_attn_entropy.extend(entropy)
            
            # Peak attention value
            all_peak_attn.extend(alpha_norm.max(axis=1))
            
            if len(all_attn_entropy) >= num_patients:
                break
    
    print(f"\n{'='*50}")
    print(f"ATTENTION SUMMARY (n={len(all_attn_entropy)})")
    print(f"{'='*50}")
    print(f"  Mean peak attention: {np.mean(all_peak_attn):.4f}")
    print(f"  Std peak attention:  {np.std(all_peak_attn):.4f}")
    print(f"  Mean entropy:        {np.mean(all_attn_entropy):.4f}")
    print(f"  Std entropy:         {np.std(all_attn_entropy):.4f}")
    print(f"  Min entropy:         {np.min(all_attn_entropy):.4f}")
    print(f"  Max entropy:         {np.max(all_attn_entropy):.4f}")
    
    # Uniform baseline entropy = ln(48) ≈ 3.87
    uniform_entropy = np.log(48)
    print(f"\n  Uniform baseline entropy: {uniform_entropy:.4f}")
    
    if np.mean(all_attn_entropy) < uniform_entropy - 0.5:
        print("  ✅ Attention is focused (lower entropy than uniform)")
    else:
        print("  ⚠️ Attention is near-uniform (not focused on specific timesteps)")
    
    return all_attn_entropy, all_peak_attn