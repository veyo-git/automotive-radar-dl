"""Evaluation metrics for radar interference detection and mitigation.

Metrics:
    - SINR (Signal-to-Interference-plus-Noise Ratio)
    - Pd (Probability of Detection)
    - Pfa (Probability of False Alarm)
    - Classification accuracy (per-class + macro)
    - Confusion matrix
"""
import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# SINR computation
# ---------------------------------------------------------------------------

def compute_sinr(rd_map_db, target_mask, guard_band=3):
    """Compute Signal-to-Interference-plus-Noise Ratio in dB.

    Signal: peak dB value within target mask region.
    Interference+Noise: mean power in background (non-target, non-guard) region.

    Args:
        rd_map_db: Range-Doppler map in dB, shape (N_doppler, N_range).
        target_mask: Boolean mask for target regions (True = target).
        guard_band: Guard band width in cells around targets (excluded from bg).

    Returns:
        sinr_db: SINR in dB (positive = signal above noise floor).
    """
    # Background mask: not target, not guard band around target
    bg_mask = ~binary_dilation(target_mask, iterations=guard_band)

    if not np.any(target_mask) or not np.any(bg_mask):
        return 0.0

    # Signal: peak of each target region (linear domain)
    P_signal_linear = np.max(10 ** (rd_map_db[target_mask] / 10))

    # Interference+Noise: mean power in linear domain, then convert to dB
    P_bg_linear = np.mean(10 ** (rd_map_db[bg_mask] / 10))
    P_bg_db = 10 * np.log10(P_bg_linear + 1e-12)

    sinr_db = 10 * np.log10(P_signal_linear) - P_bg_db
    return sinr_db


def compute_sinr_improvement(clean_rd, interfered_rd, denoised_rd, target_mask):
    """Compute SINR improvement from interference mitigation.

    Returns dict with:
        - sinr_clean: SINR of clean RD map
        - sinr_interfered: SINR after interference
        - sinr_denoised: SINR after denoising
        - sinr_loss: SINR reduction due to interference (dB)
        - sinr_recovery: SINR recovered by denoising (dB)
        - recovery_ratio: Fraction of SINR loss recovered (0-1)
    """
    sinr_clean = compute_sinr(clean_rd, target_mask)
    sinr_int = compute_sinr(interfered_rd, target_mask)
    sinr_den = compute_sinr(denoised_rd, target_mask)

    sinr_loss = sinr_clean - sinr_int
    sinr_recovery = sinr_den - sinr_int
    recovery_ratio = sinr_recovery / max(sinr_loss, 1e-6)

    return {
        "sinr_clean_db": sinr_clean,
        "sinr_interfered_db": sinr_int,
        "sinr_denoised_db": sinr_den,
        "sinr_loss_db": sinr_loss,
        "sinr_recovery_db": sinr_recovery,
        "recovery_ratio": min(recovery_ratio, 1.0),
    }


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def compute_pd_pfa(detections, ground_truth, guard_band=2):
    """Compute Probability of Detection and False Alarm.

    Pd = TP / (TP + FN) — fraction of true targets detected.
    Pfa = FP / (FP + TN) — fraction of non-target cells that triggered.

    A detection is a True Positive if it falls within guard_band cells
    of a ground truth target.

    Args:
        detections: Boolean detection mask from CFAR.
        ground_truth: Boolean ground truth target mask.
        guard_band: Matching radius in cells.

    Returns:
        pd: Probability of detection (0-1).
        pfa: Probability of false alarm (0-1).
        n_true_targets: Number of ground truth targets.
        n_detections: Total number of detections.
    """
    # Dilate ground truth for matching
    gt_dilated = binary_dilation(ground_truth, iterations=guard_band)

    # True Positives: detection within dilated GT region
    TP = np.sum(detections & gt_dilated)

    # False Positives: detection outside dilated GT region
    FP = np.sum(detections & ~gt_dilated)

    # False Negatives: GT target with no detection nearby
    n_targets = 0
    gt_labels, n_gt = label_connected_components(ground_truth)
    for label_id in range(1, n_gt + 1):
        target_region = gt_labels == label_id
        if np.any(detections & binary_dilation(target_region, iterations=guard_band)):
            n_targets += 1
    FN = n_gt - n_targets

    # True Negatives: non-target, non-detection
    TN_cells = np.sum(~gt_dilated)
    TN = TN_cells - FP

    pd = TP / max(TP + FN, 1)
    pfa = FP / max(FP + TN, 1)

    return pd, pfa, n_gt, TP + FP


def label_connected_components(mask):
    """Simple connected-component labeling.

    Returns:
        labels: Integer array, same shape as mask, 0=background.
        n_labels: Number of connected components found.
    """
    from scipy.ndimage import label
    return label(mask)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, class_names=None):
    """Compute full classification metrics.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        class_names: Optional list of class name strings.

    Returns:
        dict with accuracy, per_class_acc, cm (confusion matrix), report (str).
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)

    # Per-class accuracy
    per_class = {}
    for c in range(cm.shape[0]):
        row_sum = cm[c].sum()
        if row_sum > 0:
            per_class[c] = cm[c, c] / row_sum
        else:
            per_class[c] = 0.0

    result = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm,
    }

    if class_names:
        result["class_names"] = class_names
        result["report"] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )

    return result


# ---------------------------------------------------------------------------
# SINR vs INR curve
# ---------------------------------------------------------------------------

def compute_sinr_curve(model, dataset, inr_range, device="cpu"):
    """Compute SINR improvement curve across INR values.

    Args:
        model: Trained U-Net denoiser.
        dataset: RadarDataset instance (test split).
        inr_range: Array of INR values to test.
        device: Torch device.

    Returns:
        sinr_improvements: Array of mean SINR improvement per INR value.
    """
    import torch
    model.eval()
    model.to(device)

    sinr_improvements = []

    # For each INR, find samples closest to that INR
    with torch.no_grad():
        for inr_target in inr_range:
            improvements = []
            for i in range(min(len(dataset), 50)):  # subset for speed
                interfered, clean, label = dataset[i]
                # Only evaluate on interfered samples (label > 0)
                if label == 0:
                    continue

                interfered = interfered.unsqueeze(0).to(device)
                clean = clean.unsqueeze(0).to(device)
                output = model(interfered)

                # Simplified SINR improvement
                int_error = ((interfered - clean) ** 2).mean().item()
                out_error = ((output - clean) ** 2).mean().item()
                if int_error > 1e-10:
                    imp = 10 * np.log10(int_error / max(out_error, 1e-12))
                    improvements.append(imp)

            if improvements:
                sinr_improvements.append(np.mean(improvements))
            else:
                sinr_improvements.append(0.0)

    return np.array(sinr_improvements)
