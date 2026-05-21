"""Static visualization functions for radar data.

Provides publication-quality plots using Matplotlib:
    - Range-Doppler maps (dB scale with colorbars)
    - CFAR detection overlays
    - Interference type comparison grids
    - SINR vs INR curves
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Consistent style
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "image.cmap": "inferno",
})


def plot_rd_map(rd_map_db, range_bins, velocity_bins, title="Range-Doppler Map",
                ax=None, figsize=(8, 6), vmin=None, vmax=None):
    """Plot a single Range-Doppler map in dB scale.

    Args:
        rd_map_db: RD map in dB, shape (N_doppler, N_range).
        range_bins: Range axis values (m), shape (N_range,).
        velocity_bins: Velocity axis values (m/s), shape (N_doppler,).
        title: Plot title.
        ax: Optional matplotlib Axes.
        figsize: Figure size if creating new figure.
        vmin, vmax: Colorbar range in dB. Auto if None.

    Returns:
        fig, ax: Matplotlib figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    extent = [range_bins[0], range_bins[-1], velocity_bins[0], velocity_bins[-1]]

    if vmin is None:
        vmin = np.percentile(rd_map_db, 5)
    if vmax is None:
        vmax = np.max(rd_map_db)

    im = ax.imshow(rd_map_db, aspect="auto", origin="lower", extent=extent,
                   vmin=vmin, vmax=vmax, cmap="inferno")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Power (dB)")
    return fig, ax


def plot_cfar_detections(rd_map_db, detections, range_bins, velocity_bins,
                          title="CFAR Detections", ax=None, figsize=(8, 6)):
    """Plot RD map with CFAR detections overlaid as red markers.

    Args:
        rd_map_db: RD map in dB.
        detections: Boolean detection mask (same shape as rd_map_db).
        range_bins, velocity_bins: Axes values.
        title: Plot title.
        ax: Optional Axes.

    Returns:
        fig, ax: Figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    extent = [range_bins[0], range_bins[-1], velocity_bins[0], velocity_bins[-1]]
    vmin = np.percentile(rd_map_db, 5)
    vmax = np.max(rd_map_db)

    ax.imshow(rd_map_db, aspect="auto", origin="lower", extent=extent,
              vmin=vmin, vmax=vmax, cmap="inferno")

    # Overlay detections
    det_ranges, det_vels = np.where(detections)
    if len(det_ranges) > 0:
        # Swap indices: detections[d_idx, r_idx] -> range_bins[r_idx], velocity_bins[d_idx]
        ax.plot(range_bins[det_vels], velocity_bins[det_ranges], "rx",
                markersize=3, markeredgewidth=0.5, label=f"{len(det_ranges)} detections")
        ax.legend(fontsize=8)

    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(title)
    return fig, ax


def plot_interference_comparison(clean_rd, interfered_rd, denoised_rd=None,
                                  range_bins=None, velocity_bins=None,
                                  titles=None, figsize=(16, 5)):
    """Side-by-side comparison of clean, interfered, and (optionally) denoised RD maps.

    Args:
        clean_rd: Clean RD map in dB.
        interfered_rd: Interfered RD map in dB.
        denoised_rd: Optional denoised RD map in dB.
        range_bins, velocity_bins: Axis values. Auto if None.
        titles: List of 2-3 subplot titles.
        figsize: Figure size.

    Returns:
        fig: Matplotlib figure.
    """
    n_plots = 3 if denoised_rd is not None else 2
    if titles is None:
        titles = ["Clean", "Interfered", "Denoised"][:n_plots]

    if range_bins is None:
        range_bins = np.arange(clean_rd.shape[1])
    if velocity_bins is None:
        velocity_bins = np.arange(clean_rd.shape[0])

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Shared color scale
    vmin = min(np.percentile(m, 5) for m in [clean_rd, interfered_rd]
               if m is not None)
    vmax = max(np.max(m) for m in [clean_rd, interfered_rd]
               if m is not None)

    extent = [range_bins[0], range_bins[-1], velocity_bins[0], velocity_bins[-1]]
    maps = [clean_rd, interfered_rd, denoised_rd]

    for i, (ax, rd, title) in enumerate(zip(axes, maps, titles)):
        if rd is None:
            continue
        im = ax.imshow(rd, aspect="auto", origin="lower", extent=extent,
                       vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(title)
        ax.set_xlabel("Range (m)")
        if i == 0:
            ax.set_ylabel("Velocity (m/s)")

    plt.colorbar(im, ax=axes[-1] if n_plots > 1 else axes, label="Power (dB)",
                 fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_sinr_curve(inr_values, sinr_improvements, labels=None,
                     title="SINR Improvement vs Input INR", ax=None, figsize=(8, 5)):
    """Plot SINR improvement curves for different interference types.

    Args:
        inr_values: Array of INR values (dB).
        sinr_improvements: Shape (n_types, n_inr) — SINR improvement per type per INR.
        labels: List of interference type names.
        title: Plot title.
        ax: Optional Axes.

    Returns:
        fig, ax: Figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = plt.cm.tab10(np.linspace(0, 1, len(sinr_improvements)))

    for i, (sinr_imp, color) in enumerate(zip(sinr_improvements, colors)):
        label = labels[i] if labels else f"Type {i}"
        ax.plot(inr_values, sinr_imp, "o-", color=color, label=label,
                markersize=5, linewidth=1.5)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Input INR (dB)")
    ax.set_ylabel("SINR Improvement (dB)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix",
                           ax=None, figsize=(8, 7)):
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (n_classes x n_classes).
        class_names: List of class name strings.
        title: Plot title.
        ax: Optional Axes.

    Returns:
        fig, ax: Figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:.1%}" if cm[i, j] < 1 else f"{cm[i, j]:.0f}",
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return fig, ax


def figure_to_bytes(fig):
    """Convert a Matplotlib figure to PNG bytes for Gradio display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf
