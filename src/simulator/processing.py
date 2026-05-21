"""Range-Doppler processing: 2D FFT and CFAR detection.

Processing chain:
    1. Apply window functions (Hann) to suppress sidelobes
    2. Range FFT: FFT along fast-time axis (per chirp)
    3. Doppler FFT: FFT along slow-time axis (per range bin)
    4. Magnitude -> dB scale
    5. CA-CFAR / OS-CFAR detection
"""
import numpy as np
from scipy import constants
from scipy.ndimage import binary_dilation


# ---------------------------------------------------------------------------
# 2D FFT Range-Doppler processing
# ---------------------------------------------------------------------------

def range_doppler_map(adc_data, window_range="hann", window_doppler="hann"):
    """Compute the 2D Range-Doppler map from ADC data matrix.

    Args:
        adc_data: Complex ADC samples, shape (N_chirps, N_samp).
            - axis 0 (rows): slow-time (chirp index)
            - axis 1 (cols): fast-time (sample index per chirp)
        window_range: Window function for range FFT ('hann', 'hamming', 'blackman', None).
        window_doppler: Window function for Doppler FFT.

    Returns:
        rd_map_db: Range-Doppler map in dB, shape (N_chirps, N_samp//2).
            Doppler axis is fftshifted (0 Doppler at center).
        range_bins: Range axis values in meters, shape (N_samp//2,).
        velocity_bins: Velocity axis values in m/s, shape (N_chirps,).
    """
    N_chirps, N_samp = adc_data.shape

    # Range window
    if window_range and window_range != "none":
        w_range = np.hanning(N_samp) if window_range == "hann" else \
                  np.hamming(N_samp) if window_range == "hamming" else \
                  np.blackman(N_samp) if window_range == "blackman" else \
                  np.ones(N_samp)
    else:
        w_range = np.ones(N_samp)

    # Doppler window
    if window_doppler and window_doppler != "none":
        w_doppler = np.hanning(N_chirps) if window_doppler == "hann" else \
                    np.hamming(N_chirps) if window_doppler == "hamming" else \
                    np.blackman(N_chirps) if window_doppler == "blackman" else \
                    np.ones(N_chirps)
    else:
        w_doppler = np.ones(N_chirps)

    # Range FFT: FFT along fast-time (axis=1), real input -> rfft for efficiency
    range_fft = np.fft.fft(adc_data * w_range[np.newaxis, :], axis=1)

    # Take only positive range bins (real signal, symmetric FFT)
    N_range = N_samp // 2
    range_fft = range_fft[:, :N_range]

    # Doppler FFT: FFT along slow-time (axis=0)
    doppler_fft = np.fft.fftshift(
        np.fft.fft(range_fft * w_doppler[:, np.newaxis], axis=0),
        axes=0
    )

    # Magnitude in dB
    rd_map = np.abs(doppler_fft)
    rd_map_db = 20 * np.log10(rd_map + 1e-12)

    return rd_map_db, rd_map


def compute_axes(fc, B, T_chirp, N_chirps, N_samp):
    """Compute range and velocity axis values.

    Args:
        fc: Carrier frequency (Hz).
        B: Chirp bandwidth (Hz).
        T_chirp: Chirp duration (s).
        N_chirps: Number of chirps per frame.
        N_samp: Number of ADC samples per chirp.

    Returns:
        range_bins: Range axis (m), shape (N_samp//2,).
        velocity_bins: Velocity axis (m/s), shape (N_chirps,). fftshifted.
    """
    c = constants.speed_of_light
    wavelength = c / fc
    N_range = N_samp // 2

    # Range axis
    range_bins = np.arange(N_range) * c / (2 * B)

    # Velocity axis (fftshifted: -v_max/2 to +v_max/2)
    v_max = wavelength / (2 * T_chirp)
    velocity_bins = np.linspace(-v_max/2, v_max/2, N_chirps, endpoint=False)

    return range_bins, velocity_bins


# ---------------------------------------------------------------------------
# CFAR Detection
# ---------------------------------------------------------------------------

def ca_cfar(rd_map_linear, pfa=1e-4, guard_cells=4, training_cells=8):
    """Cell-Averaging CFAR detection on range-Doppler map.

    For each cell under test (CUT), estimates noise power from surrounding
    training cells, excluding guard cells adjacent to the CUT.

    Args:
        rd_map_linear: Range-Doppler map in linear (not dB) scale,
            shape (N_doppler, N_range).
        pfa: Desired probability of false alarm.
        guard_cells: Number of guard cells on each side of CUT.
        training_cells: Number of training cells on each side.

    Returns:
        detections: Boolean mask, True where target detected.
        threshold: Threshold map (same shape as rd_map_linear).
    """
    N_doppler, N_range = rd_map_linear.shape
    threshold = np.zeros_like(rd_map_linear)
    detections = np.zeros_like(rd_map_linear, dtype=bool)

    # CFAR constant: T = alpha * P_noise
    # alpha = N_train * (Pfa^(-1/N_train) - 1)
    N_train = 2 * (2 * training_cells + 1)  # both sides, both dimensions
    alpha = N_train * (pfa ** (-1.0 / N_train) - 1)

    half_g = guard_cells
    half_t = training_cells
    half_w = half_g + half_t

    for d in range(half_w, N_doppler - half_w):
        for r in range(half_w, N_range - half_w):
            # Extract training region
            training_region = rd_map_linear[
                d - half_w : d + half_w + 1,
                r - half_w : r + half_w + 1
            ]

            # Zero out guard cells and CUT
            training_region[half_w - half_g : half_w + half_g + 1,
                           half_w - half_g : half_w + half_g + 1] = 0

            # Mean noise power (excluding zeros from guard/CUT)
            noise_power = np.sum(training_region) / np.count_nonzero(training_region)
            threshold[d, r] = alpha * noise_power

            if rd_map_linear[d, r] > threshold[d, r]:
                detections[d, r] = True

    return detections, threshold


def os_cfar(rd_map_linear, pfa=1e-4, guard_cells=4, training_cells=8, os_rank=0.75):
    """Ordered-Statistics CFAR.

    Instead of averaging, uses the k-th ordered value from the training
    cells as the noise estimate. More robust in multi-target scenarios.

    Args:
        rd_map_linear: Range-Doppler map in linear scale.
        pfa: Desired probability of false alarm.
        guard_cells, training_cells: As in ca_cfar.
        os_rank: Fractional rank (0-1) for ordered statistic
            (0.75 = use the 75th percentile value).

    Returns:
        detections, threshold: As in ca_cfar.
    """
    N_doppler, N_range = rd_map_linear.shape
    threshold = np.zeros_like(rd_map_linear)
    detections = np.zeros_like(rd_map_linear, dtype=bool)

    N_train = 2 * (2 * training_cells + 1)
    alpha = N_train * (pfa ** (-1.0 / N_train) - 1)

    half_g = guard_cells
    half_t = training_cells
    half_w = half_g + half_t

    for d in range(half_w, N_doppler - half_w):
        for r in range(half_w, N_range - half_w):
            training_region = rd_map_linear[
                d - half_w : d + half_w + 1,
                r - half_w : r + half_w + 1
            ]

            # Zero out guard cells and CUT
            training_region[half_w - half_g : half_w + half_g + 1,
                           half_w - half_g : half_w + half_g + 1] = 0

            training_flat = training_region[training_region > 0]
            if len(training_flat) == 0:
                continue

            training_sorted = np.sort(training_flat)
            k = int(len(training_sorted) * os_rank)
            noise_estimate = training_sorted[max(0, min(k, len(training_sorted) - 1))]

            threshold[d, r] = alpha * noise_estimate
            if rd_map_linear[d, r] > threshold[d, r]:
                detections[d, r] = True

    return detections, threshold
