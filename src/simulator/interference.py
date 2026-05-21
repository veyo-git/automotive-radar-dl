"""Radar interference generators for automotive FMCW radar scenarios.

Models 5 interference types commonly encountered in the 76-81 GHz band:
    1. CW (Continuous Wave): Single-tone jammer in the IF band.
    2. Cross-FMCW: Mutual interference from another vehicle's FMCW radar.
    3. Noise Jamming: Broadband Gaussian noise injection.
    4. Spoofing: Fake target ghost generation (deceptive jamming).
    5. Chirp-Swept: Frequency-swept interference across the chirp band.

Each generator returns the interference signal sampled at the radar's
ADC rate, which can then be injected into the IF signal at a specified
Interference-to-Noise Ratio (INR).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Individual interference generators
# ---------------------------------------------------------------------------

def generate_cw_interference(t, fc_if, amplitude=1.0, phase=0.0):
    """Generate a continuous-wave (single-tone) interference in the IF band.

    Appears as a bright horizontal line or localized spot in the RD map.

    Args:
        t: Time vector, shape (N_samp,).
        fc_if: Interference frequency within the IF band (Hz).
        amplitude: Signal amplitude.
        phase: Initial phase (radians).

    Returns:
        Complex interference signal, shape (N_samp,).
    """
    return amplitude * np.exp(1j * (2 * np.pi * fc_if * t + phase))


def generate_cross_fmcw_interference(t, K_victim, K_interferer,
                                      delta_t=0, fc_offset=0, amplitude=1.0):
    """Generate cross-FMCW mutual interference between two radars.

    This is THE signature problem for automotive radar. When two vehicles'
    FMCW radars chirp simultaneously with overlapping time-frequency
    resources, the interfering chirp after mixing with the victim's LO:
        i(t) = A * exp(j*2*pi * [fc*delta_t + (K1-K2)*t^2/2 + K2*delta_t*t])

    The interference appears as a wideband chirp spanning many range bins.
    In the RD map, it shows as a diagonal/slanted bright band that masks
    real targets.

    Args:
        t: Time vector (seconds).
        K_victim: Victim radar chirp slope (Hz/s) = B/T.
        K_interferer: Interferer radar chirp slope (Hz/s).
            Different slopes produce different RD interference patterns.
            - Same slope (K1=K2): appears as a narrow spot (CW-like)
            - Slightly different slope: diagonal band
            - Very different slope: wideband noise-like spread
        delta_t: Time offset between interferer and victim chirp start (seconds).
        fc_offset: Carrier frequency offset between the two radars (Hz).
            In practice, both at ~77 GHz but may differ by a few MHz.
        amplitude: Interference amplitude.

    Returns:
        Complex interference signal, shape (N_samp,).
    """
    N_samp = len(t)
    delta_K = K_victim - K_interferer

    # Phase evolution of the cross-FMCW interference
    # phi(t) = 2*pi * [fc_offset*t + 0.5*delta_K*t^2 + K_interferer*delta_t*t]
    phase = 2 * np.pi * (
        fc_offset * t +
        0.5 * delta_K * t ** 2 +
        K_interferer * delta_t * t
    )

    return amplitude * np.exp(1j * phase)


def generate_noise_jamming(t, amplitude=1.0):
    """Generate broadband Gaussian noise jamming.

    Raises the noise floor uniformly across the RD map, reducing
    detection probability for all targets.

    Args:
        t: Time vector, shape (N_samp,).
        amplitude: RMS noise amplitude.

    Returns:
        Complex noise signal, shape (N_samp,).
    """
    N_samp = len(t)
    noise = (np.random.randn(N_samp) + 1j * np.random.randn(N_samp)) / np.sqrt(2)
    return amplitude * noise


def generate_spoofing(t, fc, B, T_chirp, fake_range, fake_velocity, amplitude=1.0):
    """Generate a spoofing (deceptive) interference signal.

    Creates a fake target ghost by retransmitting a delayed and
    Doppler-shifted replica of the victim's own chirp.

    Args:
        t: Time vector, shape (N_samp,).
        fc: Victim radar carrier frequency (Hz).
        B: Victim radar bandwidth (Hz).
        T_chirp: Victim chirp duration (s).
        fake_range: Apparent range of the ghost target (m).
        fake_velocity: Apparent velocity of the ghost target (m/s).
        amplitude: Ghost signal amplitude.

    Returns:
        Complex spoofing signal, shape (N_samp,).
    """
    from scipy import constants
    c = constants.speed_of_light
    K = B / T_chirp

    tau_fake = 2 * fake_range / c
    f_d_fake = 2 * fake_velocity * fc / c
    f_b_fake = K * tau_fake + f_d_fake

    return amplitude * np.exp(1j * 2 * np.pi * f_b_fake * t)


def generate_chirp_swept_interference(t, f_start, f_stop, amplitude=1.0):
    """Generate a frequency-swept interference (swept jammer).

    Sweeps linearly from f_start to f_stop over the chirp duration,
    creating a diagonal interference pattern in the RD map.

    Args:
        t: Time vector, shape (N_samp,).
        f_start: Start frequency within IF band (Hz).
        f_stop: Stop frequency within IF band (Hz).
        amplitude: Signal amplitude.

    Returns:
        Complex swept interference signal, shape (N_samp,).
    """
    T_total = t[-1] - t[0]
    # Instantaneous frequency: f(t) = f_start + (f_stop - f_start) * t / T
    # Phase = integral of 2*pi*f(t) dt
    sweep_rate = (f_stop - f_start) / T_total
    phase = 2 * np.pi * (f_start * t + 0.5 * sweep_rate * t ** 2)
    return amplitude * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# INR-controlled injection
# ---------------------------------------------------------------------------

def compute_signal_power(signal):
    """Compute average power of a complex signal."""
    return np.mean(np.abs(signal) ** 2)


def inject_interference(s_if, i_signal, snr_db=20, inr_db=10):
    """Inject interference into IF signal at a controlled INR.

    INR = P_interference / P_noise (after receiver noise floor).
    The interference is scaled so that INR_dB = 10*log10(P_int / P_noise).

    Args:
        s_if: Clean complex IF signal (signal + noise), shape (N_samp,).
        i_signal: Complex interference signal, shape (N_samp,).
        snr_db: Signal-to-Noise Ratio of the clean signal (dB).
        inr_db: Interference-to-Noise Ratio (dB). Higher = stronger interference.

    Returns:
        s_interfered: IF signal with interference injected.
        scaling_factor: The factor applied to i_signal to achieve target INR.
    """
    P_signal = compute_signal_power(s_if)
    P_noise = P_signal / (10 ** (snr_db / 10) + 1e-12)
    P_int_target = P_noise * (10 ** (inr_db / 10))

    i_normalized = i_signal / np.sqrt(compute_signal_power(i_signal) + 1e-12)
    scaling = np.sqrt(P_int_target + 1e-12)
    s_interfered = s_if + scaling * i_normalized

    return s_interfered, scaling


# ---------------------------------------------------------------------------
# Interference type label mapping
# ---------------------------------------------------------------------------

INTERFERENCE_LABELS = {
    0: "clean",
    1: "cw",
    2: "cross_fmcw",
    3: "noise_jamming",
    4: "spoofing",
    5: "chirp_swept",
}

LABEL_TO_INT = {v: k for k, v in INTERFERENCE_LABELS.items()}
N_CLASSES = len(INTERFERENCE_LABELS)


def get_label_name(label_idx):
    """Get human-readable interference type name."""
    return INTERFERENCE_LABELS.get(label_idx, "unknown")
