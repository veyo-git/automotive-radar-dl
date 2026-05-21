"""FMCW chirp waveform generation and IF signal processing.

FMCW (Frequency Modulated Continuous Wave) 线性调频连续波信号生成。

Mathematical model:
    TX chirp:  s_tx(t) = exp(j*2*pi * (fc*t + 0.5*K*t^2))
    where K = B / T_chirp is the chirp slope (Hz/s).

    Target delay:    tau = 2*R / c
    Target Doppler:  f_d = 2*v / lambda = 2*v*fc / c

    RX signal:  s_rx(t) = sum_i A_i * s_tx(t - tau_i) * exp(j*2*pi*f_d_i*t)

    IF (beat) signal after mixer + LPF:
        s_if(t) = sum_i A_if_i * exp(j*2*pi*(K*tau_i + f_d_i)*t + j*2*pi*fc*tau_i)

    Beat frequency:  f_b = K*tau + f_d = (2*B*R)/(c*T_chirp) + (2*v*fc)/c
"""
import numpy as np
from scipy import constants


def generate_chirp(fc, B, T_chirp, fs):
    """Generate a single FMCW chirp (complex baseband equivalent at fc=0).

    Args:
        fc: Carrier frequency in Hz (e.g., 77e9 for 77 GHz)
        B:  Chirp bandwidth in Hz (e.g., 750e6 for 750 MHz)
        T_chirp: Chirp duration in seconds (e.g., 40e-6 for 40 us)
        fs: Sampling frequency in Hz (e.g., 10e6 for 10 MSPS)

    Returns:
        chirp: Complex chirp samples, shape (N_samp,)
        t: Time vector, shape (N_samp,)
    """
    N_samp = int(T_chirp * fs)
    t = np.arange(N_samp) / fs
    K = B / T_chirp  # chirp slope (Hz/s)
    # Phase = pi * K * t^2 (baseband, fc=0 term handled separately in RX)
    phase = np.pi * K * t ** 2
    chirp = np.exp(1j * phase)
    return chirp, t


def compute_beat_frequency(target_range, target_velocity, fc, B, T_chirp):
    """Compute the IF beat frequency for a target.

    f_b = (2*B*R) / (c*T_chirp)  +  (2*v*fc) / c

    Args:
        target_range: Target range in meters.
        target_velocity: Target radial velocity in m/s (positive = approaching).
        fc: Carrier frequency in Hz.
        B: Chirp bandwidth in Hz.
        T_chirp: Chirp duration in seconds.

    Returns:
        Beat frequency in Hz.
    """
    c = constants.speed_of_light
    f_range = (2 * B * target_range) / (c * T_chirp)
    f_doppler = (2 * target_velocity * fc) / c
    return f_range + f_doppler


def generate_if_signal(t, targets, fc, B, T_chirp):
    """Generate IF beat signal from multi-target returns.

    For each target i with range R_i and velocity v_i:
        tau_i = 2*R_i / c
        f_d_i = 2*v_i / lambda
        s_if_i(t) = A_i * exp(j*2*pi * (K*tau_i + f_d_i)*t + j*phi_i)
    where phi_i includes the constant phase term.

    Args:
        t: Time vector from generate_chirp().
        targets: List of dicts with keys 'range', 'velocity', 'rcs'.
        fc: Carrier frequency in Hz.
        B: Chirp bandwidth in Hz.
        T_chirp: Chirp duration in seconds.

    Returns:
        s_if: Complex IF signal, shape (len(t),).
        target_params: List of dicts with per-target info (tau, f_b, amplitude).
    """
    c = constants.speed_of_light
    K = B / T_chirp
    wavelength = c / fc
    N = len(t)

    s_if = np.zeros(N, dtype=complex)
    target_params = []

    for i, tgt in enumerate(targets):
        R = tgt["range"]
        v = tgt["velocity"]
        rcs = tgt.get("rcs", 1.0)

        tau = 2 * R / c              # round-trip delay
        f_d = 2 * v / wavelength     # Doppler shift
        f_b = K * tau + f_d          # beat frequency

        # Amplitude from radar equation (simplified proportional form):
        # A ~ sqrt(RCS) / R^2
        A = np.sqrt(rcs) / (R ** 2 + 1e-6)

        # IF signal for this target
        phi_const = 2 * np.pi * fc * tau  # constant phase term
        s_target = A * np.exp(1j * (2 * np.pi * f_b * t + phi_const))
        s_if += s_target

        target_params.append({
            "range": R,
            "velocity": v,
            "rcs": rcs,
            "tau": tau,
            "f_doppler": f_d,
            "f_beat": f_b,
            "amplitude": A,
        })

    return s_if, target_params


def adc_sampling(s_if, N_samp, N_chirps):
    """Organize ADC samples into fast-time/slow-time matrix.

    In a real radar, each chirp produces N_samp ADC samples.
    Repeating for N_chirps gives a (N_chirps, N_samp) matrix.

    For simulation, we replicate the same IF signal across chirps
    with a phase progression to model slow-time Doppler.

    Args:
        s_if: Base IF signal for one chirp, shape (N_samp,).
        N_samp: Number of samples per chirp.
        N_chirps: Number of chirps in a frame.

    Returns:
        adc_data: Complex ADC matrix, shape (N_chirps, N_samp).
    """
    if len(s_if) != N_samp:
        # Resample to desired length
        t_old = np.linspace(0, 1, len(s_if))
        t_new = np.linspace(0, 1, N_samp)
        s_if_real = np.interp(t_new, t_old, s_if.real)
        s_if_imag = np.interp(t_new, t_old, s_if.imag)
        s_if = s_if_real + 1j * s_if_imag

    # Slow-time phase progression for Doppler resolution
    # Each chirp advances the Doppler phase by exp(j*2*pi*f_d*T_chirp)
    # For simulation we generate a base signal and later apply
    # per-chirp Doppler in processing.py via the 2nd FFT.
    adc_data = np.tile(s_if, (N_chirps, 1))

    return adc_data


def compute_received_signal_full(t, targets, fc, B, T_chirp, noise_dbm=-90):
    """Full received signal chain: chirp generation + mixing + sampling.

    This is the one-stop function for generating radar data.

    Args:
        t: Time vector (from generate_chirp).
        targets: List of target dicts.
        fc, B, T_chirp: Radar parameters.
        noise_dbm: Receiver noise floor in dBm (for adding thermal noise).

    Returns:
        s_if: Complex IF signal, shape (len(t),).
        target_params: Per-target beat frequency and amplitude info.
    """
    s_if, target_params = generate_if_signal(t, targets, fc, B, T_chirp)

    # Add complex Gaussian thermal noise
    signal_power = np.mean(np.abs(s_if) ** 2) + 1e-12
    noise_power_linear = 10 ** (noise_dbm / 10) * 1e-3  # dBm -> W (approx)
    # Scale noise relative to signal for simulation consistency
    noise_std = np.sqrt(signal_power / (10 ** (30 / 10)))  # ~30 dB SNR by default
    noise = (noise_std / np.sqrt(2)) * (
        np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    )
    s_if += noise

    return s_if, target_params
