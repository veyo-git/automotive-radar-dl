"""Radar target model: range, velocity, RCS, and multi-target aggregation."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy import constants


@dataclass
class Target:
    """A single radar target with kinematics and radar cross section.

    Attributes:
        range: Range in meters.
        velocity: Radial velocity in m/s (positive = approaching radar).
        rcs: Radar cross section in m^2 (typical: car=10-100, pedestrian=0.5-2,
            bicycle=1-5).
        azimuth: Azimuth angle in degrees (0 = boresight). Optional.
        elevation: Elevation angle in degrees. Optional.
    """
    range: float
    velocity: float
    rcs: float = 1.0
    azimuth: Optional[float] = None
    elevation: Optional[float] = None

    def to_dict(self):
        """Convert to dict for waveform functions."""
        d = {"range": self.range, "velocity": self.velocity, "rcs": self.rcs}
        if self.azimuth is not None:
            d["azimuth"] = self.azimuth
        if self.elevation is not None:
            d["elevation"] = self.elevation
        return d


def compute_received_signal(chirp_t, targets, fc, B, T_chirp):
    """Build multi-target received signal in IF (beat) domain.

    Convenience wrapper around waveform.generate_if_signal that accepts
    Target objects and converts them automatically.

    Args:
        chirp_t: Time vector for one chirp, shape (N_samp,).
        targets: List of Target objects.
        fc: Carrier frequency (Hz).
        B: Chirp bandwidth (Hz).
        T_chirp: Chirp duration (s).

    Returns:
        s_if: Complex IF signal, shape (len(chirp_t),).
        target_params: List of per-target diagnostic info dicts.
    """
    from .waveform import generate_if_signal
    tgt_dicts = [t.to_dict() if isinstance(t, Target) else t for t in targets]
    return generate_if_signal(chirp_t, tgt_dicts, fc, B, T_chirp)


def max_unambiguous_range(B, fs):
    """Maximum unambiguous range for given bandwidth and sample rate.

    R_max = c * fs / (2 * K) where K = B / T_chirp and fs = N_samp / T_chirp.
    Simplified: R_max = c * N_samp / (2 * B)

    Args:
        B: Chirp bandwidth in Hz.
        fs: ADC sample rate in Hz. Alternatively pass N_samp and T_chirp.

    Returns:
        Maximum unambiguous range in meters.
    """
    c = constants.speed_of_light
    return c * fs / (2 * B)


def max_unambiguous_velocity(fc, T_chirp, N_chirps):
    """Maximum unambiguous velocity (radial).

    v_max = lambda / (4 * T_chirp) for a single chirp.
    With N_chirps in a frame: v_res = lambda / (2 * T_chirp * N_chirps).
    v_max = lambda / (4 * T_chirp)

    Args:
        fc: Carrier frequency in Hz.
        T_chirp: Chirp duration in seconds.
        N_chirps: Number of chirps per frame (used for resolution, not max).

    Returns:
        Maximum unambiguous radial velocity in m/s.
    """
    c = constants.speed_of_light
    wavelength = c / fc
    return wavelength / (4 * T_chirp)


def range_resolution(B):
    """Range resolution: delta_R = c / (2 * B)."""
    return constants.speed_of_light / (2 * B)


def velocity_resolution(fc, T_chirp, N_chirps):
    """Velocity resolution: delta_v = lambda / (2 * T_chirp * N_chirps)."""
    wavelength = constants.speed_of_light / fc
    return wavelength / (2 * T_chirp * N_chirps)
