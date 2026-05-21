"""Tests for interference generators."""
import numpy as np
import pytest
from scipy import constants

from src.simulator.interference import (
    generate_cw_interference,
    generate_cross_fmcw_interference,
    generate_noise_jamming,
    generate_spoofing,
    generate_chirp_swept_interference,
    inject_interference,
    compute_signal_power,
    INTERFERENCE_LABELS,
    N_CLASSES,
)


def make_time_vector(N_samp=400, T_chirp=40e-6):
    """Create a time vector matching typical FMCW chirp sampling."""
    return np.arange(N_samp) / (N_samp / T_chirp)


class TestCWInterference:
    def test_tone_shape(self):
        t = make_time_vector()
        i_cw = generate_cw_interference(t, fc_if=1e6, amplitude=2.0)
        assert len(i_cw) == len(t)
        assert np.iscomplexobj(i_cw)

    def test_tone_power(self):
        t = make_time_vector()
        i_cw = generate_cw_interference(t, fc_if=1e6, amplitude=3.0)
        power = compute_signal_power(i_cw)
        assert power == pytest.approx(9.0, rel=0.05)  # amplitude^2


class TestCrossFMCWInterference:
    def test_basic_shape(self):
        t = make_time_vector()
        K1 = 750e6 / 40e-6   # victim: 1.875e13 Hz/s
        K2 = 800e6 / 38e-6   # interferer: 2.105e13 Hz/s (different)
        i_fmcw = generate_cross_fmcw_interference(t, K1, K2, delta_t=1e-6)
        assert len(i_fmcw) == len(t)

    def test_same_slope_no_offset(self):
        t = make_time_vector()
        K = 750e6 / 40e-6
        i_fmcw = generate_cross_fmcw_interference(t, K, K, delta_t=0, fc_offset=0)
        # Same slope, no offset -> constant frequency (CW-like)
        assert len(i_fmcw) == len(t)

    def test_different_slopes_produce_interference(self):
        t = make_time_vector()
        K_victim = 750e6 / 40e-6   # 18.75e12 Hz/s
        K_interf = 500e6 / 50e-6   # 10e12 Hz/s
        i_fmcw = generate_cross_fmcw_interference(
            t, K_victim, K_interf, delta_t=0.5e-6, fc_offset=2e6
        )
        power = compute_signal_power(i_fmcw)
        assert power > 0


class TestNoiseJamming:
    def test_shape(self):
        t = make_time_vector()
        i_noise = generate_noise_jamming(t, amplitude=2.0)
        assert len(i_noise) == len(t)

    def test_approximate_power(self):
        t = make_time_vector(10000)  # more samples for statistical accuracy
        i_noise = generate_noise_jamming(t, amplitude=2.0)
        power = compute_signal_power(i_noise)
        assert power == pytest.approx(4.0, rel=0.1)  # E[|noise|^2] = amplitude^2


class TestSpoofing:
    def test_basic_shape(self):
        t = make_time_vector()
        i_spoof = generate_spoofing(
            t, fc=77e9, B=750e6, T_chirp=40e-6,
            fake_range=60, fake_velocity=8, amplitude=1.0
        )
        assert len(i_spoof) == len(t)

    def test_spoof_is_coherent(self):
        # Spoofing should be a pure tone (coherent replica of victim chirp)
        t = make_time_vector()
        i_spoof = generate_spoofing(
            t, fc=77e9, B=750e6, T_chirp=40e-6,
            fake_range=60, fake_velocity=0, amplitude=1.0
        )
        # Should have nearly constant envelope
        envelope = np.abs(i_spoof)
        envelope_std = np.std(envelope)
        assert envelope_std < 1e-6  # pure tone has constant envelope


class TestChirpSwept:
    def test_basic_shape(self):
        t = make_time_vector()
        i_swept = generate_chirp_swept_interference(
            t, f_start=0.5e6, f_stop=5e6, amplitude=1.0
        )
        assert len(i_swept) == len(t)

    def test_frequency_sweep_range(self):
        t = make_time_vector()
        f_start, f_stop = 0.1e6, 4e6
        i_swept = generate_chirp_swept_interference(
            t, f_start=f_start, f_stop=f_stop, amplitude=1.0
        )
        # Verify by checking that phase diff (instantaneous freq) spans the range
        phase = np.unwrap(np.angle(i_swept))
        freq_instant = np.diff(phase) / (2 * np.pi * (t[1] - t[0]))
        # Should roughly span f_start to f_stop
        assert freq_instant[0] == pytest.approx(f_start, rel=0.3)
        assert freq_instant[-1] == pytest.approx(f_stop, rel=0.3)


class TestInjection:
    def test_inject_increases_power(self):
        t = make_time_vector()
        # Create a simple IF signal (tone at 2 MHz)
        s_if = np.exp(1j * 2 * np.pi * 2e6 * t) + 0.1 * (
            np.random.randn(len(t)) + 1j * np.random.randn(len(t))
        )
        i_cw = generate_cw_interference(t, fc_if=3e6, amplitude=1.0)
        power_before = compute_signal_power(s_if)
        s_interfered, _ = inject_interference(s_if, i_cw, snr_db=20, inr_db=20)
        power_after = compute_signal_power(s_interfered)
        assert power_after > power_before

    def test_higher_inr_stronger_interference(self):
        t = make_time_vector()
        s_if = np.ones(len(t), dtype=complex)  # simple signal
        i_noise = generate_noise_jamming(t, amplitude=1.0)

        s_low, _ = inject_interference(s_if, i_noise, snr_db=30, inr_db=0)
        s_high, _ = inject_interference(s_if, i_noise, snr_db=30, inr_db=20)

        # Higher INR = more power added
        power_low = compute_signal_power(s_low - s_if)
        power_high = compute_signal_power(s_high - s_if)
        assert power_high > power_low


class TestLabels:
    def test_n_classes(self):
        assert N_CLASSES == 6  # clean + 5 interference types

    def test_label_roundtrip(self):
        for idx, name in INTERFERENCE_LABELS.items():
            assert idx < N_CLASSES
            assert isinstance(name, str)
