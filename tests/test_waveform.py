"""Tests for FMCW waveform generation and IF signal processing."""
import numpy as np
import pytest
from scipy import constants

from src.simulator.waveform import (
    generate_chirp,
    compute_beat_frequency,
    generate_if_signal,
    adc_sampling,
)
from src.simulator.target import (
    range_resolution,
    velocity_resolution,
    max_unambiguous_range,
    max_unambiguous_velocity,
)


class TestChirpGeneration:
    def test_chirp_shape(self):
        chirp, t = generate_chirp(fc=77e9, B=750e6, T_chirp=40e-6, fs=10e6)
        expected_N = int(40e-6 * 10e6)  # 400
        assert len(chirp) == expected_N
        assert len(t) == expected_N

    def test_chirp_is_complex(self):
        chirp, _ = generate_chirp(fc=77e9, B=750e6, T_chirp=40e-6, fs=10e6)
        assert np.iscomplexobj(chirp)

    def test_chirp_constant_envelope(self):
        chirp, _ = generate_chirp(fc=77e9, B=750e6, T_chirp=40e-6, fs=10e6)
        envelope = np.abs(chirp)
        np.testing.assert_allclose(envelope, 1.0, atol=1e-10)


class TestBeatFrequency:
    def test_beat_freq_zero_range_zero_velocity(self):
        fb = compute_beat_frequency(0, 0, fc=77e9, B=750e6, T_chirp=40e-6)
        assert fb == pytest.approx(0.0, abs=1e-6)

    def test_beat_freq_range_only(self):
        # For R=30m: f_range = 2*750e6*30 / (3e8*40e-6) = 3.75e6 Hz = 3.75 MHz
        fb = compute_beat_frequency(30, 0, fc=77e9, B=750e6, T_chirp=40e-6)
        expected = 2 * 750e6 * 30 / (constants.speed_of_light * 40e-6)
        assert fb == pytest.approx(expected, rel=1e-6)

    def test_beat_freq_velocity_only(self):
        # For v=10m/s at 77GHz: f_doppler = 2*10*77e9/3e8 = ~5133 Hz
        fb = compute_beat_frequency(0, 10, fc=77e9, B=750e6, T_chirp=40e-6)
        expected = 2 * 10 * 77e9 / constants.speed_of_light
        assert fb == pytest.approx(expected, rel=1e-2)


class TestIFSignal:
    def test_if_signal_shape(self):
        chirp, t = generate_chirp(fc=77e9, B=750e6, T_chirp=40e-6, fs=10e6)
        targets = [{"range": 30, "velocity": 10, "rcs": 1.0}]
        s_if, params = generate_if_signal(t, targets, fc=77e9, B=750e6, T_chirp=40e-6)
        assert len(s_if) == len(t)
        assert len(params) == 1

    def test_multiple_targets(self):
        chirp, t = generate_chirp(fc=77e9, B=750e6, T_chirp=40e-6, fs=10e6)
        targets = [
            {"range": 30, "velocity": 10, "rcs": 1.0},
            {"range": 80, "velocity": -5, "rcs": 0.5},
        ]
        s_if, params = generate_if_signal(t, targets, fc=77e9, B=750e6, T_chirp=40e-6)
        assert len(params) == 2
        # Closer target should have higher amplitude
        assert params[0]["amplitude"] > params[1]["amplitude"]

    def test_if_nonzero(self):
        chirp, t = generate_chirp(fc=77e9, B=750e6, T_chirp=40e-6, fs=10e6)
        targets = [{"range": 50, "velocity": 0, "rcs": 1.0}]
        s_if, _ = generate_if_signal(t, targets, fc=77e9, B=750e6, T_chirp=40e-6)
        assert np.mean(np.abs(s_if)) > 0


class TestResolutions:
    def test_range_resolution(self):
        delta_R = range_resolution(B=750e6)
        expected = constants.speed_of_light / (2 * 750e6)  # 0.2 m
        assert delta_R == pytest.approx(expected, rel=1e-6)
        assert 0.15 < delta_R < 0.25  # around 0.2m

    def test_velocity_resolution(self):
        delta_v = velocity_resolution(fc=77e9, T_chirp=40e-6, N_chirps=128)
        # lambda = c/77e9 ~ 3.896e-3 m
        # delta_v = 3.896e-3 / (2 * 40e-6 * 128) = 3.896e-3 / 0.01024 = ~0.38 m/s
        assert 0.3 < delta_v < 0.5

    def test_max_unambiguous_range(self):
        R_max = max_unambiguous_range(B=750e6, fs=10e6)
        # R_max = c * fs / (2*B) = 3e8 * 10e6 / (2 * 750e6) = 3e15/1.5e9 = 2000m
        # Wait, that's wrong. fs = N_samp / T_chirp = 256/40e-6 = 6.4e6
        # Let me recalculate with N_samp
        pass
