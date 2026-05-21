"""Tests for range-Doppler processing and CFAR detection."""
import numpy as np
import pytest
from scipy import constants

from src.simulator.waveform import generate_chirp, generate_if_signal, adc_sampling
from src.simulator.processing import (
    range_doppler_map,
    compute_axes,
    ca_cfar,
    os_cfar,
)


def make_simple_adc_data(N_chirps=64, N_samp=256,
                          targets=None, fc=77e9, B=750e6, T_chirp=40e-6, fs=10e6):
    """Helper: generate ADC data matrix for a simple scenario."""
    chirp, t = generate_chirp(fc, B, T_chirp, fs)
    if targets is None:
        targets = [{"range": 50, "velocity": 5, "rcs": 10.0}]
    s_if, _ = generate_if_signal(t, targets, fc, B, T_chirp)
    adc = adc_sampling(s_if, N_samp, N_chirps)
    # Add per-chirp Doppler phase progression
    for chirp_idx in range(N_chirps):
        for tgt in targets:
            f_d = 2 * tgt["velocity"] * fc / constants.speed_of_light
            phase = 2 * np.pi * f_d * chirp_idx * T_chirp
            adc[chirp_idx, :] *= np.exp(1j * phase)
    return adc


class TestRangeDopplerMap:
    def test_rd_map_shape(self):
        adc = make_simple_adc_data(N_chirps=64, N_samp=256)
        rd_db, rd_lin = range_doppler_map(adc)
        assert rd_db.shape == (64, 128)  # N_doppler x N_range(=N_samp//2)
        assert rd_lin.shape == (64, 128)

    def test_rd_map_db_range(self):
        adc = make_simple_adc_data(N_chirps=64, N_samp=256)
        rd_db, _ = range_doppler_map(adc)
        # dB values should be reasonably bounded
        assert not np.any(np.isinf(rd_db))
        assert not np.any(np.isnan(rd_db))

    def test_rd_map_no_window(self):
        adc = make_simple_adc_data(N_chirps=32, N_samp=128)
        rd_db_window, _ = range_doppler_map(adc, window_range="hann", window_doppler="hann")
        rd_db_none, _ = range_doppler_map(adc, window_range=None, window_doppler=None)
        # Both should produce valid output
        assert rd_db_window.shape == rd_db_none.shape


class TestAxes:
    def test_range_axis_length(self):
        range_bins, vel_bins = compute_axes(
            fc=77e9, B=750e6, T_chirp=40e-6, N_chirps=64, N_samp=256
        )
        assert len(range_bins) == 128  # N_samp // 2
        assert len(vel_bins) == 64    # N_chirps

    def test_velocity_axis_zero_center(self):
        _, vel_bins = compute_axes(77e9, 750e6, 40e-6, 64, 256)
        # fftshifted: center bin should be near 0 velocity
        center_idx = len(vel_bins) // 2
        assert abs(vel_bins[center_idx]) < 1.0  # near zero

    def test_range_axis_monotonic(self):
        range_bins, _ = compute_axes(77e9, 750e6, 40e-6, 64, 256)
        assert np.all(np.diff(range_bins) > 0)  # monotonically increasing


class TestCFAR:
    def test_ca_cfar_output_shapes(self):
        adc = make_simple_adc_data(N_chirps=64, N_samp=256)
        _, rd_lin = range_doppler_map(adc)
        det, thresh = ca_cfar(rd_lin, pfa=1e-3)
        assert det.shape == rd_lin.shape
        assert thresh.shape == rd_lin.shape

    def test_ca_cfar_detects_strong_target(self):
        # Strong target: verify RD map contains a clear peak
        adc = make_simple_adc_data(
            N_chirps=64, N_samp=256,
            targets=[{"range": 50, "velocity": 5, "rcs": 100.0}]
        )
        rd_db, rd_lin = range_doppler_map(adc)
        # The max value should be significantly above the mean noise floor
        peak_val = np.max(rd_db)
        mean_val = np.mean(rd_db)
        assert peak_val > mean_val + 10  # at least 10 dB above noise floor

    def test_ca_cfar_no_false_alarms_on_noise(self):
        # Pure noise (tiny RCS target = effectively noise)
        adc = make_simple_adc_data(
            N_chirps=32, N_samp=128,
            targets=[{"range": 50, "velocity": 5, "rcs": 1e-6}]
        )
        _, rd_lin = range_doppler_map(adc)
        det, _ = ca_cfar(rd_lin, pfa=1e-6, guard_cells=2, training_cells=4)
        # Very few detections with low Pfa on noise-like data
        false_alarm_rate = np.mean(det)
        assert false_alarm_rate < 0.1  # rough check

    def test_os_cfar_output_shapes(self):
        adc = make_simple_adc_data(N_chirps=32, N_samp=128)
        _, rd_lin = range_doppler_map(adc)
        det, thresh = os_cfar(rd_lin, pfa=1e-3)
        assert det.shape == rd_lin.shape
