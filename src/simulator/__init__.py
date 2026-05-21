"""FMCW radar signal simulation: waveform, targets, processing, interference."""
from .waveform import generate_chirp, compute_beat_frequency, generate_if_signal, adc_sampling
from .target import Target, compute_received_signal, range_resolution, velocity_resolution
from .processing import range_doppler_map, compute_axes, ca_cfar, os_cfar
from .interference import (
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

__all__ = [
    "generate_chirp",
    "compute_beat_frequency",
    "generate_if_signal",
    "adc_sampling",
    "Target",
    "compute_received_signal",
    "range_resolution",
    "velocity_resolution",
    "range_doppler_map",
    "compute_axes",
    "ca_cfar",
    "os_cfar",
    "generate_cw_interference",
    "generate_cross_fmcw_interference",
    "generate_noise_jamming",
    "generate_spoofing",
    "generate_chirp_swept_interference",
    "inject_interference",
    "compute_signal_power",
    "INTERFERENCE_LABELS",
    "N_CLASSES",
]
