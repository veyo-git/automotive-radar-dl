"""Shared pytest fixtures for radar-dl tests."""
import pytest
import numpy as np

# TI AWR1843-like parameters (77 GHz automotive radar)
RADAR_77GHZ = {
    "fc": 77e9,        # 77 GHz carrier
    "B": 750e6,        # 750 MHz bandwidth -> 0.2m range resolution
    "T_chirp": 40e-6,  # 40 us chirp duration
    "N_chirps": 64,    # 64 chirps per frame
    "fs": 10e6,        # 10 MSPS ADC
    "N_samp": 256,     # 256 samples per chirp
}

SIMPLE_TARGETS = [
    {"range": 30.0, "velocity": 10.0, "rcs": 1.0},
    {"range": 80.0, "velocity": -5.0, "rcs": 0.5},
]

INTERFERENCE_TYPES = ["cw", "cross_fmcw", "noise", "spoofing", "chirp_swept"]


@pytest.fixture
def radar_params():
    return RADAR_77GHZ.copy()


@pytest.fixture
def simple_targets():
    return SIMPLE_TARGETS.copy()
