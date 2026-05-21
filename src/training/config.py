"""Configuration dataclasses for radar simulation and training.

All hyperparameters are centralized here for reproducibility.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch


# ---------------------------------------------------------------------------
# Radar parameters
# ---------------------------------------------------------------------------

@dataclass
class RadarConfig:
    """FMCW radar waveform parameters.

    Defaults match TI AWR1843 Boost configuration for automotive radar.
    """
    fc: float = 77e9           # Carrier frequency (Hz), 77 GHz
    B: float = 750e6           # Chirp bandwidth (Hz), 750 MHz -> 0.2m range resolution
    T_chirp: float = 40e-6     # Chirp duration (s), 40 us
    N_chirps: int = 64         # Chirps per frame
    fs: float = 10e6           # ADC sample rate (Hz), 10 MSPS
    N_samp: int = 256          # ADC samples per chirp
    frame_rate: float = 30.0   # Frame rate (Hz)

    @property
    def chirp_slope(self) -> float:
        """Chirp slope K = B/T_chirp (Hz/s)."""
        return self.B / self.T_chirp

    @property
    def wavelength(self) -> float:
        """Carrier wavelength (m)."""
        from scipy import constants
        return constants.speed_of_light / self.fc


@dataclass
class TargetConfig:
    """Target scenario configuration."""
    n_targets_min: int = 1
    n_targets_max: int = 5
    range_min: float = 5.0     # Minimum range (m)
    range_max: float = 150.0   # Maximum range (m)
    velocity_min: float = -30.0  # Minimum radial velocity (m/s)
    velocity_max: float = 30.0   # Maximum radial velocity (m/s)
    rcs_min: float = 0.1       # Minimum RCS (m^2)
    rcs_max: float = 100.0     # Maximum RCS (m^2)


@dataclass
class InterferenceConfig:
    """Interference simulation parameters."""
    types: Tuple[str, ...] = ("cw", "cross_fmcw", "noise_jamming", "spoofing", "chirp_swept")
    inr_min: float = -5.0      # Minimum INR (dB)
    inr_max: float = 25.0      # Maximum INR (dB)
    snr_dB: float = 25.0       # Base SNR for clean signal (dB)


# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Dataset generation and storage configuration."""
    n_samples: int = 12000     # Total samples to generate
    train_split: float = 0.8   # Fraction for training
    val_split: float = 0.1     # Fraction for validation
    test_split: float = 0.1    # Fraction for testing
    rd_size: int = 128         # RD map size (square: rd_size x rd_size)
    hdf5_path: str = "data/radar_dataset.h5"
    compression: str = "gzip"
    compression_level: int = 4


# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # General
    device: str = "auto"       # "auto", "cpu", "cuda"
    seed: int = 42

    # Classifier
    cls_epochs: int = 50
    cls_batch_size: int = 32
    cls_lr: float = 1e-3
    cls_weight_decay: float = 1e-4
    cls_scheduler: str = "cosine"  # "cosine", "plateau", "none"

    # Denoiser
    den_epochs: int = 80
    den_batch_size: int = 16
    den_lr: float = 1e-3
    den_weight_decay: float = 1e-5
    den_curriculum: bool = True    # Use progressive INR curriculum

    # U-Net feature dimensions
    unet_features: Tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

# TI AWR1843 Boost (medium-range automotive radar)
PRESET_TI_AWR1843 = RadarConfig(
    fc=77e9, B=750e6, T_chirp=40e-6, N_chirps=128, fs=10e6, N_samp=256
)

# TI AWR2243 Cascade (long-range, high-resolution imaging radar)
PRESET_TI_AWR2243_CASCADE = RadarConfig(
    fc=77e9, B=1.5e9, T_chirp=60e-6, N_chirps=256, fs=20e6, N_samp=512
)

# Short-range radar (parking assist, blind spot detection)
PRESET_SRR = RadarConfig(
    fc=79e9, B=4e9, T_chirp=20e-6, N_chirps=32, fs=40e6, N_samp=512
)
