"""Generate synthetic radar interference dataset and save to HDF5.

Usage:
    python scripts/generate_dataset.py --n_samples 12000 --rd_size 128
    python scripts/generate_dataset.py --n_samples 1000 --rd_size 64 --output data/small.h5
"""
import argparse
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator.waveform import generate_chirp, generate_if_signal, adc_sampling
from src.simulator.processing import range_doppler_map, compute_axes
from src.simulator.interference import (
    generate_cw_interference,
    generate_cross_fmcw_interference,
    generate_noise_jamming,
    generate_spoofing,
    generate_chirp_swept_interference,
    inject_interference,
    INTERFERENCE_LABELS,
    N_CLASSES,
)
from src.training.config import RadarConfig, TargetConfig, InterferenceConfig


def generate_one_sample(radar, tgt_cfg, int_cfg, rd_size, rng):
    """Generate one (clean, interfered, label) sample.

    Args:
        radar: RadarConfig.
        tgt_cfg: TargetConfig.
        int_cfg: InterferenceConfig.
        rd_size: Target RD map size (square).
        rng: NumPy RandomState for reproducibility.

    Returns:
        clean_rd: Clean RD map (dB), shape (rd_size, rd_size).
        interfered_rd: Interfered RD map (dB), shape (rd_size, rd_size).
        label: Integer interference type label.
        inr_db: Actual INR used (dB).
    """
    # Random target scenario
    n_targets = rng.randint(tgt_cfg.n_targets_min, tgt_cfg.n_targets_max + 1)
    targets = []
    for _ in range(n_targets):
        targets.append({
            "range": rng.uniform(tgt_cfg.range_min, tgt_cfg.range_max),
            "velocity": rng.uniform(tgt_cfg.velocity_min, tgt_cfg.velocity_max),
            "rcs": 10 ** rng.uniform(np.log10(tgt_cfg.rcs_min),
                                      np.log10(tgt_cfg.rcs_max)),
        })

    # Random interference type (0=clean, 1-5=interference types)
    if rng.random() < 0.12:
        # ~12% clean samples (class balance)
        label = 0
    else:
        label = rng.randint(1, N_CLASSES)

    # Generate clean IF signal
    chirp, t = generate_chirp(radar.fc, radar.B, radar.T_chirp, radar.fs)
    s_if, tgt_params = generate_if_signal(t, targets, radar.fc, radar.B, radar.T_chirp)

    # Add thermal noise
    snr_linear = 10 ** (int_cfg.snr_dB / 10)
    signal_power = np.mean(np.abs(s_if) ** 2) + 1e-12
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.randn(len(t)) + 1j * rng.randn(len(t))
    )
    s_clean = s_if + noise

    # Apply interference if label != 0
    if label == 0:
        s_interfered = s_clean
        inr_db = -999  # no interference
    else:
        inr_db = rng.uniform(int_cfg.inr_min, int_cfg.inr_max)

        # Generate the appropriate interference type
        if label == 1:  # CW
            f_cw = rng.uniform(0.1e6, radar.fs / 2 * 0.9)
            i_sig = generate_cw_interference(t, fc_if=f_cw, amplitude=1.0,
                                              phase=rng.uniform(0, 2 * np.pi))

        elif label == 2:  # Cross-FMCW
            # Use a different chirp slope for the interferer
            K_victim = radar.B / radar.T_chirp
            B_int = rng.uniform(0.5 * radar.B, 2 * radar.B)
            T_int = rng.uniform(0.5 * radar.T_chirp, 2 * radar.T_chirp)
            K_interferer = B_int / T_int
            delta_t = rng.uniform(-radar.T_chirp * 0.5, radar.T_chirp * 0.5)
            fc_off = rng.uniform(-5e6, 5e6)  # few MHz carrier offset
            i_sig = generate_cross_fmcw_interference(
                t, K_victim, K_interferer, delta_t=delta_t, fc_offset=fc_off
            )

        elif label == 3:  # Noise jamming
            i_sig = generate_noise_jamming(t, amplitude=1.0)

        elif label == 4:  # Spoofing
            fake_R = rng.uniform(10, 200)
            fake_v = rng.uniform(-30, 30)
            i_sig = generate_spoofing(
                t, radar.fc, radar.B, radar.T_chirp,
                fake_range=fake_R, fake_velocity=fake_v
            )

        elif label == 5:  # Chirp-swept
            f_s = rng.uniform(0.1e6, radar.fs / 4)
            f_e = rng.uniform(radar.fs / 4, radar.fs / 2 * 0.9)
            i_sig = generate_chirp_swept_interference(t, f_start=f_s, f_stop=f_e)

        else:
            raise ValueError(f"Unknown label: {label}")

        s_interfered, _ = inject_interference(
            s_clean, i_sig, snr_db=int_cfg.snr_dB, inr_db=inr_db
        )

    # ADC -> RD processing
    adc_clean = adc_sampling(s_clean, radar.N_samp, radar.N_chirps)
    adc_int = adc_sampling(s_interfered, radar.N_samp, radar.N_chirps)

    # Apply per-chirp Doppler phase for realistic slow-time processing
    for chirp_idx in range(radar.N_chirps):
        for tgt in targets:
            f_d = 2 * tgt["velocity"] * radar.fc / 299792458.0
            phase = 2 * np.pi * f_d * chirp_idx * radar.T_chirp
            adc_clean[chirp_idx] *= np.exp(1j * phase)
            adc_int[chirp_idx] *= np.exp(1j * phase)

    rd_clean_db, rd_clean_lin = range_doppler_map(adc_clean)
    rd_int_db, rd_int_lin = range_doppler_map(adc_int)

    # Resize to target size (center crop or pad)
    clean_resized = _resize_2d(rd_clean_db, rd_size)
    int_resized = _resize_2d(rd_int_db, rd_size)

    return clean_resized, int_resized, label, inr_db


def _resize_2d(arr, target_size):
    """Resize a 2D array to target_size x target_size via center crop or pad."""
    h, w = arr.shape
    # Center crop
    if h > target_size:
        start_h = (h - target_size) // 2
        arr = arr[start_h:start_h + target_size, :]
    if w > target_size:
        start_w = (w - target_size) // 2
        arr = arr[:, start_w:start_w + target_size]
    # Pad if needed
    if arr.shape[0] < target_size:
        pad_h = target_size - arr.shape[0]
        arr = np.pad(arr, ((pad_h // 2, pad_h - pad_h // 2), (0, 0)),
                     mode="constant", constant_values=-60)
    if arr.shape[1] < target_size:
        pad_w = target_size - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (pad_w // 2, pad_w - pad_w // 2)),
                     mode="constant", constant_values=-60)
    return arr[:target_size, :target_size]


def main():
    parser = argparse.ArgumentParser(description="Generate radar interference dataset")
    parser.add_argument("--n_samples", type=int, default=12000,
                        help="Total number of samples")
    parser.add_argument("--rd_size", type=int, default=128,
                        help="Range-Doppler map size (square)")
    parser.add_argument("--output", type=str, default="data/radar_dataset.h5",
                        help="Output HDF5 path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Configs
    radar = RadarConfig()
    tgt_cfg = TargetConfig()
    int_cfg = InterferenceConfig()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Calculate split sizes
    n_train = int(args.n_samples * args.train_split)
    n_val = int(args.n_samples * args.val_split)
    n_test = args.n_samples - n_train - n_val

    print(f"Generating {args.n_samples} samples ({n_train}/{n_val}/{n_test} split)")
    print(f"RD map size: {args.rd_size}x{args.rd_size}")
    print(f"Output: {args.output}")
    print(f"Classes: {list(INTERFERENCE_LABELS.values())}")
    print()

    # Pre-allocate arrays for each split
    samples = {"train": n_train, "val": n_val, "test": n_test}

    # Generate and write to HDF5
    with h5py.File(args.output, "w") as f:
        # Write global metadata
        f.attrs["n_classes"] = N_CLASSES
        f.attrs["class_names"] = list(INTERFERENCE_LABELS.values())
        f.attrs["rd_size"] = args.rd_size
        f.attrs["radar_fc"] = radar.fc
        f.attrs["radar_B"] = radar.B
        f.attrs["radar_T_chirp"] = radar.T_chirp
        f.attrs["seed"] = args.seed

        for split, count in samples.items():
            # Create datasets
            grp = f.create_group(split)
            clean_ds = grp.create_dataset(
                "clean", (count, args.rd_size, args.rd_size),
                dtype=np.float32, compression="gzip", compression_opts=4
            )
            interfered_ds = grp.create_dataset(
                "interfered", (count, args.rd_size, args.rd_size),
                dtype=np.float32, compression="gzip", compression_opts=4
            )
            labels_ds = grp.create_dataset("labels", (count,), dtype=np.int64)
            inr_ds = grp.create_dataset("inr_db", (count,), dtype=np.float32)

            # Generate samples
            labels_count = np.zeros(N_CLASSES, dtype=int)
            for i in tqdm(range(count), desc=f"Generating {split}"):
                clean_rd, int_rd, label, inr_db = generate_one_sample(
                    radar, tgt_cfg, int_cfg, args.rd_size, rng
                )
                clean_ds[i] = clean_rd
                interfered_ds[i] = int_rd
                labels_ds[i] = label
                inr_ds[i] = inr_db
                labels_count[label] += 1

            grp.attrs["n_samples"] = count
            grp.attrs["label_distribution"] = labels_count

            print(f"  {split}: {dict(zip(INTERFERENCE_LABELS.values(), labels_count))}")

    print(f"\nDataset saved to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
