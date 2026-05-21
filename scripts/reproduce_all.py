"""One-click reproduction of all experiments.

Generates dataset, trains classifier and denoiser, runs full benchmark.

Usage:
    python scripts/reproduce_all.py --n_samples 12000 --seed 42
    python scripts/reproduce_all.py --n_samples 2000 --rd_size 64  # quick test
"""
import os
import sys
import argparse
import subprocess
import time


def run_step(cmd, description):
    """Run a command and print its output in real-time."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {cmd}")
    print()

    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ERROR: Step failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n  Completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all radar-dl experiments"
    )
    parser.add_argument("--n_samples", type=int, default=12000)
    parser.add_argument("--rd_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs_cls", type=int, default=50)
    parser.add_argument("--epochs_den", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_dataset", action="store_true",
                        help="Skip dataset generation (use existing)")
    parser.add_argument("--skip_cls", action="store_true",
                        help="Skip classifier training")
    parser.add_argument("--skip_den", action="store_true",
                        help="Skip denoiser training")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    steps = []
    total_start = time.time()

    # Step 1: Generate dataset
    if not args.skip_dataset:
        steps.append((
            f"python scripts/generate_dataset.py "
            f"--n_samples {args.n_samples} --rd_size {args.rd_size} "
            f"--seed {args.seed}",
            f"Step 1/4: Generate Dataset ({args.n_samples} samples, {args.rd_size}x{args.rd_size})"
        ))

    # Step 2: Train classifier
    if not args.skip_cls:
        steps.append((
            f"python src/training/train_classifier.py "
            f"--epochs {args.epochs_cls} --batch_size {args.batch_size} "
            f"--device {args.device} --seed {args.seed}",
            f"Step 2/4: Train Classifier ({args.epochs_cls} epochs)"
        ))

    # Step 3: Train denoiser
    if not args.skip_den:
        steps.append((
            f"python src/training/train_denoiser.py "
            f"--epochs {args.epochs_den} --batch_size {max(args.batch_size // 2, 8)} "
            f"--device {args.device} --seed {args.seed}",
            f"Step 3/4: Train Denoiser ({args.epochs_den} epochs)"
        ))

    # Step 4: Benchmark
    steps.append((
        f"python src/eval/benchmark.py "
        f"--classifier models/classifier.pth --denoiser models/denoiser.pth "
        f"--device {args.device}",
        "Step 4/4: Run Full Benchmark"
    ))

    for cmd, desc in steps:
        run_step(cmd, desc)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"  Check results/: benchmark.json")
    print(f"  Launch demo: python src/viz/app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
