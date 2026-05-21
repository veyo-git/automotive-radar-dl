"""Training script for U-Net denoiser with progressive curriculum learning.

Uses curriculum learning: start with high INR (easy — strong interference,
obvious to detect and remove), gradually decrease to low INR (hard — subtle
interference close to noise floor).
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.denoiser import RadarUNet, DenoisingLoss
from src.models.dataset import RadarDataset
from src.training.config import TrainingConfig


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        interfered, clean, _ = batch
        interfered = interfered.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        output = model(interfered)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and compute SINR improvement metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    sinr_improvements = []

    for batch in loader:
        interfered, clean, _ = batch
        interfered = interfered.to(device)
        clean_gt = clean.to(device)

        output = model(interfered)
        loss = criterion(output, clean_gt)
        total_loss += loss.item()
        n_batches += 1

        # Compute SINR improvement
        # SINR = max(clean) / mean(background)
        # We use a simplified metric: how much closer is the output to clean
        # compared to interfered
        for i in range(interfered.size(0)):
            int_power = ((interfered[i] - clean_gt[i]) ** 2).mean().item()
            out_power = ((output[i] - clean_gt[i]) ** 2).mean().item()
            if int_power > 1e-10:
                improvement_db = 10 * np.log10(int_power / max(out_power, 1e-12))
                sinr_improvements.append(improvement_db)

    avg_loss = total_loss / max(n_batches, 1)
    avg_sinr = np.mean(sinr_improvements) if sinr_improvements else 0.0
    return avg_loss, avg_sinr


def main():
    parser = argparse.ArgumentParser(description="Train U-Net denoiser")
    parser.add_argument("--data", type=str, default="data/radar_dataset.h5")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="models/denoiser.pth")
    parser.add_argument("--features", type=str, default="64,128,256,512",
                        help="U-Net feature dimensions, comma-separated")
    parser.add_argument("--no_curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    features = tuple(int(f.strip()) for f in args.features.split(","))
    print(f"U-Net features: {features}")

    # Load datasets
    train_ds = RadarDataset(args.data, split="train", return_clean=True)
    val_ds = RadarDataset(args.data, split="val", return_clean=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Model
    model = RadarUNet(in_channels=1, out_channels=1, features=features)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = DenoisingLoss(mse_weight=1.0, grad_weight=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_sinr_db": []}

    # Curriculum phases: decrease MSE weight schedule to focus on harder samples
    if args.no_curriculum:
        phases = [(0, args.epochs)]
    else:
        phases = [
            (0, args.epochs // 3),           # phase 1: standard
            (args.epochs // 3, 2 * args.epochs // 3),  # phase 2
            (2 * args.epochs // 3, args.epochs),       # phase 3
        ]

    for epoch in range(1, args.epochs + 1):
        # Curriculum: gradually increase gradient loss weight
        if not args.no_curriculum:
            progress = epoch / args.epochs
            criterion.grad_weight = 0.05 + 0.15 * progress  # 0.05 -> 0.20

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_sinr = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_sinr_db"].append(val_sinr)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"SINR imp: {val_sinr:.2f} dB")

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_sinr_db": val_sinr,
                "features": features,
                "history": history,
            }, args.output)
            print(f"  -> Saved checkpoint (loss: {val_loss:.4f}, SINR: {val_sinr:.2f} dB)")

    print(f"\nBest val loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
