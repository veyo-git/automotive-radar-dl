"""Training script for interference type classifier.

Trains a 2D CNN to classify interference types from Range-Doppler maps.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.classifier import InterferenceClassifier
from src.models.dataset import RadarDataset
from src.training.config import TrainingConfig


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if len(batch) == 3:
            x, _, y = batch  # interfered, clean, label
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        if len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Per-class accuracy
    per_class_acc = {}
    for c in range(model.n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == c).float().mean().item()

    return total_loss / total, correct / total, per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Train interference classifier")
    parser.add_argument("--data", type=str, default="data/radar_dataset.h5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="models/classifier.pth",
                        help="Model checkpoint save path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    train_ds = RadarDataset(args.data, split="train", return_clean=False)
    val_ds = RadarDataset(args.data, split="val", return_clean=False)
    n_classes = train_ds.n_classes
    print(f"Classes: {train_ds.class_names}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Class distribution (train): {train_ds.class_distribution()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Model, loss, optimizer, scheduler
    model = InterferenceClassifier(n_classes=n_classes).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Class-balanced loss
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, per_class = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": train_ds.class_names,
                "history": history,
            }, args.output)
            print(f"  -> Saved checkpoint (acc: {val_acc:.3f})")

    print(f"\nBest val accuracy: {best_acc:.3f}")
    print(f"Checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
