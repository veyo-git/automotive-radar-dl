"""Full evaluation benchmark for classifier and denoiser models.

Usage:
    python -m src.eval.benchmark --data data/radar_dataset.h5 \\
        --classifier models/classifier.pth --denoiser models/denoiser.pth
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.classifier import InterferenceClassifier
from src.models.denoiser import RadarUNet
from src.models.dataset import RadarDataset
from src.eval.metrics import classification_metrics


@torch.no_grad()
def evaluate_classifier(model, dataset, device, batch_size=32):
    """Full classification evaluation on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Classifying"):
        x = batch[0].to(device)
        if len(batch) == 3:
            y = batch[2]
        else:
            y = batch[1]
        all_labels.append(y.numpy())

        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    return classification_metrics(y_true, y_pred, dataset.class_names)


@torch.no_grad()
def evaluate_denoiser(model, dataset, device, batch_size=16):
    """Full denoising evaluation.

    Computes per-sample MSE and SINR improvement.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_mse = 0.0
    sinr_improvements = []
    n_samples = 0

    for batch in tqdm(loader, desc="Denoising"):
        interfered, clean, labels = batch
        interfered = interfered.to(device)
        clean_gt = clean.to(device)

        output = model(interfered)

        # MSE per sample
        for i in range(interfered.size(0)):
            mse_int = ((interfered[i] - clean_gt[i]) ** 2).mean().item()
            mse_out = ((output[i] - clean_gt[i]) ** 2).mean().item()
            total_mse += mse_out

            if mse_int > 1e-10 and labels[i].item() > 0:
                imp_db = 10 * np.log10(mse_int / max(mse_out, 1e-12))
                sinr_improvements.append(imp_db)

            n_samples += 1

    avg_mse = total_mse / max(n_samples, 1)
    avg_sinr_imp = np.mean(sinr_improvements) if sinr_improvements else 0.0

    # Percentiles
    p25 = np.percentile(sinr_improvements, 25) if sinr_improvements else 0
    p50 = np.percentile(sinr_improvements, 50) if sinr_improvements else 0
    p75 = np.percentile(sinr_improvements, 75) if sinr_improvements else 0

    return {
        "mse": float(avg_mse),
        "sinr_improvement_mean_db": float(avg_sinr_imp),
        "sinr_improvement_p25_db": float(p25),
        "sinr_improvement_p50_db": float(p50),
        "sinr_improvement_p75_db": float(p75),
        "n_samples": n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Full benchmark evaluation")
    parser.add_argument("--data", type=str, default="data/radar_dataset.h5")
    parser.add_argument("--classifier", type=str, default=None,
                        help="Path to classifier checkpoint")
    parser.add_argument("--denoiser", type=str, default=None,
                        help="Path to denoiser checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test set
    try:
        test_ds = RadarDataset(args.data, split="test", return_clean=True)
    except FileNotFoundError:
        print(f"Dataset not found: {args.data}")
        print("Run: python scripts/generate_dataset.py --n_samples 12000")
        sys.exit(1)

    print(f"Test samples: {len(test_ds)}")
    print(f"Classes: {test_ds.class_names}")
    print(f"Distribution: {test_ds.class_distribution()}")

    results = {}

    # Evaluate classifier
    if args.classifier and os.path.exists(args.classifier):
        print(f"\n{'='*50}")
        print("Evaluating Classifier")
        print(f"{'='*50}")

        checkpoint = torch.load(args.classifier, map_location=device, weights_only=False)
        n_classes = len(checkpoint.get("class_names", test_ds.class_names))
        model_cls = InterferenceClassifier(n_classes=n_classes).to(device)
        model_cls.load_state_dict(checkpoint["model_state_dict"])

        cls_metrics = evaluate_classifier(model_cls, test_ds, device)
        results["classifier"] = {
            "accuracy": float(cls_metrics["accuracy"]),
            "per_class_accuracy": {
                cls_metrics["class_names"][k]: float(v)
                for k, v in cls_metrics["per_class_accuracy"].items()
            },
        }
        results["classifier"]["confusion_matrix"] = cls_metrics["confusion_matrix"].tolist()

        print(f"Overall accuracy: {cls_metrics['accuracy']:.4f}")
        for name, acc in results["classifier"]["per_class_accuracy"].items():
            print(f"  {name:20s}: {acc:.4f}")

    # Evaluate denoiser
    if args.denoiser and os.path.exists(args.denoiser):
        print(f"\n{'='*50}")
        print("Evaluating Denoiser")
        print(f"{'='*50}")

        checkpoint = torch.load(args.denoiser, map_location=device, weights_only=False)
        features = checkpoint.get("features", (64, 128, 256, 512))
        model_den = RadarUNet(in_channels=1, out_channels=1, features=features)
        model_den = model_den.to(device)
        model_den.load_state_dict(checkpoint["model_state_dict"])

        den_metrics = evaluate_denoiser(model_den, test_ds, device)
        results["denoiser"] = den_metrics

        print(f"MSE: {den_metrics['mse']:.6f}")
        print(f"SINR improvement (mean): {den_metrics['sinr_improvement_mean_db']:.2f} dB")
        print(f"SINR improvement (P50):  {den_metrics['sinr_improvement_p50_db']:.2f} dB")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
