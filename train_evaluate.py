"""
train_evaluate.py
-----------------
Training and evaluation pipeline for BEV occupancy forecasting.

Full pipeline:
  1. Load extracted Waymo frames from outputs/frames/
  2. Convert to BEV grids via IPM (bev_generator.py)
  3. Train ConvLSTM model to forecast T+1 grid from T past grids
  4. Evaluate with IoU on held-out test set
  5. Visualise predictions vs ground truth

Usage:
    python train_evaluate.py \
        --frames_dir  C:/Users/Lindsay/projects/waymo-dinov2/outputs/frames \
        --out_dir     ../outputs \
        --epochs      20 \
        --grid_size   128

Results are saved to outputs/:
  - training_curve.png     loss + IoU over epochs
  - predictions.png        predicted vs ground truth BEV grids
  - model.pth              saved model weights
  - results.txt            final IoU score
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from bev_generator import process_frames, visualise_bev_sample
from convlstm_model import BEVForecastNet, BEVSequenceDataset, compute_iou


# ── training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_iou  = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss  = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += compute_iou(preds.detach(), targets)

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou  = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        preds   = model(inputs)
        loss    = criterion(preds, targets)

        total_loss += loss.item()
        total_iou  += compute_iou(preds, targets)

    n = len(loader)
    return total_loss / n, total_iou / n


# ── visualisation ──────────────────────────────────────────────────────────────

def plot_predictions(model, dataset, out_path: str, device, n: int = 4):
    """
    Save a grid showing: input sequence | ground truth | prediction
    """
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 5, figsize=(18, 4 * n))
    fig.patch.set_facecolor("#111111")

    with torch.no_grad():
        for row, idx in enumerate(indices):
            inputs, target = dataset[idx]
            pred = model(inputs.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            gt   = target.squeeze().numpy()

            # Show last 3 input frames + gt + pred
            for col in range(3):
                axes[row, col].imshow(
                    inputs[col, 0].numpy(), cmap="hot", vmin=0, vmax=1
                )
                axes[row, col].set_title(
                    f"Input T-{2-col}", color="white", fontsize=9
                )
                axes[row, col].axis("off")

            axes[row, 3].imshow(gt,   cmap="hot", vmin=0, vmax=1)
            axes[row, 3].set_title("Ground truth T+1", color="white", fontsize=9)
            axes[row, 3].axis("off")

            iou = compute_iou(
                torch.tensor(pred).unsqueeze(0).unsqueeze(0),
                torch.tensor(gt).unsqueeze(0).unsqueeze(0)
            )
            axes[row, 4].imshow(pred, cmap="hot", vmin=0, vmax=1)
            axes[row, 4].set_title(
                f"Predicted T+1\nIoU={iou:.3f}", color="white", fontsize=9
            )
            axes[row, 4].axis("off")

    plt.suptitle(
        "BEV Occupancy Forecasting — ConvLSTM\n"
        "Input: 3 past grids → Predict: next grid (T+1)\n"
        "White = occupied   Black = free space",
        color="white", fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"  ✓ Predictions → {out_path}")


def plot_training_curves(
    train_losses, val_losses, train_ious, val_ious, out_path: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111111")
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "c-", label="Train loss", linewidth=2)
    ax1.plot(epochs, val_losses,   "m-", label="Val loss",   linewidth=2)
    ax1.set_title("Binary Cross-Entropy Loss", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss",  color="white")
    ax1.legend()
    ax1.set_facecolor("#1a1a1a")
    ax1.tick_params(colors="white")

    ax2.plot(epochs, train_ious, "c-", label="Train IoU", linewidth=2)
    ax2.plot(epochs, val_ious,   "m-", label="Val IoU",   linewidth=2)
    ax2.set_title("IoU Score (↑ better)", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("IoU",   color="white")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_facecolor("#1a1a1a")
    ax2.tick_params(colors="white")

    plt.suptitle(
        "ConvLSTM Training — BEV Occupancy Forecasting on Waymo Open Dataset",
        color="white", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"  ✓ Training curves → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ConvLSTM for BEV occupancy forecasting on Waymo data"
    )
    parser.add_argument("--frames_dir", required=True,
                        help="Folder with PNG camera frames (from Project 1 outputs)")
    parser.add_argument("--out_dir",    type=str, default="../outputs")
    parser.add_argument("--grid_size",  type=int, default=128)
    parser.add_argument("--seq_len",    type=int, default=3,
                        help="Number of past frames as input (default: 3)")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=int, default=32,
                        help="ConvLSTM hidden channels (default: 32)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bev_dir = os.path.join(args.out_dir, "bev_grids")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Grid size: {args.grid_size}×{args.grid_size}")
    print(f"Sequence length: {args.seq_len} frames → predict 1")

    # ── Step 1: Generate BEV grids ──────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Step 1: Camera frames → BEV occupancy grids")
    process_frames(
        frames_dir=args.frames_dir,
        out_dir=bev_dir,
        grid_size=args.grid_size,
        max_frames=500,
    )
    visualise_bev_sample(
        args.frames_dir,
        os.path.join(args.out_dir, "bev_ipm_sample.png"),
        args.grid_size,
        n_samples=3,
    )

    # ── Step 2: Build dataset ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Step 2: Building sequence dataset")
    dataset = BEVSequenceDataset(bev_dir, seq_len=args.seq_len,
                                 grid_size=args.grid_size)

    n_val   = max(4, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train} samples  |  Val: {n_val} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Step 3: Initialise model ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Step 3: Initialising ConvLSTM model")
    model     = BEVForecastNet(hidden_channels=args.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Step 4: Training loop ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Step 4: Training for {args.epochs} epochs …\n")

    train_losses, val_losses = [], []
    train_ious,   val_ious   = [], []
    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        vl_loss, vl_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_ious.append(tr_iou)
        val_ious.append(vl_iou)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"train_IoU={tr_iou:.4f}  val_IoU={vl_iou:.4f}")

        if vl_iou > best_val_iou:
            best_val_iou = vl_iou
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, "model_best.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pth"))

    # ── Step 5: Results ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Step 5: Generating output visualisations …")

    # Load best model for visualisation
    model.load_state_dict(
        torch.load(os.path.join(args.out_dir, "model_best.pth"),
                   map_location=device)
    )

    plot_training_curves(
        train_losses, val_losses, train_ious, val_ious,
        os.path.join(args.out_dir, "training_curves.png")
    )
    plot_predictions(
        model, val_ds.dataset, 
        os.path.join(args.out_dir, "predictions.png"),
        device, n=4
    )

    # Save results summary
    results_path = os.path.join(args.out_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("BEV Occupancy Forecasting Results\n")
        f.write("==================================\n\n")
        f.write(f"Dataset      : Waymo Open Dataset v2.0.1 (camera-based BEV)\n")
        f.write(f"Model        : ConvLSTM (hidden={args.hidden}, layers=2)\n")
        f.write(f"Grid size    : {args.grid_size}×{args.grid_size}\n")
        f.write(f"Seq length   : {args.seq_len} frames → predict 1\n")
        f.write(f"Parameters   : {n_params:,}\n")
        f.write(f"Epochs       : {args.epochs}\n\n")
        f.write(f"Final train IoU : {train_ious[-1]:.4f}\n")
        f.write(f"Final val IoU   : {val_ious[-1]:.4f}\n")
        f.write(f"Best val IoU    : {best_val_iou:.4f}\n")

    print(f"\n{'='*50}")
    print(f"✓ Training complete!")
    print(f"  Best val IoU : {best_val_iou:.4f}")
    print(f"  Outputs in   : {args.out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
