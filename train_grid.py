"""
Train a CNN to detect the Connect 4 board state from a camera image.

Input:  640x480 RGB camera frame
Output: 42 cells, each classified as empty (0), black (1), or white (2)

Uses a ResNet18 backbone with the final layer replaced by a head that outputs
3 logits per cell (42 cells x 3 classes = 126 outputs). Trained with
cross-entropy loss per cell.

Usage:
    uv run python train_grid.py
    uv run python train_grid.py --epochs 50
    uv run python train_grid.py --data data/grid_labels --out models/grid_model.pt

The labeled data is expected in data/grid_labels/ with pairs:
    frame_NNNN.png  +  frame_NNNN.npy (6x7 array, values 0/1/2)
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from fourinarow_board import corners_to_warp_crop, crop_board

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROWS = 6
COLS = 7
NUM_CLASSES = 3  # empty, black, white
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GridDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, auto_crop_board: bool = False):
        self.data_dir = data_dir
        self.transform = transform
        self.auto_crop_board = auto_crop_board
        self.samples: list[tuple[str, str]] = []

        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".png"):
                npy = f.replace(".png", ".npy")
                npy_path = os.path.join(data_dir, npy)
                if os.path.exists(npy_path):
                    self.samples.append((
                        os.path.join(data_dir, f),
                        npy_path,
                    ))

        print(f"Found {len(self.samples)} labeled samples in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        corners_path = img_path.replace(".png", "_corners.npy")
        if os.path.exists(corners_path):
            corners = np.load(corners_path)
            if corners.shape == (4, 2):
                img = corners_to_warp_crop(img, corners)
        elif self.auto_crop_board:
            img, _ = crop_board(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        # Load label: 6x7 with values 0, 1, 2 -> flat 42 long tensor
        label = np.load(label_path).astype(np.int64).flatten()
        label = torch.from_numpy(label)

        return img, label


def compute_class_weights(
    samples: list[tuple[str, str]],
    indices: list[int],
) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for idx in indices:
        _, label_path = samples[idx]
        label = np.load(label_path).astype(np.int64).flatten()
        bincount = np.bincount(label, minlength=NUM_CLASSES).astype(np.float64)
        counts += bincount

    counts = np.maximum(counts, 1.0)
    total = np.sum(counts)
    weights = total / (NUM_CLASSES * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def make_model(device: str) -> nn.Module:
    """ResNet18 with a 3-class-per-cell head (42 cells x 3 = 126 outputs)."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model._modules["fc"] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, ROWS * COLS * NUM_CLASSES),
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    if args.auto_crop_board:
        print("Board auto-crop: enabled")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=3, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = GridDataset(args.data, transform=transform, auto_crop_board=args.auto_crop_board)

    if len(full_dataset) == 0:
        print("No samples found. Run label_grid.py first.")
        return

    # Train/val split (80/20)
    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = GridDataset(args.data, transform=val_transform, auto_crop_board=args.auto_crop_board)
    val_set = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    model = make_model(device)
    class_weights = None
    if args.class_weighted:
        class_weights = compute_class_weights(full_dataset.samples, train_indices).to(device)
        print("Class weights (empty, black, white):", [round(float(x), 4) for x in class_weights])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)  # (B, 126)
            # Reshape to (B*42, 3) for cross-entropy
            logits = logits.view(-1, NUM_CLASSES)
            targets = labels.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_set)

        # Validate
        model.eval()
        val_loss = 0.0
        correct_cells = 0
        total_cells = 0
        correct_boards = 0
        total_boards = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                logits_flat = logits.view(-1, NUM_CLASSES)
                targets_flat = labels.view(-1)
                loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item() * imgs.size(0)

                # Per-cell accuracy
                preds = logits.view(-1, ROWS * COLS, NUM_CLASSES).argmax(dim=2)
                correct_cells += (preds == labels).sum().item()
                total_cells += labels.numel()
                correct_boards += (preds == labels).all(dim=1).sum().item()
                total_boards += labels.size(0)

        val_loss /= len(val_set)
        cell_acc = correct_cells / total_cells if total_cells > 0 else 0
        board_acc = correct_boards / total_boards if total_boards > 0 else 0

        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"cell_acc={cell_acc:.3f}  board_acc={board_acc:.3f}")

        if board_acc >= best_val_acc:
            best_val_acc = board_acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  -> Saved best model ({board_acc:.3f})")

    print(f"\nTraining complete. Best board accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/grid_labels")
    parser.add_argument("--out", default="models/grid_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--auto-crop-board", action="store_true")
    parser.add_argument("--class-weighted", action="store_true")
    args = parser.parse_args()
    train(args)
