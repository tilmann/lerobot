"""Train a per-cell classifier for Connect-4 board recognition.

This treats every cell as an individual training example, using the shared
board warp/crop path and then splitting the warped board into 42 cell patches.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fourinarow_cells import (
    BLACK,
    NUM_CLASSES,
    BoardSample,
    CellClassifier,
    extract_board_cells,
    list_board_samples,
    load_board_image,
    load_label,
    make_cell_transform,
)


class CellDataset(Dataset):
    def __init__(self, board_samples: list[BoardSample], *, transform, auto_crop_board: bool = False):
        self.transform = transform
        self.examples: list[tuple[np.ndarray, int, int, int]] = []

        for board_id, sample in enumerate(board_samples):
            board_img, _ = load_board_image(sample, auto_crop_board=auto_crop_board)
            label = load_label(sample.label_path)
            cells = extract_board_cells(board_img)
            for cell_idx, patch in enumerate(cells):
                row = cell_idx // label.shape[1]
                col = cell_idx % label.shape[1]
                self.examples.append((patch, int(label[row, col]), board_id, cell_idx))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        patch, label, board_id, cell_idx = self.examples[idx]
        tensor = self.transform(patch[:, :, ::-1])
        return tensor, label, board_id, cell_idx


def compute_class_weights(dataset: CellDataset, indices: list[int]) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for idx in indices:
        _, label, _, _ = dataset.examples[idx]
        counts[label] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = np.sum(counts) / (NUM_CLASSES * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def make_model(device: str) -> nn.Module:
    return CellClassifier().to(device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct_cells = 0
    total_cells = 0
    board_preds: dict[int, np.ndarray] = {}
    board_labels: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for imgs, labels, board_ids, cell_indices in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            correct_cells += (preds == labels).sum().item()
            total_cells += labels.numel()

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            board_ids_np = board_ids.cpu().numpy()
            cell_indices_np = cell_indices.cpu().numpy()

            for pred, target, board_id, cell_idx in zip(preds_np, labels_np, board_ids_np, cell_indices_np, strict=True):
                if board_id not in board_preds:
                    board_preds[board_id] = np.full(42, -1, dtype=np.int64)
                    board_labels[board_id] = np.full(42, -1, dtype=np.int64)
                board_preds[board_id][cell_idx] = pred
                board_labels[board_id][cell_idx] = target

    board_correct = 0
    for board_id in board_preds:
        if np.array_equal(board_preds[board_id], board_labels[board_id]):
            board_correct += 1

    val_loss = total_loss / len(loader.dataset)
    cell_acc = correct_cells / total_cells if total_cells else 0.0
    board_acc = board_correct / len(board_preds) if board_preds else 0.0
    return val_loss, cell_acc, board_acc


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    board_samples = list_board_samples(args.data)
    print(f"Found {len(board_samples)} labeled boards in {args.data}")
    if not board_samples:
        print("No labeled boards found.")
        return

    indices = list(range(len(board_samples)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_board_indices = indices[:split]
    val_board_indices = indices[split:]

    train_samples = [board_samples[idx] for idx in train_board_indices]
    val_samples = [board_samples[idx] for idx in val_board_indices]

    train_dataset = CellDataset(train_samples, transform=make_cell_transform(train=True), auto_crop_board=args.auto_crop_board)
    val_dataset = CellDataset(val_samples, transform=make_cell_transform(train=False), auto_crop_board=args.auto_crop_board)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train boards: {len(train_samples)}, Val boards: {len(val_samples)}")
    print(f"Train cells: {len(train_dataset)}, Val cells: {len(val_dataset)}")

    model = make_model(device)
    class_weights = None
    if args.class_weighted:
        class_weights = compute_class_weights(train_dataset, list(range(len(train_dataset)))).to(device)
        print("Class weights (empty, black, white):", [round(float(x), 4) for x in class_weights])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_board_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, labels, _board_ids, _cell_indices in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_dataset)
        val_loss, cell_acc, board_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  cell_acc={cell_acc:.3f}  board_acc={board_acc:.3f}"
        )

        if board_acc >= best_board_acc:
            best_board_acc = board_acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> Saved best model ({board_acc:.3f})")

    print(f"\nTraining complete. Best board accuracy: {best_board_acc:.3f}")
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/grid_labels")
    parser.add_argument("--out", default="models/grid_cell_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-crop-board", action="store_true")
    parser.add_argument("--class-weighted", action="store_true")
    args = parser.parse_args()
    train(args)