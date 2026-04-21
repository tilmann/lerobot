"""Diagnostic checks for Connect-4 training and inference mismatch.

Usage:
    uv run python diagnose_grid_pipeline.py
    uv run python diagnose_grid_pipeline.py --model models/grid_model.pt
    uv run python diagnose_grid_pipeline.py --data data/grid_labels --limit 50
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from fourinarow_board import corners_to_warp_crop, crop_board

ROWS = 6
COLS = 7
NUM_CLASSES = 3
IMG_SIZE = 224

EMPTY = 0
BLACK = 1
WHITE = 2

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class Sample:
    name: str
    img_path: Path
    label_path: Path
    corners_path: Path


@dataclass
class LabelChecks:
    black_count: int
    white_count: int
    empty_count: int
    illegal_turn_balance: bool
    gravity_violations: int


def load_model(path: str, device: str) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model._modules["fc"] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, ROWS * COLS * NUM_CLASSES),
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def list_samples(data_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for entry in sorted(data_dir.glob("frame_*.png")):
        label_path = entry.with_suffix(".npy")
        if not label_path.exists():
            continue
        corners_path = entry.with_name(entry.stem + "_corners.npy")
        samples.append(
            Sample(
                name=entry.name,
                img_path=entry,
                label_path=label_path,
                corners_path=corners_path,
            )
        )
    return samples


def load_label(label_path: Path) -> np.ndarray:
    return np.load(label_path).astype(np.int64).reshape(ROWS, COLS)


def check_gravity(label: np.ndarray) -> int:
    violations = 0
    for col in range(COLS):
        found_empty_below_disc = False
        for row in range(ROWS - 1, -1, -1):
            value = int(label[row, col])
            if value == EMPTY:
                found_empty_below_disc = True
            elif found_empty_below_disc:
                violations += 1
    return violations


def check_label(label: np.ndarray) -> LabelChecks:
    black_count = int(np.sum(label == BLACK))
    white_count = int(np.sum(label == WHITE))
    empty_count = int(np.sum(label == EMPTY))
    illegal_turn_balance = abs(black_count - white_count) > 1
    gravity_violations = check_gravity(label)
    return LabelChecks(
        black_count=black_count,
        white_count=white_count,
        empty_count=empty_count,
        illegal_turn_balance=illegal_turn_balance,
        gravity_violations=gravity_violations,
    )


def preprocess_train_like(frame: np.ndarray, sample: Sample, auto_crop_board: bool) -> tuple[np.ndarray, str]:
    if sample.corners_path.exists():
        corners = np.load(sample.corners_path)
        if corners.shape == (4, 2):
            return corners_to_warp_crop(frame, corners), "corners"
    if auto_crop_board:
        cropped, _ = crop_board(frame)
        return cropped, "auto_crop"
    return frame, "raw"


def preprocess_detect_like(frame: np.ndarray, sample: Sample, auto_crop_board: bool) -> tuple[np.ndarray, str]:
    # Equivalent to detect path when a per-frame calibration is available.
    if sample.corners_path.exists():
        corners = np.load(sample.corners_path)
        if corners.shape == (4, 2):
            return corners_to_warp_crop(frame, corners), "corners"
    if auto_crop_board:
        cropped, _ = crop_board(frame)
        return cropped, "auto_crop"
    return frame, "raw"


def to_tensor(bgr_frame: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return TRANSFORM(rgb).unsqueeze(0).to(device)


def raw_to_gravity_board(raw_board: np.ndarray) -> np.ndarray:
    board = np.zeros((ROWS, COLS), dtype=np.int64)
    for col in range(COLS):
        black_count = int(np.sum(raw_board[:, col] == BLACK))
        white_count = int(np.sum(raw_board[:, col] == WHITE))
        row = ROWS - 1
        for _ in range(black_count):
            if row >= 0:
                board[row, col] = BLACK
                row -= 1
        for _ in range(white_count):
            if row >= 0:
                board[row, col] = WHITE
                row -= 1
    return board


def strict_board_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/grid_labels")
    parser.add_argument("--model", default="models/grid_model.pt")
    parser.add_argument("--auto-crop-board", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples")
    parser.add_argument("--disc-bias", type=float, default=0.0,
                        help="Positive values make black/white easier to predict than empty")
    args = parser.parse_args()

    data_dir = Path(args.data)
    samples = list_samples(data_dir)
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Samples: {len(samples)}")
    if not samples:
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.model, device)

    total_empty = 0
    total_black = 0
    total_white = 0
    illegal_turn_files: list[str] = []
    gravity_issue_files: list[str] = []

    parity_sum_mae = 0.0
    parity_max_mae = 0.0
    parity_mismatch_count = 0

    raw_cell_correct = 0
    post_cell_correct = 0
    total_cells = 0
    raw_board_correct = 0
    post_board_correct = 0

    post_changed_samples = 0
    post_changed_cells = 0

    for sample in samples:
        frame = cv2.imread(str(sample.img_path))
        if frame is None:
            print(f"WARN cannot read {sample.img_path}")
            continue

        label = load_label(sample.label_path)
        checks = check_label(label)

        total_empty += checks.empty_count
        total_black += checks.black_count
        total_white += checks.white_count

        if checks.illegal_turn_balance:
            illegal_turn_files.append(sample.name)
        if checks.gravity_violations > 0:
            gravity_issue_files.append(sample.name)

        train_img, train_path = preprocess_train_like(frame, sample, args.auto_crop_board)
        detect_img, detect_path = preprocess_detect_like(frame, sample, args.auto_crop_board)

        if train_path != detect_path:
            parity_mismatch_count += 1

        train_resized = cv2.resize(train_img, (IMG_SIZE, IMG_SIZE))
        detect_resized = cv2.resize(detect_img, (IMG_SIZE, IMG_SIZE))
        mae = float(np.mean(np.abs(train_resized.astype(np.float32) - detect_resized.astype(np.float32))))
        parity_sum_mae += mae
        parity_max_mae = max(parity_max_mae, mae)

        with torch.no_grad():
            logits = model(to_tensor(detect_img, device))
            logits = logits.view(ROWS, COLS, NUM_CLASSES)
            if args.disc_bias != 0.0:
                logits[:, :, BLACK] += args.disc_bias
                logits[:, :, WHITE] += args.disc_bias
            probs = torch.softmax(logits, dim=2).cpu().numpy()

        raw_board = np.argmax(probs, axis=2).astype(np.int64)
        post_board = raw_to_gravity_board(raw_board)

        raw_cell_correct += int(np.sum(raw_board == label))
        post_cell_correct += int(np.sum(post_board == label))
        total_cells += ROWS * COLS

        raw_board_correct += 1 if strict_board_equal(raw_board, label) else 0
        post_board_correct += 1 if strict_board_equal(post_board, label) else 0

        changed = raw_board != post_board
        changed_count = int(np.sum(changed))
        if changed_count > 0:
            post_changed_samples += 1
            post_changed_cells += changed_count

    n = max(1, len(samples))

    print("\n=== Label Integrity ===")
    print(f"Total cell labels: {total_cells}")
    print(f"EMPTY={total_empty} ({total_empty / total_cells:.3f})")
    print(f"BLACK={total_black} ({total_black / total_cells:.3f})")
    print(f"WHITE={total_white} ({total_white / total_cells:.3f})")
    print(f"Illegal turn-balance files: {len(illegal_turn_files)}")
    if illegal_turn_files:
        print("  examples:", ", ".join(illegal_turn_files[:10]))
    print(f"Gravity-violation files: {len(gravity_issue_files)}")
    if gravity_issue_files:
        print("  examples:", ", ".join(gravity_issue_files[:10]))

    print("\n=== Preprocessing Parity (train-like vs detect-like) ===")
    print(f"Path mismatches: {parity_mismatch_count}")
    print(f"Mean absolute pixel diff (224x224 BGR): {parity_sum_mae / n:.4f}")
    print(f"Max absolute pixel diff (224x224 BGR): {parity_max_mae:.4f}")

    print("\n=== Prediction Diagnostics ===")
    print(f"Disc bias: {args.disc_bias:.3f}")
    print(f"Raw cell accuracy:  {raw_cell_correct / total_cells:.3f}")
    print(f"Post cell accuracy: {post_cell_correct / total_cells:.3f}")
    print(f"Raw board accuracy:  {raw_board_correct / n:.3f}")
    print(f"Post board accuracy: {post_board_correct / n:.3f}")
    print(f"Samples changed by post-processing: {post_changed_samples}/{n}")
    print(f"Cells changed by post-processing: {post_changed_cells}/{total_cells} ({post_changed_cells / total_cells:.3f})")


if __name__ == "__main__":
    main()
