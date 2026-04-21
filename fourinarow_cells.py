from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from fourinarow_board import corners_to_warp_crop, crop_board

ROWS = 6
COLS = 7
NUM_CLASSES = 3

EMPTY = 0
BLACK = 1
WHITE = 2

CELL_SIZE = 48
CELL_MARGIN_RATIO = 0.12


@dataclass(frozen=True)
class BoardSample:
    image_path: Path
    label_path: Path
    corners_path: Path


class CellClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def list_board_samples(data_dir: str) -> list[BoardSample]:
    data_path = Path(data_dir)
    samples: list[BoardSample] = []
    for image_path in sorted(data_path.glob("frame_*.png")):
        label_path = image_path.with_suffix(".npy")
        if not label_path.exists():
            continue
        corners_path = image_path.with_name(f"{image_path.stem}_corners.npy")
        samples.append(BoardSample(image_path=image_path, label_path=label_path, corners_path=corners_path))
    return samples


def load_label(label_path: Path) -> np.ndarray:
    return np.load(label_path).astype(np.int64).reshape(ROWS, COLS)


def prepare_board_image(
    frame: np.ndarray,
    *,
    corners: np.ndarray | None = None,
    auto_crop_board: bool = False,
) -> tuple[np.ndarray, object | None]:
    if corners is not None and corners.shape == (4, 2):
        return corners_to_warp_crop(frame, corners), None
    if auto_crop_board:
        return crop_board(frame)
    return frame, None


def load_board_image(sample: BoardSample, auto_crop_board: bool = False) -> tuple[np.ndarray, object | None]:
    frame = cv2.imread(str(sample.image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {sample.image_path}")

    corners = None
    if sample.corners_path.exists():
        loaded = np.load(sample.corners_path)
        if loaded.shape == (4, 2):
            corners = loaded

    return prepare_board_image(frame, corners=corners, auto_crop_board=auto_crop_board)


def extract_cell_patch(
    board_bgr: np.ndarray,
    row: int,
    col: int,
    *,
    out_size: int = CELL_SIZE,
    margin_ratio: float = CELL_MARGIN_RATIO,
) -> np.ndarray:
    height, width = board_bgr.shape[:2]
    x0 = int(round(col * width / COLS))
    x1 = int(round((col + 1) * width / COLS))
    y0 = int(round(row * height / ROWS))
    y1 = int(round((row + 1) * height / ROWS))

    margin_x = max(1, int((x1 - x0) * margin_ratio))
    margin_y = max(1, int((y1 - y0) * margin_ratio))
    x0 = min(x0 + margin_x, x1 - 1)
    x1 = max(x0 + 1, x1 - margin_x)
    y0 = min(y0 + margin_y, y1 - 1)
    y1 = max(y0 + 1, y1 - margin_y)

    patch = board_bgr[y0:y1, x0:x1]
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)


def extract_board_cells(
    board_bgr: np.ndarray,
    *,
    out_size: int = CELL_SIZE,
    margin_ratio: float = CELL_MARGIN_RATIO,
) -> np.ndarray:
    patches = [
        extract_cell_patch(board_bgr, row, col, out_size=out_size, margin_ratio=margin_ratio)
        for row in range(ROWS)
        for col in range(COLS)
    ]
    return np.stack(patches, axis=0)


def make_cell_transform(train: bool) -> transforms.Compose:
    transform_steps: list[object] = [
        transforms.ToPILImage(),
        transforms.Resize((CELL_SIZE, CELL_SIZE)),
    ]
    if train:
        transform_steps.extend([
            transforms.ColorJitter(brightness=0.35, contrast=0.45, saturation=0.2, hue=0.04),
            transforms.RandomAffine(degrees=4, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        ])
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transform_steps)


def load_cell_model(path: str, device: str) -> nn.Module:
    model = CellClassifier()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict_board_from_cells(
    model: nn.Module,
    board_bgr: np.ndarray,
    device: str,
    *,
    disc_bias: float = 0.0,
    black_bias: float = 0.0,
    white_bias: float = 0.0,
    margin_ratio: float = CELL_MARGIN_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    patches = extract_board_cells(board_bgr, margin_ratio=margin_ratio)
    transform = make_cell_transform(train=False)
    batch = torch.stack([
        transform(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        for patch in patches
    ], dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)
        if disc_bias != 0.0:
            logits[:, BLACK] += disc_bias
            logits[:, WHITE] += disc_bias
        if black_bias != 0.0:
            logits[:, BLACK] += black_bias
        if white_bias != 0.0:
            logits[:, WHITE] += white_bias
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    board = np.argmax(probs, axis=1).reshape(ROWS, COLS)
    return board, probs.reshape(ROWS, COLS, NUM_CLASSES)