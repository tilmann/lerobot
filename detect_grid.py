"""Connect 4 grid detector using a trained CNN model."""

import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from pathlib import Path

from fourinarow_board import BoardBBox, corners_to_warp_crop, crop_board, draw_board_bbox, run_corner_calibration

DEFAULT_CALIBRATION_PATH = Path("models/board_calibration.npy")

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

COLS = 7
ROWS = 6
NUM_CLASSES = 3

EMPTY = 0
BLACK = 1
WHITE = 2

STABLE_FRAMES = 5
IMG_SIZE = 224


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


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_board(
    model: nn.Module,
    frame: np.ndarray,
    device: str,
    *,
    auto_crop_board: bool = False,
    calibration_corners: np.ndarray | None = None,
    enforce_gravity: bool = False,
    disc_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, BoardBBox | None]:
    working_frame = frame
    board_bbox = None
    if calibration_corners is not None:
        working_frame = corners_to_warp_crop(frame, calibration_corners)
    elif auto_crop_board:
        working_frame, board_bbox = crop_board(frame)

    rgb = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
    img = TRANSFORM(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        logits = logits.view(ROWS, COLS, NUM_CLASSES)
        if disc_bias != 0.0:
            logits[:, :, BLACK] += disc_bias
            logits[:, :, WHITE] += disc_bias
        probs = torch.softmax(logits, dim=2).cpu().numpy()

    raw_board = np.argmax(probs, axis=2)
    board = raw_board.copy()
    if enforce_gravity:
        board.fill(EMPTY)
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

    return board, probs, board_bbox


def print_board(board: np.ndarray) -> None:
    symbols = {EMPTY: ".", BLACK: "B", WHITE: "W"}
    print("\n  " + " ".join(str(i + 1) for i in range(COLS)))
    print("  " + "-" * (COLS * 2 - 1))
    for row in range(ROWS):
        cells = " ".join(symbols[board[row, col]] for col in range(COLS))
        print(f"  {cells}")
    black_total = int(np.sum(board == BLACK))
    white_total = int(np.sum(board == WHITE))
    print(f"  ({black_total}B + {white_total}W)")
    print()


def draw_calibration_quad(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    display = frame.copy()
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
    for idx, (x, y) in enumerate(corners):
        cv2.circle(display, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(display, str(idx + 1), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return display


def draw_overlay(
    frame: np.ndarray,
    board: np.ndarray,
    probs: np.ndarray,
    board_bbox: BoardBBox | None,
    calibration_corners: np.ndarray | None = None,
) -> np.ndarray:
    if calibration_corners is not None:
        display = draw_calibration_quad(frame, calibration_corners)
    else:
        display = draw_board_bbox(frame, board_bbox)
    cell = 22
    ox, oy = 10, 10

    cv2.rectangle(
        display,
        (ox - 2, oy - 2),
        (ox + COLS * cell + 2, oy + ROWS * cell + 2),
        (40, 40, 40),
        -1,
    )

    for row in range(ROWS):
        for col in range(COLS):
            cx = ox + col * cell + cell // 2
            cy = oy + row * cell + cell // 2
            radius = cell // 2 - 2
            value = board[row, col]

            if value == BLACK:
                cv2.circle(display, (cx, cy), radius, (50, 50, 50), -1)
                cv2.circle(display, (cx, cy), radius, (0, 0, 0), 1)
            elif value == WHITE:
                cv2.circle(display, (cx, cy), radius, (255, 255, 255), -1)
                cv2.circle(display, (cx, cy), radius, (180, 180, 180), 1)
            else:
                cv2.circle(display, (cx, cy), radius, (80, 80, 80), 1)

            probability = probs[row, col, value]
            cv2.putText(
                display,
                f"{probability:.1f}",
                (cx - 8, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 255, 255),
                1,
            )

    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/grid_model.pt")
    parser.add_argument("--auto-crop-board", action="store_true")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH),
                        help="Path to board calibration .npy file")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run corner calibration on startup and save to --calibration path")
    parser.add_argument("--enforce-gravity", action="store_true",
                        help="Post-process predictions to enforce per-column gravity")
    parser.add_argument("--disc-bias", type=float, default=0.0,
                        help="Positive values make black/white easier to predict than empty")
    parser.add_argument("--stable-frames", type=int, default=STABLE_FRAMES,
                        help="Number of consecutive matching frames required before updating the board")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    print("Model loaded.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    calibration_corners: np.ndarray | None = None
    calibration_path = Path(args.calibration)

    if args.calibrate:
        print("Calibration mode: click the four board corners (UL, UR, LR, LL) then press c.")
        ret, frame = cap.read()
        if ret:
            calibration_corners = run_corner_calibration(frame)
            if calibration_corners is not None:
                calibration_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(calibration_path, calibration_corners)
                print(f"Calibration saved to {calibration_path}")
            else:
                print("Calibration cancelled.")
    elif calibration_path.exists():
        calibration_corners = np.load(calibration_path)
        print(f"Loaded calibration from {calibration_path}")
    elif args.auto_crop_board:
        print("Board auto-crop: enabled (heuristic)")

    if calibration_corners is not None:
        print(f"Using corner-based board crop. Run with --calibrate to redo.")

    prev_board = np.zeros((ROWS, COLS), dtype=int)
    candidate_board = None
    stable_count = 0

    print("Monitoring... Press 'q' to quit.\n")
    print_board(prev_board)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            board, probs, board_bbox = predict_board(
                model,
                frame,
                device,
                auto_crop_board=args.auto_crop_board,
                calibration_corners=calibration_corners,
                enforce_gravity=args.enforce_gravity,
                disc_bias=args.disc_bias,
            )

            if candidate_board is not None and np.array_equal(board, candidate_board):
                stable_count += 1
            else:
                candidate_board = board.copy()
                stable_count = 1

            if stable_count >= args.stable_frames:
                total_new = np.sum(board != EMPTY)
                total_old = np.sum(prev_board != EMPTY)
                if not np.array_equal(board, prev_board) and total_new >= total_old:
                    prev_board = board.copy()
                    print("--- Board changed ---")
                    print_board(board)

            display = draw_overlay(frame, board, probs, board_bbox, calibration_corners)
            cv2.imshow("Grid Detector", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
