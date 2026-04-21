"""Live Connect-4 detector using a per-cell classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from detect_grid import DEFAULT_CALIBRATION_PATH, draw_overlay, print_board
from fourinarow_board import crop_board, run_corner_calibration
from fourinarow_cells import load_cell_model, predict_board_from_cells, prepare_board_image

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ROWS = 6
COLS = 7
EMPTY = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/grid_cell_model.pt")
    parser.add_argument("--auto-crop-board", action="store_true")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--disc-bias", type=float, default=0.0)
    parser.add_argument("--black-bias", type=float, default=0.0)
    parser.add_argument("--white-bias", type=float, default=0.0)
    parser.add_argument("--cell-margin-ratio", type=float, default=0.12,
                        help="Crop less of each cell when smaller; crop more tightly to center when larger")
    parser.add_argument("--stable-frames", type=int, default=3)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading cell model from {args.model}...")
    model = load_cell_model(args.model, device)
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
    elif calibration_path.exists():
        calibration_corners = np.load(calibration_path)
        print(f"Loaded calibration from {calibration_path}")
    elif args.auto_crop_board:
        print("Board auto-crop: enabled (heuristic)")

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

            board_img, board_bbox = prepare_board_image(
                frame,
                corners=calibration_corners,
                auto_crop_board=args.auto_crop_board,
            )
            board, probs = predict_board_from_cells(
                model,
                board_img,
                device,
                disc_bias=args.disc_bias,
                black_bias=args.black_bias,
                white_bias=args.white_bias,
                margin_ratio=args.cell_margin_ratio,
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
            cv2.imshow("Cell Grid Detector", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()