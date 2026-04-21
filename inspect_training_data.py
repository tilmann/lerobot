"""
Inspect training data: for each labeled frame, show the warped board crop
with disc positions overlaid as coloured circles.

Usage:
    uv run python inspect_training_data.py
    uv run python inspect_training_data.py --data data/grid_labels
    uv run python inspect_training_data.py --out outputs/inspect   # save to files instead
    uv run python inspect_training_data.py --out outputs/inspect --data data/grid_labels
"""

import argparse
import os

import cv2
import numpy as np

from fourinarow_board import BOARD_CROP_H, BOARD_CROP_W, corners_to_warp_crop

ROWS = 6
COLS = 7

# BGR colours for empty / black / white
CELL_COLOURS = {
    0: (100, 100, 100),   # empty  – grey circle outline
    1: (30,  30,  30),    # black  – dark filled circle
    2: (220, 220, 220),   # white  – light filled circle
}
CELL_NAMES = {0: ".", 1: "B", 2: "W"}


def draw_board_overlay(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Draw a 6x7 grid + coloured circles on the warped board crop."""
    h, w = img.shape[:2]
    cell_w = w / COLS
    cell_h = h / ROWS

    vis = img.copy()

    for row in range(ROWS):
        for col in range(COLS):
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)
            radius = int(min(cell_w, cell_h) * 0.35)
            val = int(label[row, col])
            colour = CELL_COLOURS[val]
            if val == 0:
                cv2.circle(vis, (cx, cy), radius, colour, 1)
            else:
                cv2.circle(vis, (cx, cy), radius, colour, -1)
            # grid lines
            gx = int(col * cell_w)
            gy = int(row * cell_h)
            cv2.rectangle(vis, (gx, gy), (gx + int(cell_w), gy + int(cell_h)), (0, 200, 0), 1)

    return vis


def process_sample(img_path: str, label_path: str, corners_path: str | None) -> np.ndarray | None:
    """Return visualisation image or None if data is missing."""
    raw = cv2.imread(img_path)
    if raw is None:
        print(f"WARN: cannot read {img_path}")
        return None

    label = np.load(label_path).astype(np.int64).reshape(ROWS, COLS)

    # Apply warp if corners available, otherwise use raw frame resized
    if corners_path and os.path.exists(corners_path):
        corners = np.load(corners_path)
        if corners.shape == (4, 2):
            board = corners_to_warp_crop(raw, corners, out_w=BOARD_CROP_W, out_h=BOARD_CROP_H)
        else:
            board = cv2.resize(raw, (BOARD_CROP_W, BOARD_CROP_H))
    else:
        board = cv2.resize(raw, (BOARD_CROP_W, BOARD_CROP_H))

    vis = draw_board_overlay(board, label)

    # Add filename + label summary strip at bottom
    name = os.path.basename(img_path)
    board_str = "  ".join(
        "".join(CELL_NAMES[int(label[r, c])] for c in range(COLS))
        for r in range(ROWS)
    )
    non_empty = int(np.sum(label != 0))
    info = f"{name}  discs={non_empty}"
    cv2.putText(vis, info, (4, BOARD_CROP_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/grid_labels")
    parser.add_argument(
        "--out",
        default=None,
        help="If given, save PNG files here instead of interactive display",
    )
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    samples = []
    for f in sorted(os.listdir(args.data)):
        if not f.endswith(".png"):
            continue
        img_path = os.path.join(args.data, f)
        label_path = img_path.replace(".png", ".npy")
        if not os.path.exists(label_path):
            continue
        corners_path = img_path.replace(".png", "_corners.npy")
        samples.append((img_path, label_path, corners_path))

    print(f"Found {len(samples)} samples in {args.data}")
    if not samples:
        return

    if args.out:
        for img_path, label_path, corners_path in samples:
            vis = process_sample(img_path, label_path, corners_path)
            if vis is None:
                continue
            out_path = os.path.join(args.out, os.path.basename(img_path))
            cv2.imwrite(out_path, vis)
        print(f"Saved {len(samples)} images to {args.out}/")
        return

    # Interactive mode: show one at a time, navigate with arrow keys / n / p
    idx = 0
    window = "Training data inspector  [←/→ or n/p: navigate | q: quit]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, BOARD_CROP_W * 3, BOARD_CROP_H * 3)

    while True:
        img_path, label_path, corners_path = samples[idx]
        vis = process_sample(img_path, label_path, corners_path)
        if vis is None:
            idx = (idx + 1) % len(samples)
            continue

        label = np.load(label_path).reshape(ROWS, COLS)
        print(f"\n[{idx+1}/{len(samples)}] {os.path.basename(img_path)}")
        for r in range(ROWS):
            row_str = "  ".join(CELL_NAMES[int(label[r, c])] for c in range(COLS))
            print(f"  row {r}: {row_str}")

        cv2.imshow(window, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), 83, ord("d")):  # right arrow / n / d
            idx = (idx + 1) % len(samples)
        elif key in (ord("p"), 81, ord("a")):  # left arrow / p / a
            idx = (idx - 1) % len(samples)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
