"""
Connect 4 grid detector using a trained CNN model.

Detects three states per cell: empty (0), black/robot (1), white/human (2).
Loads the model from models/grid_model.pt and runs inference on the live
camera feed.

Usage:
    uv run python detect_grid.py
    uv run python detect_grid.py --model models/grid_model.pt

Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

COLS = 7
ROWS = 6
NUM_CLASSES = 3

EMPTY = 0
BLACK = 1  # robot
WHITE = 2  # human

STABLE_FRAMES = 5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

IMG_SIZE = 224


def load_model(path: str, device: str) -> nn.Module:
    """Load the trained grid detection model."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_board(model: nn.Module, frame: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a camera frame.

    Returns:
        board: 6x7 int array (0=empty, 1=black, 2=white), gravity-enforced
        probs: 6x7x3 float array of class probabilities
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = TRANSFORM(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        logits = logits.view(ROWS, COLS, NUM_CLASSES)
        probs = torch.softmax(logits, dim=2).cpu().numpy()

    raw_board = np.argmax(probs, axis=2)  # 0, 1, or 2

    # Enforce gravity per column: count black and white, stack from bottom
    board = np.zeros((ROWS, COLS), dtype=int)
    for col in range(COLS):
        black_count = int(np.sum(raw_board[:, col] == BLACK))
        white_count = int(np.sum(raw_board[:, col] == WHITE))
        # Stack from bottom: first the earlier discs, then later ones
        # We don't know the exact order, so just stack them
        row = ROWS - 1
        for _ in range(black_count):
            if row >= 0:
                board[row, col] = BLACK
                row -= 1
        for _ in range(white_count):
            if row >= 0:
                board[row, col] = WHITE
                row -= 1

    return board, probs


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


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


def draw_overlay(frame: np.ndarray, board: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Draw board state overlay on the camera frame."""
    display = frame.copy()
    cell = 22
    ox, oy = 10, 10

    cv2.rectangle(display, (ox - 2, oy - 2),
                  (ox + COLS * cell + 2, oy + ROWS * cell + 2),
                  (40, 40, 40), -1)

    for row in range(ROWS):
        for col in range(COLS):
            cx = ox + col * cell + cell // 2
            cy = oy + row * cell + cell // 2
            r = cell // 2 - 2
            val = board[row, col]

            if val == BLACK:
                cv2.circle(display, (cx, cy), r, (50, 50, 50), -1)
                cv2.circle(display, (cx, cy), r, (0, 0, 0), 1)
            elif val == WHITE:
                cv2.circle(display, (cx, cy), r, (255, 255, 255), -1)
                cv2.circle(display, (cx, cy), r, (180, 180, 180), 1)
            else:
                cv2.circle(display, (cx, cy), r, (80, 80, 80), 1)

            # Show top class probability
            p = probs[row, col, val]
            cv2.putText(display, f"{p:.1f}", (cx - 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)

    return display


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/grid_model.pt")
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

            board, probs = predict_board(model, frame, device)

            # Debounce
            if candidate_board is not None and np.array_equal(board, candidate_board):
                stable_count += 1
            else:
                candidate_board = board.copy()
                stable_count = 1

            # Only accept changes that add discs (game only goes forward)
            if stable_count >= STABLE_FRAMES:
                total_new = np.sum(board != EMPTY)
                total_old = np.sum(prev_board != EMPTY)
                if not np.array_equal(board, prev_board) and total_new >= total_old:
                    prev_board = board.copy()
                    print("--- Board changed ---")
                    print_board(board)

            display = draw_overlay(frame, board, probs)
            cv2.imshow("Grid Detector", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
Connect 4 grid detector using a trained CNN model.

Loads the model from models/grid_model.pt and runs inference on the live
camera feed to detect the board state.

Usage:
    uv run python detect_grid.py
    uv run python detect_grid.py --model models/grid_model.pt

Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

COLS = 7
ROWS = 6

STABLE_FRAMES = 5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

IMG_SIZE = 224


def load_model(path: str, device: str) -> nn.Module:
    """Load the trained grid detection model."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, ROWS * COLS),
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_board(model: nn.Module, frame: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a camera frame.

    Returns:
        board: 6x7 int array (0=empty, 1=filled), gravity-enforced
        probs: 6x7 float array of probabilities
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = TRANSFORM(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(ROWS, COLS)

    raw_board = (probs > 0.5).astype(int)

    # Enforce gravity
    board = np.zeros((ROWS, COLS), dtype=int)
    for col in range(COLS):
        filled_count = int(np.sum(raw_board[:, col]))
        for row in range(ROWS - 1, ROWS - 1 - filled_count, -1):
            if row >= 0:
                board[row, col] = 1

    return board, probs


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_board(board: np.ndarray) -> None:
    symbols = {0: ".", 1: "X"}
    print("\n  " + " ".join(str(i + 1) for i in range(COLS)))
    print("  " + "-" * (COLS * 2 - 1))
    for row in range(ROWS):
        cells = " ".join(symbols[board[row, col]] for col in range(COLS))
        print(f"  {cells}")
    print()


def draw_overlay(frame: np.ndarray, board: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Draw board state overlay on the camera frame."""
    display = frame.copy()
    cell = 22
    ox, oy = 10, 10

    cv2.rectangle(display, (ox - 2, oy - 2),
                  (ox + COLS * cell + 2, oy + ROWS * cell + 2),
                  (0, 0, 0), -1)

    for row in range(ROWS):
        for col in range(COLS):
            cx = ox + col * cell + cell // 2
            cy = oy + row * cell + cell // 2
            p = probs[row, col]

            if board[row, col] == 1:
                cv2.circle(display, (cx, cy), cell // 2 - 2, (0, 0, 255), -1)
            else:
                cv2.circle(display, (cx, cy), cell // 2 - 2, (80, 80, 80), 1)

            # Show probability
            cv2.putText(display, f"{p:.1f}", (cx - 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)

    return display


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/grid_model.pt")
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

            board, probs = predict_board(model, frame, device)

            # Debounce
            if candidate_board is not None and np.array_equal(board, candidate_board):
                stable_count += 1
            else:
                candidate_board = board.copy()
                stable_count = 1

            if stable_count >= STABLE_FRAMES:
                if not np.array_equal(board, prev_board) and np.sum(board) >= np.sum(prev_board):
                    prev_board = board.copy()
                    disc_count = int(np.sum(board))
                    print(f"--- Board changed ({disc_count} disc{'s' if disc_count != 1 else ''}) ---")
                    print_board(board)

            display = draw_overlay(frame, board, probs)
            cv2.imshow("Grid Detector", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
Connect 4 grid detector using the workarea camera.

Automatically finds the grid (largest dark rectangle) and detects discs by
finding the 7 bright vertical slots (gaps between bars) and checking which
row positions are blocked.

Usage:
    uv run python detect_grid.py

Press 'q' in the camera window to quit.
Press 'r' to re-detect the grid rectangle.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Grid dimensions
COLS = 7
ROWS = 6

# Size of the warped (rectified) grid image
WARP_WIDTH = 350
WARP_HEIGHT = 300

# How many consecutive stable frames before accepting a board change.
STABLE_FRAMES = 10

# Minimum fraction of image area that the grid contour must occupy.
MIN_GRID_AREA_RATIO = 0.05

# ---------------------------------------------------------------------------
# Automatic grid detection
# ---------------------------------------------------------------------------


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    # Sort by y first (top vs bottom), then by x (left vs right)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest x+y
    rect[1] = pts[np.argmin(d)]   # top-right: smallest x-y
    rect[3] = pts[np.argmax(d)]   # bottom-left: largest x-y

    return rect


def find_grid_contour(frame: np.ndarray) -> np.ndarray | None:
    """
    Find the 4 corners of the grid in the raw camera frame.

    The grid is a dark (black) 3D-printed object against a lighter background.
    We threshold for dark pixels, find the largest quadrilateral contour.

    Returns a (4, 2) float32 array of ordered corners, or None if not found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold: grid is dark, so we look for dark regions
    # Use adaptive threshold for robustness to lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    # Morphological close to fill gaps in the grid structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    img_area = frame.shape[0] * frame.shape[1]
    min_area = img_area * MIN_GRID_AREA_RATIO

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break

        # Approximate to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # We want a quadrilateral
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return order_corners(corners)

    # Fallback: use the minimum-area rotated rectangle of the largest contour
    if contours and cv2.contourArea(contours[0]) >= min_area:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect).astype(np.float32)
        return order_corners(box)

    return None


def get_warp_matrix(corners: np.ndarray) -> np.ndarray:
    """Compute perspective transform from detected corners to a flat rectangle."""
    dst = np.array([
        [0, 0],
        [WARP_WIDTH, 0],
        [WARP_WIDTH, WARP_HEIGHT],
        [0, WARP_HEIGHT],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, dst)


def warp_frame(frame: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply perspective warp to extract the rectified grid."""
    return cv2.warpPerspective(frame, M, (WARP_WIDTH, WARP_HEIGHT))


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------


def find_slot_columns(gray: np.ndarray) -> list[int]:
    """
    Find the X centers of the 7 bright vertical slots by looking at the
    vertical brightness profile (mean brightness per column).
    Returns a sorted list of 7 x-coordinates.
    """
    col_profile = np.mean(gray, axis=0)

    # Smooth to reduce noise
    kernel = np.ones(5) / 5
    col_profile = np.convolve(col_profile, kernel, mode="same")

    # Find peaks: local maxima above a minimum brightness
    min_bright = np.median(col_profile) + 10
    peaks = []
    for x in range(2, len(col_profile) - 2):
        if (col_profile[x] > col_profile[x - 1]
                and col_profile[x] > col_profile[x + 1]
                and col_profile[x] > min_bright):
            peaks.append((x, col_profile[x]))

    # Merge peaks that are too close (within 15px = same slot)
    merged = []
    for x, val in sorted(peaks):
        if merged and x - merged[-1][0] < 15:
            if val > merged[-1][1]:
                merged[-1] = (x, val)
        else:
            merged.append((x, val))

    # Take the 7 brightest if we found more
    if len(merged) > COLS:
        merged.sort(key=lambda p: p[1], reverse=True)
        merged = merged[:COLS]

    slot_xs = sorted(p[0] for p in merged)
    return slot_xs


def measure_brightness(gray: np.ndarray, slot_xs: list[int]) -> np.ndarray:
    """
    Measure mean brightness at each slot/row position.
    Returns a 6x7 float array.
    """
    cell_h = WARP_HEIGHT / ROWS
    slot_half_w = 8

    brightness = np.zeros((ROWS, COLS), dtype=float)

    for col_idx, sx in enumerate(slot_xs):
        x1 = max(0, sx - slot_half_w)
        x2 = min(WARP_WIDTH, sx + slot_half_w)

        for row in range(ROWS):
            y1 = int(row * cell_h + cell_h * 0.1)
            y2 = int((row + 1) * cell_h - cell_h * 0.1)
            cell = gray[y1:y2, x1:x2]
            if cell.size > 0:
                brightness[row, col_idx] = np.mean(cell)

    return brightness


def calibrate_threshold(gray: np.ndarray, slot_xs: list[int]) -> float:
    """
    Measure brightness on an empty board and set the threshold at 50%
    of the mean empty-cell brightness. This way the threshold adapts
    to actual lighting conditions.
    """
    brightness = measure_brightness(gray, slot_xs)
    mean_empty = np.mean(brightness)
    threshold = mean_empty * 0.5
    print(f"Calibrated threshold: {threshold:.0f}  (mean empty brightness: {mean_empty:.0f})")
    return threshold


def detect_board(gray: np.ndarray, slot_xs: list[int], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the board state by checking brightness in each slot at each row.

    Returns:
        board: 6x7 int array (0=empty, 1=filled), gravity-enforced
        brightness: 6x7 float array of mean brightness per cell (for debug)
    """
    brightness = measure_brightness(gray, slot_xs)

    # Empty = bright (background visible), filled = dark (disc blocks light)
    raw_board = (brightness < threshold).astype(int)

    # Enforce gravity: stack filled cells from bottom
    board = np.zeros((ROWS, COLS), dtype=int)
    for col in range(COLS):
        filled_count = int(np.sum(raw_board[:, col]))
        for row in range(ROWS - 1, ROWS - 1 - filled_count, -1):
            if row >= 0:
                board[row, col] = 1

    return board, brightness


def print_board(board: np.ndarray) -> None:
    """Pretty-print the board state."""
    symbols = {0: ".", 1: "X"}
    print("\n  " + " ".join(str(i + 1) for i in range(COLS)))
    print("  " + "-" * (COLS * 2 - 1))
    for row in range(ROWS):
        cells = " ".join(symbols[board[row, col]] for col in range(COLS))
        print(f"  {cells}")
    print()


def draw_debug(warped: np.ndarray, board: np.ndarray, slot_xs: list[int],
               brightness: np.ndarray) -> np.ndarray:
    """Draw slot lines, detection results, and brightness values."""
    debug = warped.copy()
    cell_h = WARP_HEIGHT / ROWS

    for sx in slot_xs:
        cv2.line(debug, (sx, 0), (sx, WARP_HEIGHT), (255, 0, 0), 1)

    for row in range(ROWS):
        for col_idx, sx in enumerate(slot_xs):
            cy = int(row * cell_h + cell_h / 2)
            radius = int(cell_h * 0.3)

            if board[row, col_idx] == 1:
                cv2.circle(debug, (sx, cy), radius, (0, 0, 255), -1)
            else:
                cv2.circle(debug, (sx, cy), radius, (0, 255, 0), 1)

            bv = int(brightness[row, col_idx])
            cv2.putText(debug, str(bv), (sx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    return debug


def draw_detection_overlay(frame: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    """Draw the detected grid outline on the raw camera frame."""
    display = frame.copy()
    if corners is not None:
        pts = corners.astype(int).reshape(-1, 1, 2)
        cv2.polylines(display, [pts], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(corners.astype(int)):
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
    return display


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    print("Grid detector running. Looking for grid...\n")

    # Step 1: Auto-detect grid rectangle
    corners = None
    print("Detecting grid rectangle...")
    for attempt in range(60):  # try for up to ~2 seconds
        ret, frame = cap.read()
        if not ret:
            continue

        corners = find_grid_contour(frame)
        overlay = draw_detection_overlay(frame, corners)
        cv2.imshow("Grid Detector", overlay)
        cv2.waitKey(1)

        if corners is not None:
            print(f"Grid found at corners: {corners.astype(int).tolist()}")
            break

    if corners is None:
        print("Error: Could not find grid rectangle.")
        print("Make sure the dark grid is visible against a lighter background.")
        cap.release()
        cv2.destroyAllWindows()
        return

    M = get_warp_matrix(corners)

    # Step 2: Detect slot positions from a few frames
    print("Detecting slot positions...")
    slot_xs = None
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            continue
        warped = warp_frame(frame, M)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        detected = find_slot_columns(gray)
        if len(detected) == COLS:
            slot_xs = detected
            break

    if slot_xs is None or len(slot_xs) != COLS:
        print(f"Error: Found {len(slot_xs) if slot_xs else 0} slots, expected {COLS}.")
        print("Try adjusting lighting or grid position.")
        cap.release()
        cv2.destroyAllWindows()
        return

    print(f"Found {COLS} slots at x={slot_xs}\n")

    prev_board = np.zeros((ROWS, COLS), dtype=int)
    candidate_board = None
    stable_count = 0

    print("Monitoring for changes... Press 'q' to quit, 'r' to re-detect grid.\n")
    print_board(prev_board)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            warped = warp_frame(frame, M)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            board, brightness = detect_board(gray, slot_xs)

            # Debounce
            if candidate_board is not None and np.array_equal(board, candidate_board):
                stable_count += 1
            else:
                candidate_board = board.copy()
                stable_count = 1

            # Only accept changes that add discs (game only goes forward)
            if stable_count >= STABLE_FRAMES:
                if not np.array_equal(board, prev_board) and np.sum(board) >= np.sum(prev_board):
                    prev_board = board.copy()
                    disc_count = int(np.sum(board))
                    print(f"--- Board changed ({disc_count} disc{'s' if disc_count != 1 else ''}) ---")
                    print_board(board)

            debug = draw_debug(warped, board, slot_xs, brightness)
            cv2.imshow("Grid Detector", debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Re-detect grid
                print("\nRe-detecting grid...")
                new_corners = find_grid_contour(frame)
                if new_corners is not None:
                    corners = new_corners
                    M = get_warp_matrix(corners)
                    # Re-detect slots
                    warped = warp_frame(frame, M)
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    detected = find_slot_columns(gray)
                    if len(detected) == COLS:
                        slot_xs = detected
                        threshold = calibrate_threshold(gray, slot_xs)
                        print(f"Grid re-detected. Slots at x={slot_xs}")
                    else:
                        print(f"Grid found but only {len(detected)} slots detected.")
                else:
                    print("Could not re-detect grid.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
Connect 4 grid detector using the workarea camera.

Detects discs in a grid by finding the 7 bright vertical slots (gaps between
bars) and checking which row positions are blocked. Works with any disc color
(black, white, etc.) since it only looks for brightness changes in the slots.

On startup:
  1. Click the 4 corners of the grid (TL, TR, BR, BL) to correct perspective.
  2. Slot positions are auto-detected from the vertical brightness profile.
  3. Detection runs — prints the board whenever it changes.

Usage:
    uv run python detect_grid.py

Press 'q' in the camera window to quit.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Grid dimensions
COLS = 7
ROWS = 6

# Size of the warped (rectified) grid image
WARP_WIDTH = 350
WARP_HEIGHT = 300

# A slot cell is "empty" if its mean brightness is above this threshold.
# The background showing through empty slots is bright; discs block the light.
EMPTY_BRIGHTNESS = 120

# How many consecutive stable frames before accepting a board change.
STABLE_FRAMES = 10

# ---------------------------------------------------------------------------
# Perspective calibration
# ---------------------------------------------------------------------------


def pick_corners(cap) -> np.ndarray:
    """
    Let the user click 4 corners of the grid.
    Order: top-left, top-right, bottom-right, bottom-left.
    Returns a (4, 2) float32 array.
    """
    corners = []
    labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))

    cv2.namedWindow("Grid Detector")
    cv2.setMouseCallback("Grid Detector", on_click)

    print("Click the 4 corners of the grid in order:")
    print("  1) Top-left  2) Top-right  3) Bottom-right  4) Bottom-left")

    while len(corners) < 4:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()

        idx = len(corners)
        cv2.putText(display, f"Click: {labels[idx]} ({idx+1}/4)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for i, (cx, cy) in enumerate(corners):
            cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(display, labels[i], (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                cv2.line(display, corners[i], corners[i + 1], (0, 0, 255), 2)
            if len(corners) == 4:
                cv2.line(display, corners[3], corners[0], (0, 0, 255), 2)

        cv2.imshow("Grid Detector", display)
        cv2.waitKey(1)

    cv2.setMouseCallback("Grid Detector", lambda *a: None)
    print(f"Corners selected: {corners}\n")
    return np.array(corners, dtype=np.float32)


def get_warp_matrix(corners: np.ndarray) -> np.ndarray:
    """Compute perspective transform from clicked corners to a flat rectangle."""
    dst = np.array([
        [0, 0],
        [WARP_WIDTH, 0],
        [WARP_WIDTH, WARP_HEIGHT],
        [0, WARP_HEIGHT],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, dst)


def warp_frame(frame: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply perspective warp to extract the rectified grid."""
    return cv2.warpPerspective(frame, M, (WARP_WIDTH, WARP_HEIGHT))


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------


def find_slot_columns(gray: np.ndarray) -> list[int]:
    """
    Find the X centers of the 7 bright vertical slots by looking at the
    vertical brightness profile (mean brightness per column).
    Returns a sorted list of 7 x-coordinates.
    """
    # Average brightness per pixel column across all rows
    col_profile = np.mean(gray, axis=0)

    # Smooth to reduce noise
    kernel = np.ones(5) / 5
    col_profile = np.convolve(col_profile, kernel, mode="same")

    # Find peaks: local maxima above a minimum brightness
    min_bright = np.median(col_profile) + 10
    peaks = []
    for x in range(2, len(col_profile) - 2):
        if (col_profile[x] > col_profile[x - 1]
                and col_profile[x] > col_profile[x + 1]
                and col_profile[x] > min_bright):
            peaks.append((x, col_profile[x]))

    # Merge peaks that are too close (within 15px = same slot)
    merged = []
    for x, val in sorted(peaks):
        if merged and x - merged[-1][0] < 15:
            if val > merged[-1][1]:
                merged[-1] = (x, val)
        else:
            merged.append((x, val))

    # Take the 7 brightest if we found more
    if len(merged) > COLS:
        merged.sort(key=lambda p: p[1], reverse=True)
        merged = merged[:COLS]

    slot_xs = sorted(p[0] for p in merged)
    return slot_xs


def detect_board(gray: np.ndarray, slot_xs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the board state by checking brightness in each slot at each row.

    Returns:
        board: 6x7 int array (0=empty, 1=filled), gravity-enforced
        brightness: 6x7 float array of mean brightness per cell (for debug)
    """
    cell_h = WARP_HEIGHT / ROWS
    slot_half_w = 8  # pixels to sample on each side of the slot center

    brightness = np.zeros((ROWS, COLS), dtype=float)

    for col_idx, sx in enumerate(slot_xs):
        x1 = max(0, sx - slot_half_w)
        x2 = min(WARP_WIDTH, sx + slot_half_w)

        for row in range(ROWS):
            y1 = int(row * cell_h + cell_h * 0.1)
            y2 = int((row + 1) * cell_h - cell_h * 0.1)
            cell = gray[y1:y2, x1:x2]
            if cell.size > 0:
                brightness[row, col_idx] = np.mean(cell)

    # Empty = bright (background visible), filled = dark (disc blocks light)
    raw_board = (brightness < EMPTY_BRIGHTNESS).astype(int)

    # Enforce gravity: stack filled cells from bottom
    board = np.zeros((ROWS, COLS), dtype=int)
    for col in range(COLS):
        filled_count = int(np.sum(raw_board[:, col]))
        for row in range(ROWS - 1, ROWS - 1 - filled_count, -1):
            if row >= 0:
                board[row, col] = 1

    return board, brightness


def print_board(board: np.ndarray) -> None:
    """Pretty-print the board state."""
    symbols = {0: ".", 1: "X"}
    print("\n  " + " ".join(str(i + 1) for i in range(COLS)))
    print("  " + "-" * (COLS * 2 - 1))
    for row in range(ROWS):
        cells = " ".join(symbols[board[row, col]] for col in range(COLS))
        print(f"  {cells}")
    print()


def draw_debug(warped: np.ndarray, board: np.ndarray, slot_xs: list[int],
               brightness: np.ndarray) -> np.ndarray:
    """Draw slot lines, detection results, and brightness values."""
    debug = warped.copy()
    cell_h = WARP_HEIGHT / ROWS

    # Draw vertical lines at detected slot centers
    for sx in slot_xs:
        cv2.line(debug, (sx, 0), (sx, WARP_HEIGHT), (255, 0, 0), 1)

    for row in range(ROWS):
        for col_idx, sx in enumerate(slot_xs):
            cy = int(row * cell_h + cell_h / 2)
            radius = int(cell_h * 0.3)

            if board[row, col_idx] == 1:
                cv2.circle(debug, (sx, cy), radius, (0, 0, 255), -1)
            else:
                cv2.circle(debug, (sx, cy), radius, (0, 255, 0), 1)

            # Show brightness value
            bv = int(brightness[row, col_idx])
            cv2.putText(debug, str(bv), (sx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    return debug


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    print("Grid detector running.\n")

    # Step 1: Click the 4 corners of the grid
    corners = pick_corners(cap)
    M = get_warp_matrix(corners)

    # Step 2: Detect slot positions from a few frames
    print("Detecting slot positions...")
    slot_xs = None
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            continue
        warped = warp_frame(frame, M)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        detected = find_slot_columns(gray)
        if len(detected) == COLS:
            slot_xs = detected
            break

    if slot_xs is None or len(slot_xs) != COLS:
        print(f"Error: Found {len(slot_xs) if slot_xs else 0} slots, expected {COLS}.")
        print("Try adjusting the corner selection or lighting.")
        cap.release()
        return

    print(f"Found {COLS} slots at x={slot_xs}\n")

    prev_board = np.zeros((ROWS, COLS), dtype=int)
    candidate_board = None
    stable_count = 0

    print("Monitoring for changes... Press 'q' to quit.\n")
    print_board(prev_board)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            warped = warp_frame(frame, M)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            board, brightness = detect_board(gray, slot_xs)

            # Debounce
            if candidate_board is not None and np.array_equal(board, candidate_board):
                stable_count += 1
            else:
                candidate_board = board.copy()
                stable_count = 1

            # Only accept changes that add discs (game only goes forward)
            if stable_count >= STABLE_FRAMES:
                if not np.array_equal(board, prev_board) and np.sum(board) >= np.sum(prev_board):
                    prev_board = board.copy()
                    disc_count = int(np.sum(board))
                    print(f"--- Board changed ({disc_count} disc{'s' if disc_count != 1 else ''}) ---")
                    print_board(board)

            debug = draw_debug(warped, board, slot_xs, brightness)
            cv2.imshow("Grid Detector", debug)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
