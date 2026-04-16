"""
Labeling tool for Connect 4 grid detection.

Shows a live camera feed. Tracks two disc colors: black (robot) and white
(human). Turns auto-alternate starting with black. Press 1-7 to drop a disc
in the current player's color.

Controls:
    1-7  : Drop a disc in that column (current player's color)
    t    : Toggle current player (if you need to fix turn order)
    u    : Undo last move
    c    : Clear the board (start over)
    s    : Save a snapshot without changing the board (e.g. empty board)
    q    : Quit

Board encoding in saved .npy files:
    0 = empty
    1 = black (robot)
    2 = white (human)

Images are saved to data/grid_labels/ as:
    frame_NNNN.png   — the camera frame
    frame_NNNN.npy   — the 6x7 board state

Usage:
    uv run python label_grid.py
"""

import os
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

COLS = 7
ROWS = 6

EMPTY = 0
BLACK = 1  # robot
WHITE = 2  # human

OUTPUT_DIR = "data/grid_labels"

# ---------------------------------------------------------------------------
# Board logic
# ---------------------------------------------------------------------------


def drop_disc(board: np.ndarray, col: int, color: int) -> bool:
    """Drop a disc in the given column (0-indexed). Returns True if successful."""
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == EMPTY:
            board[row, col] = color
            return True
    return False  # column full


def undo_disc(board: np.ndarray, move_history: list[tuple[int, int]]) -> None:
    """Remove the last disc placed."""
    if not move_history:
        return
    col, _color = move_history.pop()
    for row in range(ROWS):
        if board[row, col] != EMPTY:
            board[row, col] = EMPTY
            return


def print_board(board: np.ndarray, current_player: int) -> None:
    symbols = {EMPTY: ".", BLACK: "B", WHITE: "W"}
    player_name = "BLACK (robot)" if current_player == BLACK else "WHITE (human)"
    print(f"  Current player: {player_name}")
    print("  " + " ".join(str(i + 1) for i in range(COLS)))
    print("  " + "-" * (COLS * 2 - 1))
    for row in range(ROWS):
        cells = " ".join(symbols[board[row, col]] for col in range(COLS))
        print(f"  {cells}")
    print()


def draw_overlay(frame: np.ndarray, board: np.ndarray, count: int,
                 current_player: int) -> np.ndarray:
    """Draw a small board overlay on the frame."""
    display = frame.copy()
    cell = 18
    ox, oy = 10, 10

    # Background
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
                cv2.circle(display, (cx, cy), r, (50, 50, 50), -1)      # dark
                cv2.circle(display, (cx, cy), r, (0, 0, 0), 1)
            elif val == WHITE:
                cv2.circle(display, (cx, cy), r, (255, 255, 255), -1)    # white
                cv2.circle(display, (cx, cy), r, (180, 180, 180), 1)
            else:
                cv2.circle(display, (cx, cy), r, (80, 80, 80), 1)

    # Current player indicator
    player_label = "BLACK" if current_player == BLACK else "WHITE"
    player_color = (50, 50, 50) if current_player == BLACK else (255, 255, 255)
    cv2.putText(display, f"Turn: {player_label}", (ox, oy + ROWS * cell + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, player_color, 1)
    cv2.putText(display, f"Saved: {count}", (ox + 90, oy + ROWS * cell + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(display, "1-7:drop t:toggle u:undo c:clear s:snap q:quit",
                (ox, oy + ROWS * cell + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return display


def next_index(output_dir: str) -> int:
    """Find the next available frame index."""
    existing = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(f.replace("frame_", "").replace(".png", "")))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def save_sample(frame: np.ndarray, board: np.ndarray, idx: int, output_dir: str) -> None:
    """Save a frame and its board label."""
    img_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
    label_path = os.path.join(output_dir, f"frame_{idx:04d}.npy")
    cv2.imwrite(img_path, frame)
    np.save(label_path, board.copy())
    black_count = int(np.sum(board == BLACK))
    white_count = int(np.sum(board == WHITE))
    print(f"  Saved {img_path} ({black_count}B + {white_count}W = {black_count + white_count} discs)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    board = np.zeros((ROWS, COLS), dtype=int)
    move_history: list[tuple[int, int]] = []  # (col, color)
    current_player = BLACK
    idx = next_index(OUTPUT_DIR)
    save_count = 0

    print("Grid labeling tool (2-color).")
    print("Turns auto-alternate: BLACK (robot) goes first.")
    print("Press 1-7 to drop, 't' to toggle player, 'u' to undo, 'c' to clear.\n")
    print_board(board, current_player)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = draw_overlay(frame, board, save_count, current_player)
            cv2.imshow("Label Grid", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif ord("1") <= key <= ord("7"):
                col = key - ord("1")
                if drop_disc(board, col, current_player):
                    move_history.append((col, current_player))
                    # Alternate turns
                    current_player = WHITE if current_player == BLACK else BLACK
                    time.sleep(0.05)
                    ret, frame = cap.read()
                    if ret:
                        save_sample(frame, board, idx, OUTPUT_DIR)
                        idx += 1
                        save_count += 1
                    print_board(board, current_player)
                else:
                    print(f"  Column {col + 1} is full!")

            elif key == ord("t"):
                current_player = WHITE if current_player == BLACK else BLACK
                player_name = "BLACK (robot)" if current_player == BLACK else "WHITE (human)"
                print(f"  Toggled to {player_name}")

            elif key == ord("s"):
                save_sample(frame, board, idx, OUTPUT_DIR)
                idx += 1
                save_count += 1

            elif key == ord("u"):
                if move_history:
                    _, last_color = move_history[-1]
                    undo_disc(board, move_history)
                    current_player = last_color  # restore that player's turn
                    print("  Undone.")
                    print_board(board, current_player)
                else:
                    print("  Nothing to undo.")

            elif key == ord("c"):
                board = np.zeros((ROWS, COLS), dtype=int)
                move_history.clear()
                current_player = BLACK
                print("  Board cleared.")
                print_board(board, current_player)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. Saved {save_count} samples to {OUTPUT_DIR}/")
        print(f"Total samples in directory: {idx}")


if __name__ == "__main__":
    main()
