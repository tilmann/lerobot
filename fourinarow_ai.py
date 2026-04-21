"""Minimax / alpha-beta Connect 4 AI.

Board is represented as a 6×7 numpy array (row 0 = top):
    0 = empty, 1 = BLACK (robot), 2 = WHITE (human)

Usage:
    from fourinarow_ai import best_move, check_winner, ROWS, COLS, EMPTY, BLACK, WHITE

    col = best_move(board, player=BLACK, depth=8)
"""

from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
BLACK = 1   # robot
WHITE = 2   # human

WIN_LENGTH = 4

# Large scores for terminal states
_INF = 1_000_000


def _valid_columns(board: np.ndarray) -> list[int]:
    """Return columns that still have room (top row is empty)."""
    return [c for c in range(COLS) if board[0, c] == EMPTY]


def drop_disc(board: np.ndarray, col: int, player: int) -> np.ndarray | None:
    """Return a new board with `player` disc dropped in `col`, or None if full."""
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == EMPTY:
            new = board.copy()
            new[row, col] = player
            return new
    return None


def check_winner(board: np.ndarray) -> int:
    """Return BLACK, WHITE, or EMPTY (no winner yet)."""
    for player in (BLACK, WHITE):
        # horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(board[r, c + i] == player for i in range(4)):
                    return player
        # vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(board[r + i, c] == player for i in range(4)):
                    return player
        # diagonal ↘
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(board[r + i, c + i] == player for i in range(4)):
                    return player
        # diagonal ↗
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(board[r - i, c + i] == player for i in range(4)):
                    return player
    return EMPTY


def is_draw(board: np.ndarray) -> bool:
    return int(np.sum(board == EMPTY)) == 0


def _opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


# ---- Heuristic scoring ----

def _count_windows(board: np.ndarray, player: int, length: int) -> int:
    """Count the number of `length`-in-a-row windows for `player`."""
    count = 0
    opp = _opponent(player)
    for r in range(ROWS):
        for c in range(COLS - 3):
            window = board[r, c:c + 4]
            if np.sum(window == player) == length and np.sum(window == opp) == 0:
                count += 1
    for r in range(ROWS - 3):
        for c in range(COLS):
            window = board[r:r + 4, c]
            if np.sum(window == player) == length and np.sum(window == opp) == 0:
                count += 1
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = np.array([board[r + i, c + i] for i in range(4)])
            if np.sum(window == player) == length and np.sum(window == opp) == 0:
                count += 1
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = np.array([board[r - i, c + i] for i in range(4)])
            if np.sum(window == player) == length and np.sum(window == opp) == 0:
                count += 1
    return count


def _score_position(board: np.ndarray, player: int) -> float:
    """Heuristic board evaluation from `player`'s perspective."""
    opp = _opponent(player)
    score = 0.0

    # Center column preference
    center_col = board[:, COLS // 2]
    score += 3.0 * int(np.sum(center_col == player))

    score += 100.0 * _count_windows(board, player, 4)
    score += 5.0 * _count_windows(board, player, 3)
    score += 2.0 * _count_windows(board, player, 2)

    score -= 100.0 * _count_windows(board, opp, 4)
    score -= 4.0 * _count_windows(board, opp, 3)

    return score


# ---- Minimax with alpha-beta pruning ----

def _minimax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    player: int,
) -> tuple[float, int | None]:
    """Return (score, best_col). `player` is the maximizing player."""
    opp = _opponent(player)
    winner = check_winner(board)
    if winner == player:
        return _INF + depth, None
    if winner == opp:
        return -_INF - depth, None
    if is_draw(board):
        return 0.0, None

    valid = _valid_columns(board)
    if depth == 0:
        return _score_position(board, player), None

    # Search center columns first for better pruning
    valid.sort(key=lambda c: abs(c - COLS // 2))

    if maximizing:
        best_score = -float("inf")
        best_col = valid[0]
        for col in valid:
            new_board = drop_disc(board, col, player)
            if new_board is None:
                continue
            score, _ = _minimax(new_board, depth - 1, alpha, beta, False, player)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_col
    else:
        best_score = float("inf")
        best_col = valid[0]
        for col in valid:
            new_board = drop_disc(board, col, opp)
            if new_board is None:
                continue
            score, _ = _minimax(new_board, depth - 1, alpha, beta, True, player)
            if score < best_score:
                best_score = score
                best_col = col
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best_score, best_col


def best_move(board: np.ndarray, player: int = BLACK, depth: int = 8) -> int:
    """Return the best column for `player` to play, using minimax search."""
    valid = _valid_columns(board)
    if not valid:
        raise ValueError("No valid moves available.")

    # Check for immediate winning moves first
    for col in valid:
        new_board = drop_disc(board, col, player)
        if new_board is not None and check_winner(new_board) == player:
            return col

    # Check for immediate blocking moves
    opp = _opponent(player)
    for col in valid:
        new_board = drop_disc(board, col, opp)
        if new_board is not None and check_winner(new_board) == opp:
            return col

    _, col = _minimax(board, depth, -float("inf"), float("inf"), True, player)
    return col if col is not None else valid[0]


def find_new_disc(old_board: np.ndarray, new_board: np.ndarray) -> tuple[int, int] | None:
    """Find the single cell that changed from empty to a disc.

    Returns (row, col) or None if no single new disc found.
    """
    diff = (old_board == EMPTY) & (new_board != EMPTY)
    positions = np.argwhere(diff)
    if len(positions) == 1:
        return int(positions[0, 0]), int(positions[0, 1])
    return None
