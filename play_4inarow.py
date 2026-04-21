"""
Interactive 4-in-a-row game: human vs robot.

The robot plays BLACK discs, the human plays WHITE.
A cell-model detects the board state from the workarea camera.
Minimax AI decides the robot's moves, then an ACT policy executes them.

Controls (while the camera window is focused):
    s  — start a new game (human goes first with WHITE)
    auto-detect human WHITE move every 2 seconds
    r  — retry robot move after a failed attempt
    q  — quit

Usage:
    uv run python play_4inarow.py
"""

from __future__ import annotations

import logging
import select
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors, make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.utils import init_logging

from fourinarow_ai import (
    EMPTY, BLACK, WHITE, ROWS, COLS,
    best_move, check_winner, drop_disc, find_new_disc, is_draw,
)
from fourinarow_board import corners_to_warp_crop, run_corner_calibration
from fourinarow_cells import load_cell_model, predict_board_from_cells, prepare_board_image
from detect_grid import draw_overlay, print_board, DEFAULT_CALIBRATION_PATH

# ---------------------------------------------------------------------------
# Configuration — adjust these to match your setup
# ---------------------------------------------------------------------------
ROBOT_PORT = "/dev/tty.usbmodem5A680098251"
ROBOT_ID = "my_follower"
DEVICE = "mps"
EPISODE_TIME_S = 30
FPS = 30

SNAPSHOT_LOG_PATH = Path("outputs/robot_home_taps.log")

# Auto-stop when arm returns to start (home) area after leaving it.
# Distances are in joint-position units.
HOME_TARGET_POS = {
    "elbow_flex.pos": 96.571,
    "shoulder_lift.pos": -103.429,
    "shoulder_pan.pos": 25.890,
    "wrist_flex.pos": 72.176,
    "wrist_roll.pos": -93.495,
}
HOME_RETURN_RADIUS = 14.0
HOME_LEAVE_RADIUS = 24.0

CAMERAS = {
    "gripper": OpenCVCameraConfig(index_or_path=1, fps=30, width=640, height=480),
    "workarea": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
}

# Maps column (1-indexed) -> (policy HF repo, training dataset HF repo)
SLOTS: dict[int, tuple[str, str]] = {
    1: ("tilmannb/act_4inarow_slot1", "tilmannb/4inarow_slot1"),
    2: ("tilmannb/act_4inarow_slot2", "tilmannb/4inarow_slot2"),
    3: ("tilmannb/act_4inarow_slot3", "tilmannb/4inarow_slot3"),
    4: ("tilmannb/act_4inarow_slot4", "tilmannb/4inarow_slot4"),
    5: ("tilmannb/act_4inarow_slot5", "tilmannb/4inarow_slot5"),
    6: ("tilmannb/act_4inarow_slot6", "tilmannb/4inarow_slot6"),
    7: ("tilmannb/act_4inarow_slot7", "tilmannb/4inarow_slot7"),
}

# Cell model for board-state detection
CELL_MODEL_PATH = "models/grid_cell_model.pt"
CALIBRATION_PATH = str(DEFAULT_CALIBRATION_PATH)
STABLE_FRAMES = 5          # consecutive identical reads before accepting
DETECT_TIMEOUT_S = 15.0    # seconds to wait for the human's disc to appear
BOARD_POLL_INTERVAL_S = 2.0

# Minimax search depth (higher = stronger but slower)
AI_DEPTH = 5


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy_and_processors(policy_repo: str, dataset_repo: str, device: str):
    """Load a pretrained policy and its pre/post-processors."""
    ds_meta = LeRobotDatasetMetadata(dataset_repo, revision="main")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_repo)
    policy_cfg.pretrained_path = policy_repo
    policy_cfg.device = device

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_repo,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )

    return policy, preprocessor, postprocessor, ds_meta


# ---------------------------------------------------------------------------
# Robot episode execution
# ---------------------------------------------------------------------------

def run_episode(robot, policy, preprocessor, postprocessor, ds_meta, device, duration_s, fps):
    """Execute the policy on the robot for up to `duration_s` seconds.

    Returns when the user presses Enter (stdin), the timeout elapses, or
    Ctrl+C is pressed.
    """
    torch_device = get_safe_torch_device(device)
    _, robot_action_processor, _ = make_default_processors()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t = time.perf_counter()
    step = 0
    print("  Press ENTER (terminal) or SPACE (camera window) to stop the robot.")

    last_obs_time: float | None = None
    last_joint_pos: dict[str, float] | None = None
    home_keys = sorted(HOME_TARGET_POS.keys())
    home_ref = np.array([HOME_TARGET_POS[k] for k in home_keys], dtype=np.float32)
    left_home = False
    stop_reason = "timeout"
    SNAPSHOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    while time.perf_counter() - start_t < duration_s:
        loop_start = time.perf_counter()

        # Non-blocking stdin check — user presses Enter in the terminal
        if select.select([sys.stdin], [], [], 0)[0]:
            typed = sys.stdin.readline().strip().lower()
            if typed == "t":
                # Snapshot command from terminal; continue the episode.
                print("  >>> Snapshot requested (terminal) — recording next state")
            else:
                stop_reason = "user_terminal"
                logging.info(f"User stopped episode at step {step}")
                print(f"  >>> Stopped by user at step {step}")
                break

        # Also check OpenCV window for SPACE or Enter key
        key = cv2.waitKey(1) & 0xFF
        log_snapshot_now = key == ord('t')
        if key in (ord(' '), 13, 10):  # SPACE, Enter
            stop_reason = "user_window"
            logging.info(f"User stopped episode via window key at step {step}")
            print(f"  >>> Stopped by user at step {step}")
            break

        obs = robot.get_observation()

        # Estimate max joint velocity from observation deltas.
        joint_pos = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        now = time.perf_counter()
        dt = 0.0
        max_vel = 0.0
        home_dist = 0.0
        joint_vel: dict[str, float] = {}

        if last_joint_pos is not None and last_obs_time is not None and joint_pos:
            dt = max(now - last_obs_time, 1e-6)
            for k in joint_pos.keys():
                if k in last_joint_pos:
                    joint_vel[k] = abs(joint_pos[k] - last_joint_pos[k]) / dt
            if joint_vel:
                max_vel = max(joint_vel.values())

        # Stop when the arm returns to home area for the first time after leaving it.
        if all(k in joint_pos for k in home_keys):
            current_home_vec = np.array([joint_pos[k] for k in home_keys], dtype=np.float32)
            home_dist = float(np.linalg.norm(current_home_vec - home_ref))
            if not left_home and home_dist >= HOME_LEAVE_RADIUS:
                left_home = True
            elif left_home and home_dist <= HOME_RETURN_RADIUS:
                stop_reason = "auto_home_return"
                logging.info(
                    "Episode auto-stopped on first return to home zone (dist=%.3f, leave>=%.3f, return<=%.3f)",
                    home_dist,
                    HOME_LEAVE_RADIUS,
                    HOME_RETURN_RADIUS,
                )
                print("  >>> Auto-stopped: gripper path returned near home zone")
                break

        if log_snapshot_now:
            line = (
                f"t={now - start_t:.3f}s step={step} home_dist={home_dist:.3f} "
                f"left_home={int(left_home)} max_vel={max_vel:.3f} "
                f"joint_pos=" + ";".join(f"{k}:{v:.3f}" for k, v in sorted(joint_pos.items())) + " "
                f"joint_vel=" + ";".join(f"{k}:{v:.3f}" for k, v in sorted(joint_vel.items()))
            )
            with SNAPSHOT_LOG_PATH.open("a") as f:
                f.write(line + "\n")
            print("  >>> Snapshot logged (press 't' again anytime):")
            print("      " + line)

        last_joint_pos = joint_pos
        last_obs_time = now

        observation_frame = build_dataset_frame(ds_meta.features, obs, prefix=OBS_STR)

        action = predict_action(
            observation=observation_frame,
            policy=policy,
            device=torch_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task="Place disc in slot",
            robot_type=robot.robot_type,
        )

        action_dict = make_robot_action(action, ds_meta.features)
        robot_action_to_send = robot_action_processor((action_dict, obs))

        robot.send_action(robot_action_to_send)
        step += 1

        elapsed = time.perf_counter() - loop_start
        sleep_time = (1 / fps) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    if stop_reason == "timeout":
        logging.info("Episode timed out after %.2fs", duration_s)

    logging.info(f"Episode done: {step} steps in {time.perf_counter() - start_t:.1f}s ({stop_reason})")
    print(f"  Snapshot log path: {SNAPSHOT_LOG_PATH}")


# ---------------------------------------------------------------------------
# Board-state detection helpers
# ---------------------------------------------------------------------------

def capture_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Read a single frame from the camera."""
    ret, frame = cap.read()
    return frame if ret else None


def detect_board(
    cap: cv2.VideoCapture,
    cell_model: torch.nn.Module,
    device: str,
    calibration_corners: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grab one frame, run cell model, return (board, probs, frame)."""
    frame = capture_frame(cap)
    if frame is None:
        raise RuntimeError("Camera read failed")
    board_img, _ = prepare_board_image(frame, corners=calibration_corners)
    board, probs = predict_board_from_cells(cell_model, board_img, device)
    return board, probs, frame


def wait_for_stable_board(
    cap: cv2.VideoCapture,
    cell_model: torch.nn.Module,
    device: str,
    calibration_corners: np.ndarray | None,
    stable_n: int = STABLE_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep reading until we get `stable_n` consecutive identical boards."""
    prev = None
    count = 0
    while True:
        board, probs, frame = detect_board(cap, cell_model, device, calibration_corners)
        if prev is not None and np.array_equal(board, prev):
            count += 1
        else:
            prev = board.copy()
            count = 1
        if count >= stable_n:
            return board, probs, frame


def wait_for_human_move(
    cap: cv2.VideoCapture,
    cell_model: torch.nn.Module,
    device: str,
    calibration_corners: np.ndarray | None,
    old_board: np.ndarray,
    timeout_s: float = DETECT_TIMEOUT_S,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """After user presses SPACE, read camera until we see exactly one new WHITE disc.

    Returns (new_board, (row, col)) or (old_board, None) on timeout.
    """
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        board, _, frame = detect_board(cap, cell_model, device, calibration_corners)

        # Require at least as many filled cells as before
        if np.sum(board != EMPTY) < np.sum(old_board != EMPTY):
            continue

        pos = find_new_disc(old_board, board)
        if pos is not None:
            row, col = pos
            if board[row, col] == WHITE:
                # Confirm with a stable read
                stable_board, _, _ = wait_for_stable_board(
                    cap, cell_model, device, calibration_corners, stable_n=3
                )
                pos2 = find_new_disc(old_board, stable_board)
                if pos2 is not None and stable_board[pos2[0], pos2[1]] == WHITE:
                    return stable_board, pos2
    return old_board, None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_overlay(
    frame: np.ndarray,
    board: np.ndarray,
    probs: np.ndarray,
    calibration_corners: np.ndarray | None,
    status: str = "",
) -> int:
    """Draw the board overlay + a status line and return the pressed key."""
    display = draw_overlay(frame, board, probs, None, calibration_corners)
    if status:
        cv2.putText(
            display, status, (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1,
        )
    cv2.imshow("4-in-a-Row", display)
    return cv2.waitKey(1) & 0xFF


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def main():
    init_logging()

    # ---- Set up robot ----
    robot_cfg = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=CAMERAS,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()

    # ---- Set up cell-model ----
    device = DEVICE
    print(f"Loading cell model from {CELL_MODEL_PATH}...")
    cell_model = load_cell_model(CELL_MODEL_PATH, device)
    print("Cell model loaded.")

    # ---- Set up camera for detection (workarea) ----
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: cannot open workarea camera")
        robot.disconnect()
        return

    # ---- Load board calibration ----
    calibration_corners: np.ndarray | None = None
    cal_path = Path(CALIBRATION_PATH)
    if cal_path.exists():
        calibration_corners = np.load(cal_path)
        print(f"Loaded board calibration from {cal_path}")
    else:
        print("No calibration found — run detect_grid_cells.py --calibrate first.")

    # ---- Policy cache ----
    policy_cache: dict[int, tuple] = {}

    def ensure_policy(slot: int):
        if slot not in policy_cache:
            if slot not in SLOTS:
                print(f"  WARNING: No policy configured for slot {slot}!")
                return None
            repo, ds_repo = SLOTS[slot]
            print(f"  Loading policy for slot {slot} from {repo}...")
            data = load_policy_and_processors(repo, ds_repo, device)
            policy_cache[slot] = data
            print(f"  Policy for slot {slot} ready.")
        return policy_cache[slot]

    # ---- Game state ----
    board = np.zeros((ROWS, COLS), dtype=int)
    game_active = False
    human_turn = False   # True when waiting for the human to play
    retry_slot: int | None = None
    last_poll_t = 0.0
    dummy_probs = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    dummy_probs[:, :, EMPTY] = 1.0
    status_msg = "Press 's' to start a new game"

    print("\n" + "=" * 50)
    print("  4-in-a-Row — Human (WHITE) vs Robot (BLACK)")
    print("=" * 50)
    print("  s     — start new game (human plays first)")
    print("  Auto-check every 2s for a new WHITE disc")
    print("  r     — retry last robot move if placement failed")
    print("  q     — quit")
    print("=" * 50 + "\n")

    try:
        while True:
            # Grab a frame for the live view
            frame = capture_frame(cap)
            if frame is None:
                break

            # Show overlay
            key = show_overlay(frame, board, dummy_probs, calibration_corners, status_msg)

            # ---- Key handling ----
            if key == ord("q"):
                break

            if key == ord("s"):
                # Start a new game
                board = np.zeros((ROWS, COLS), dtype=int)
                dummy_probs = np.zeros((ROWS, COLS, 3), dtype=np.float32)
                dummy_probs[:, :, EMPTY] = 1.0
                game_active = True
                human_turn = True
                retry_slot = None
                last_poll_t = 0.0
                status_msg = "NEW GAME — waiting for human WHITE move (auto every 2s)."
                print("\n>>> New game started. Human (WHITE) goes first.")
                print_board(board)
                continue

            if key == ord("r") and game_active and human_turn and retry_slot is not None:
                policy_data = ensure_policy(retry_slot)
                if policy_data is None:
                    status_msg = f"No policy for slot {retry_slot}! Press 's' to restart."
                    game_active = False
                    continue
                policy, preprocessor, postprocessor, ds_meta = policy_data
                status_msg = f"Retrying robot move in column {retry_slot}..."
                show_overlay(frame, board, dummy_probs, calibration_corners, status_msg)
                print(f"  Retrying slot {retry_slot} policy...")
                try:
                    run_episode(
                        robot, policy, preprocessor, postprocessor,
                        ds_meta, device, EPISODE_TIME_S, FPS,
                    )
                except KeyboardInterrupt:
                    print("\n  Retry aborted.")
                status_msg = (
                    f"Waiting for human WHITE move (auto every {BOARD_POLL_INTERVAL_S:.0f}s). "
                    "Press r if robot move failed."
                )
                continue

            if not (game_active and human_turn):
                continue

            now = time.perf_counter()
            if now - last_poll_t < BOARD_POLL_INTERVAL_S:
                continue
            last_poll_t = now

            # Auto-poll for human move: detect exactly one new WHITE disc.
            detected_board, _, _ = wait_for_stable_board(
                cap,
                cell_model,
                device,
                calibration_corners,
                stable_n=3,
            )
            if np.sum(detected_board != EMPTY) < np.sum(board != EMPTY):
                continue

            pos = find_new_disc(board, detected_board)
            if pos is None:
                status_msg = f"Waiting for human WHITE move (auto every {BOARD_POLL_INTERVAL_S:.0f}s)..."
                continue

            row, col = pos
            if detected_board[row, col] != WHITE:
                status_msg = f"Detected non-WHITE change. Waiting for human move (every {BOARD_POLL_INTERVAL_S:.0f}s)..."
                continue

            board = detected_board.copy()
            dummy_probs = np.zeros((ROWS, COLS, 3), dtype=np.float32)
            for r in range(ROWS):
                for c in range(COLS):
                    dummy_probs[r, c, board[r, c]] = 1.0

            print(f"  Human played column {col + 1} (row {row}).")
            print_board(board)

            winner = check_winner(board)
            if winner == WHITE:
                status_msg = "HUMAN WINS! Press 's' for new game."
                game_active = False
                print(">>> HUMAN WINS! <<<")
                continue
            if is_draw(board):
                status_msg = "DRAW! Press 's' for new game."
                game_active = False
                print(">>> DRAW <<<")
                continue

            # Robot turn: choose move and execute once.
            status_msg = "Robot is thinking..."
            show_overlay(frame, board, dummy_probs, calibration_corners, status_msg)
            ai_col = best_move(board, player=BLACK, depth=AI_DEPTH)
            slot = ai_col + 1
            print(f"  Robot chooses column {slot}.")

            policy_data = ensure_policy(slot)
            if policy_data is None:
                status_msg = f"No policy for slot {slot}! Press 's' to restart."
                game_active = False
                continue

            policy, preprocessor, postprocessor, ds_meta = policy_data
            status_msg = f"Robot placing disc in column {slot}..."
            show_overlay(frame, board, dummy_probs, calibration_corners, status_msg)
            print(f"  Executing slot {slot} policy...")
            try:
                run_episode(
                    robot, policy, preprocessor, postprocessor,
                    ds_meta, device, EPISODE_TIME_S, FPS,
                )
            except KeyboardInterrupt:
                print("\n  Episode aborted.")

            # Assume robot placement succeeded (no BLACK-disk detection).
            new_board = drop_disc(board, ai_col, BLACK)
            if new_board is not None:
                board = new_board

            dummy_probs = np.zeros((ROWS, COLS, 3), dtype=np.float32)
            for r in range(ROWS):
                for c in range(COLS):
                    dummy_probs[r, c, board[r, c]] = 1.0

            print_board(board)

            winner = check_winner(board)
            if winner == BLACK:
                status_msg = "ROBOT WINS! Press 's' for new game."
                game_active = False
                print(">>> ROBOT WINS! <<<")
                continue
            if is_draw(board):
                status_msg = "DRAW! Press 's' for new game."
                game_active = False
                print(">>> DRAW <<<")
                continue

            human_turn = True
            retry_slot = slot
            last_poll_t = time.perf_counter()
            status_msg = (
                f"Waiting for human WHITE move (auto every {BOARD_POLL_INTERVAL_S:.0f}s). "
                "Press r if robot move failed."
            )

    except (KeyboardInterrupt, EOFError):
        print("\nShutting down.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
