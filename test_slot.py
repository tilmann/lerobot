"""
Test a specific policy slot: loads the policy and runs it until the robot returns home.

Usage:
    uv run python test_slot.py
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROBOT_PORT = "/dev/tty.usbmodem5A680098251"
ROBOT_ID = "my_follower"
DEVICE = "mps"
EPISODE_TIME_S = 300  # 5 minutes max (will stop earlier when home)
FPS = 30

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

    Auto-stops when the arm returns to the home area after leaving it.
    Returns when the robot returns home, the timeout elapses, or Ctrl+C is pressed.
    """
    torch_device = get_safe_torch_device(device)
    _, robot_action_processor, _ = make_default_processors()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t = time.perf_counter()
    step = 0
    print("\n  Running policy... Press ENTER (terminal) or SPACE (camera) to stop manually.")
    print("  Will auto-stop when the arm returns to home.\n")

    last_obs_time: float | None = None
    last_joint_pos: dict[str, float] | None = None
    home_keys = sorted(HOME_TARGET_POS.keys())
    home_ref = np.array([HOME_TARGET_POS[k] for k in home_keys], dtype=np.float32)
    left_home = False
    stop_reason = "timeout"

    while time.perf_counter() - start_t < duration_s:
        loop_start = time.perf_counter()

        # Non-blocking stdin check — user presses Enter in the terminal
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            stop_reason = "user_terminal"
            logging.info(f"User stopped episode at step {step}")
            print(f"  >>> Stopped by user at step {step}")
            break

        obs = robot.get_observation()

        # Estimate joint velocity and home distance.
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
                print(f"  [step {step}] Left home zone (dist={home_dist:.3f})")
            elif left_home and home_dist <= HOME_RETURN_RADIUS:
                stop_reason = "auto_home_return"
                logging.info(
                    "Episode auto-stopped on return to home zone (dist=%.3f, leave>=%.3f, return<=%.3f)",
                    home_dist,
                    HOME_LEAVE_RADIUS,
                    HOME_RETURN_RADIUS,
                )
                print(f"  >>> Auto-stopped: arm returned to home zone (dist={home_dist:.3f})")
                break

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

        if step % 30 == 0:
            elapsed_s = time.perf_counter() - start_t
            print(
                f"  [step {step}, {elapsed_s:.1f}s] home_dist={home_dist:.3f}, "
                f"max_vel={max_vel:.3f}, left_home={left_home}"
            )

        elapsed = time.perf_counter() - loop_start
        sleep_time = (1 / fps) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    if stop_reason == "timeout":
        logging.info("Episode timed out after %.2fs", duration_s)
        print(f"  Episode timed out after {duration_s:.1f}s")

    elapsed_s = time.perf_counter() - start_t
    logging.info(f"Episode done: {step} steps in {elapsed_s:.1f}s ({stop_reason})")
    print(f"\n  Episode complete: {step} steps in {elapsed_s:.1f}s ({stop_reason})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_logging()

    print("\n" + "=" * 60)
    print("  Test Policy Slot")
    print("=" * 60)
    print("  Available slots:")
    for slot_num in sorted(SLOTS.keys()):
        policy_repo, _ = SLOTS[slot_num]
        print(f"    {slot_num}: {policy_repo}")
    print("=" * 60 + "\n")

    # Ask user for slot number
    while True:
        try:
            slot = int(input("  Enter slot number (1-7): ").strip())
            if slot not in SLOTS:
                print(f"  Invalid slot {slot}. Please enter a number between 1 and 7.")
                continue
            break
        except ValueError:
            print("  Invalid input. Please enter a number.")

    policy_repo, ds_repo = SLOTS[slot]
    print(f"\n  Testing slot {slot}:")
    print(f"    Policy:  {policy_repo}")
    print(f"    Dataset: {ds_repo}")

    # ---- Set up robot ----
    print("\n  Connecting to robot...")
    robot_cfg = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=CAMERAS,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    print("  Robot connected.")

    # ---- Load policy ----
    device = DEVICE
    print(f"\n  Loading policy from {policy_repo}...")
    policy, preprocessor, postprocessor, ds_meta = load_policy_and_processors(
        policy_repo, ds_repo, device
    )
    print("  Policy loaded.")

    # ---- Run episode ----
    print(f"\n  Starting episode (max {EPISODE_TIME_S}s, will auto-stop when home)...")
    try:
        run_episode(
            robot, policy, preprocessor, postprocessor,
            ds_meta, device, EPISODE_TIME_S, FPS,
        )
    except KeyboardInterrupt:
        print("\n  Episode interrupted by user.")
    finally:
        robot.disconnect()
        print("\n  Robot disconnected.")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
