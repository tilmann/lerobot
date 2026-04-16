"""
Interactive 4-in-a-row robot controller.

Run a trained ACT policy on the SO101 robot. Type a slot number (1 or 7) in the
terminal to execute the corresponding policy.

Usage:
    uv run python play_4inarow.py
"""

import logging
import select
import sys
import time

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors, make_robot_action
from lerobot.processor import make_default_processors, rename_stats
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.utils import init_logging

# ---------------------------------------------------------------------------
# Configuration — adjust these to match your setup
# ---------------------------------------------------------------------------
ROBOT_PORT = "/dev/tty.usbmodem5A680098251"
ROBOT_ID = "my_follower"
DEVICE = "mps"
EPISODE_TIME_S = 30
FPS = 30

CAMERAS = {
    "gripper": OpenCVCameraConfig(index_or_path=1, fps=30, width=640, height=480),
    "workarea": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
}

# Maps slot number -> (policy HF repo, training dataset HF repo)
SLOTS = {
    1: ("tilmannb/act_4inarow_slot1", "tilmannb/4inarow_slot1"),
    7: ("tilmannb/act_4inarow_slot7", "tilmannb/4inarow_slot7"),
}


def load_policy_and_processors(policy_repo: str, dataset_repo: str, device: str):
    """Load a pretrained policy and its pre/post-processors."""
    # Load dataset metadata (downloads only meta/ dir, not the actual episodes)
    ds_meta = LeRobotDatasetMetadata(dataset_repo, revision="main")

    # Load policy config from HuggingFace Hub
    policy_cfg = PreTrainedConfig.from_pretrained(policy_repo)
    policy_cfg.pretrained_path = policy_repo
    policy_cfg.device = device

    # Instantiate policy with weights
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()

    # Create pre/post-processors (handles normalization etc.)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_repo,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )

    return policy, preprocessor, postprocessor, ds_meta


def run_episode(robot, policy, preprocessor, postprocessor, ds_meta, device, duration_s, fps):
    """Run one episode: execute the policy on the robot for `duration_s` seconds."""
    torch_device = get_safe_torch_device(device)
    _, robot_action_processor, _ = make_default_processors()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t = time.perf_counter()
    step = 0
    print("  Press ENTER when disc is placed to stop the robot.")

    while time.perf_counter() - start_t < duration_s:
        loop_start = time.perf_counter()

        # Check if user pressed Enter (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            logging.info(f"User stopped episode at step {step}")
            print(f"  >>> Stopped by user at step {step}")
            break

        obs = robot.get_observation()
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

        # Convert policy action tensor to robot action dict
        action_dict = make_robot_action(action, ds_meta.features)
        robot_action_to_send = robot_action_processor((action_dict, obs))

        robot.send_action(robot_action_to_send)
        step += 1

        # Maintain target FPS
        elapsed = time.perf_counter() - loop_start
        sleep_time = (1 / fps) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    logging.info(f"Episode done: {step} steps in {time.perf_counter() - start_t:.1f}s")


def main():
    init_logging()

    # Set up robot
    robot_cfg = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=CAMERAS,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()

    # Cache loaded policies so switching is fast
    policy_cache: dict[int, tuple] = {}

    print("\n" + "=" * 50)
    print("  4-in-a-Row Robot Controller")
    print("=" * 50)
    print(f"  Available slots: {sorted(SLOTS.keys())}")
    print(f"  Episode duration: {EPISODE_TIME_S}s")
    print("  Type a slot number and press Enter to execute.")
    print("  Type 'q' to quit.")
    print("=" * 50 + "\n")

    try:
        while True:
            user_input = input("Slot> ").strip()

            if user_input.lower() == "q":
                break

            try:
                slot = int(user_input)
            except ValueError:
                print(f"Invalid input. Enter one of {sorted(SLOTS.keys())} or 'q'.")
                continue

            if slot not in SLOTS:
                print(f"Slot {slot} not available. Choose from {sorted(SLOTS.keys())}.")
                continue

            # Load policy if not cached
            if slot not in policy_cache:
                policy_repo, dataset_repo = SLOTS[slot]
                print(f"Loading policy for slot {slot} from {policy_repo}...")
                policy, preprocessor, postprocessor, ds_meta = load_policy_and_processors(
                    policy_repo, dataset_repo, DEVICE
                )
                policy_cache[slot] = (policy, preprocessor, postprocessor, ds_meta)
                print(f"Policy for slot {slot} loaded.")
            else:
                policy, preprocessor, postprocessor, ds_meta = policy_cache[slot]

            print(f"Running slot {slot} for {EPISODE_TIME_S}s... (Ctrl+C to abort)")
            try:
                run_episode(robot, policy, preprocessor, postprocessor, ds_meta, DEVICE, EPISODE_TIME_S, FPS)
            except KeyboardInterrupt:
                print("\nEpisode aborted.")

    except (KeyboardInterrupt, EOFError):
        print("\nShutting down.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
