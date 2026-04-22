"""
Training script for SmolVLA policy for 4-in-a-row slot 6.
Finetuning from lerobot/smolvla_base on episodes 50-100.
Runs directly on a Lightning AI Studio (L40s GPU).

Prerequisites:
    export HF_TOKEN=hf_xxx
    export WANDB_API_KEY=xxx

Usage:
    # Test run (100 steps):
    python lightning_train_smolvla.py --test

    # Full training (20K steps):
    python lightning_train_smolvla.py
"""

import argparse
import logging
import shutil
import threading
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.common.train_utils import get_step_checkpoint_dir
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SLOT = 6
DATASET_REPO_ID = "tilmannb/4inarow_slot6"
POLICY_REPO_ID = f"tilmannb/smolvla_4inarow_slot{SLOT}"
BASE_MODEL = "lerobot/smolvla_base"
OUTPUT_BASE = Path("/teamspace/studios/this_studio/outputs")
EPISODES = list(range(50, 100))  # episodes 51-100 (0-based indices 50-99)


def push_checkpoint_to_hub(checkpoint_dir: Path, repo_id: str, step: int) -> None:
    """Push a checkpoint's pretrained_model/ dir to HuggingFace Hub."""
    pretrained_dir = checkpoint_dir / "pretrained_model"
    if not pretrained_dir.exists():
        logging.warning(f"No pretrained_model dir at {pretrained_dir}, skipping hub push")
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=pretrained_dir,
        commit_message=f"Checkpoint at step {step}",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
    )
    logging.info(f"Step {step} pushed to {commit_info.repo_url.url}")


def train_smolvla(test: bool = False) -> str:
    steps = 100 if test else 20_000
    save_freq = 50 if test else 5_000
    output_dir = OUTPUT_BASE / f"smolvla_4inarow_slot{SLOT}"

    dataset = DatasetConfig(
        repo_id=DATASET_REPO_ID,
        episodes=EPISODES,
        video_backend="pyav",
        revision="main",
    )

    policy = SmolVLAConfig(
        device="cuda",
        repo_id=POLICY_REPO_ID,
        pretrained_path=BASE_MODEL,
    )

    wandb_cfg = WandBConfig(
        enable=not test,
        project="4inarow",
    )

    cfg = TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        output_dir=output_dir,
        job_name=f"smolvla_4inarow_slot{SLOT}",
        steps=steps,
        batch_size=64,
        save_checkpoint=True,
        save_freq=save_freq,
        wandb=wandb_cfg,
    )

    logging.info("Effective training configuration:")
    logging.info("  base_model: %s", BASE_MODEL)
    logging.info("  dataset_repo_id: %s", DATASET_REPO_ID)
    logging.info("  episodes: %d-%d (%d total)", EPISODES[0], EPISODES[-1], len(EPISODES))
    logging.info("  output_dir: %s", output_dir)
    logging.info("  steps: %d", steps)
    logging.info("  batch_size: %d", cfg.batch_size)
    logging.info("  policy_repo_id: %s", POLICY_REPO_ID)

    # Clean up stale output dir from previous runs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Background thread to watch for new checkpoints and push them to Hub
    uploaded_steps: set[int] = set()
    stop_event = threading.Event()

    def checkpoint_watcher() -> None:
        while not stop_event.is_set():
            for step in range(save_freq, steps + 1, save_freq):
                if step in uploaded_steps:
                    continue
                ckpt_dir = get_step_checkpoint_dir(output_dir, steps, step)
                if ckpt_dir.exists() and (ckpt_dir / "pretrained_model" / "model.safetensors").exists():
                    try:
                        push_checkpoint_to_hub(ckpt_dir, POLICY_REPO_ID, step)
                        uploaded_steps.add(step)
                    except Exception as e:
                        logging.warning(f"Failed to push step {step}: {e}")
            stop_event.wait(30)

    watcher = threading.Thread(target=checkpoint_watcher, daemon=True)
    watcher.start()

    cfg.validate()
    train(cfg)

    # Final push: make sure the last checkpoint is uploaded
    stop_event.set()
    watcher.join(timeout=5)
    for step in range(save_freq, steps + 1, save_freq):
        if step not in uploaded_steps:
            ckpt_dir = get_step_checkpoint_dir(output_dir, steps, step)
            if ckpt_dir.exists():
                push_checkpoint_to_hub(ckpt_dir, POLICY_REPO_ID, step)

    return f"Slot {SLOT} SmolVLA training complete ({steps} steps)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune SmolVLA for 4-in-a-row slot 6")
    parser.add_argument("--test", action="store_true", help="Run a quick test (100 steps)")
    args = parser.parse_args()

    result = train_smolvla(test=args.test)
    print(result)


if __name__ == "__main__":
    main()
