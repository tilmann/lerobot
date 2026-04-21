"""
Training script for ACT policies for 4-in-a-row (one policy per slot).
Runs directly on a Lightning AI Studio (L40s GPU).

Prerequisites:
    export HF_TOKEN=hf_xxx
    export WANDB_API_KEY=xxx

Usage:
    # Test single slot (100 steps):
    python lightning_train.py --slot 1 --test

    # Train single slot (50K steps):
    python lightning_train.py --slot 1
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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train
from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_SLOT = 5
DATASET_REPO_ID = "tilmannb/4inarow_slot5"  # single-slot dataset target
OUTPUT_BASE = Path("/teamspace/studios/this_studio/outputs")


def _build_image_transforms_config(args: argparse.Namespace) -> ImageTransformsConfig:
    tfs: dict[str, ImageTransformConfig] = {
        "brightness": ImageTransformConfig(
            weight=1.0,
            type="ColorJitter",
            kwargs={"brightness": (args.brightness[0], args.brightness[1])},
        ),
        "contrast": ImageTransformConfig(
            weight=1.0,
            type="ColorJitter",
            kwargs={"contrast": (args.contrast[0], args.contrast[1])},
        ),
        "saturation": ImageTransformConfig(
            weight=1.0,
            type="ColorJitter",
            kwargs={"saturation": (args.saturation[0], args.saturation[1])},
        ),
        "hue": ImageTransformConfig(
            weight=1.0,
            type="ColorJitter",
            kwargs={"hue": (args.hue[0], args.hue[1])},
        ),
    }
    if args.enable_sharpness:
        tfs["sharpness"] = ImageTransformConfig(
            weight=1.0,
            type="SharpnessJitter",
            kwargs={"sharpness": (args.sharpness[0], args.sharpness[1])},
        )

    return ImageTransformsConfig(
        enable=args.image_transforms_enable,
        max_num_transforms=args.image_max_num_transforms,
        random_order=args.image_random_order,
        tfs=tfs,
    )


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


def train_slot(slot: int, dataset_repo_id: str, image_transforms: ImageTransformsConfig, test: bool = False) -> str:
    steps = 100 if test else 50_000
    save_freq = 50 if test else 5_000
    repo_id = f"tilmannb/act_4inarow_slot{slot}"
    output_dir = OUTPUT_BASE / f"act_4inarow_slot{slot}"

    dataset = DatasetConfig(
        repo_id=dataset_repo_id,
        image_transforms=image_transforms,
        video_backend="pyav",
        revision="main",
    )

    policy = ACTConfig(
        device="cuda",
        repo_id=repo_id,
    )

    wandb_cfg = WandBConfig(
        enable=not test,
        project="4inarow",
    )

    cfg = TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        output_dir=output_dir,
        job_name=f"act_4inarow_slot{slot}",
        steps=steps,
        batch_size=16,
        save_checkpoint=True,
        save_freq=save_freq,
        wandb=wandb_cfg,
    )

    active_tfs = list(image_transforms.tfs.keys()) if image_transforms.enable else []
    logging.info("Effective training configuration:")
    logging.info("  dataset_repo_id: %s", dataset_repo_id)
    logging.info("  output_dir: %s", output_dir)
    logging.info("  steps: %d", steps)
    logging.info("  batch_size: %d", cfg.batch_size)
    logging.info(
        "  image_transforms: enable=%s, max_num_transforms=%d, random_order=%s, active=%s",
        image_transforms.enable,
        image_transforms.max_num_transforms,
        image_transforms.random_order,
        active_tfs,
    )

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
                        push_checkpoint_to_hub(ckpt_dir, repo_id, step)
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
                push_checkpoint_to_hub(ckpt_dir, repo_id, step)

    return f"Slot {slot} training complete ({steps} steps)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACT policies for 4-in-a-row")
    parser.add_argument("--slot", type=int, default=DEFAULT_SLOT, help="Slot to train")
    parser.add_argument("--dataset-repo-id", type=str, default=DATASET_REPO_ID, help="Single-slot dataset repo id")
    parser.add_argument("--test", action="store_true", help="Run a quick test (100 steps)")

    parser.add_argument("--image-transforms-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-max-num-transforms", type=int, default=2)
    parser.add_argument("--image-random-order", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--brightness", type=float, nargs=2, default=(0.9, 1.1), metavar=("MIN", "MAX"))
    parser.add_argument("--contrast", type=float, nargs=2, default=(0.9, 1.1), metavar=("MIN", "MAX"))
    parser.add_argument("--saturation", type=float, nargs=2, default=(0.95, 1.05), metavar=("MIN", "MAX"))
    parser.add_argument("--hue", type=float, nargs=2, default=(-0.02, 0.02), metavar=("MIN", "MAX"))
    parser.add_argument("--sharpness", type=float, nargs=2, default=(0.9, 1.1), metavar=("MIN", "MAX"))
    parser.add_argument("--enable-sharpness", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    image_transforms = _build_image_transforms_config(args)
    result = train_slot(
        slot=args.slot,
        dataset_repo_id=args.dataset_repo_id,
        image_transforms=image_transforms,
        test=args.test,
    )
    print(result)


if __name__ == "__main__":
    main()
