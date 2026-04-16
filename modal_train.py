"""
Modal app for training ACT policies for 4-in-a-row (one policy per slot).

Prerequisites:
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal secret create wandb WANDB_API_KEY=xxx

Usage:
    # Test single slot (100 steps):
    modal run modal_train.py --slot 1 --test

    # Train single slot (50K steps):
    modal run modal_train.py --slot 1

    # Train slots 1 and 7 in parallel:
    modal run modal_train.py
"""

import modal

SLOTS = [2, 3,4,5,6]  # slots to train (1-indexed); set to [1,7] to train both in parallel

app = modal.App("4inarow-training")

gpu = "A10"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "lerobot[training] @ git+https://github.com/huggingface/lerobot.git@main",
        "wandb",
        gpu=gpu,
    )
)

volume = modal.Volume.from_name("4inarow-checkpoints", create_if_missing=True)


def push_checkpoint_to_hub(checkpoint_dir, repo_id, step):
    """Push a checkpoint's pretrained_model/ dir to HuggingFace Hub."""
    import logging

    from huggingface_hub import HfApi

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


@app.function(
    image=image,
    gpu=gpu,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    volumes={"/outputs": volume},
    timeout=6 * 3600,
)
def train_slot(slot: int, test: bool = False):
    import logging
    import threading
    import time
    from pathlib import Path

    from lerobot.common.train_utils import get_step_checkpoint_dir
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts.lerobot_train import train

    steps = 100 if test else 50_000
    save_freq = 50 if test else 10_000
    repo_id = f"tilmannb/act_4inarow_slot{slot}"
    output_dir = Path(f"/outputs/act_4inarow_slot{slot}")

    dataset = DatasetConfig(
        repo_id=f"tilmannb/4inarow_slot{slot}",
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

    # Clean up stale output dir from previous runs on this volume
    import shutil

    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Background thread to watch for new checkpoints and push them to Hub
    uploaded_steps = set()
    stop_event = threading.Event()

    def checkpoint_watcher():
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

    volume.commit()
    return f"Slot {slot} training complete ({steps} steps)"


@app.local_entrypoint()
def main(slot: int = 0, test: bool = False):
    if slot > 0:
        result = train_slot.remote(slot, test=test)
        print(result)
    else:
        results = list(train_slot.map(SLOTS, kwargs={"test": test}))
        for r in results:
            print(r)
