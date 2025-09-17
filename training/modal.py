from pathlib import Path

import modal
from modal import Retries

from training.sample import sample
from training.train import train as local_train

app = modal.App("image-latent-transformer")

MODEL_MNT_DIR = "/output"
MODEL_OUTPUT_DIR = f"{MODEL_MNT_DIR}/en-he-28m-d"
# Copy the entire project into the image
library_path = Path(__file__).parent.parent
image = (
    modal.Image.from_dockerfile("Dockerfile")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # turn on faster downloads from HF
        "WANDB_PROJECT": "image-latent-transformer",
    })
    .add_local_dir(library_path / "image_latent_transformer", "/app/image_latent_transformer")
    .add_local_dir(library_path / "training", "/app/training")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
        MODEL_MNT_DIR: modal.Volume.from_name("model-output", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
        modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
    ],
    timeout=3600 * 24,
    retries=Retries(max_retries=0)
)
def train_remote(config_path: str):
    import subprocess
    # Your real training script (kept in repo) with args from env if you want
    subprocess.check_call(["nvidia-smi"])

    # Print number of CPUs
    import os
    print(f"Number of CPUs: {os.cpu_count()}")
    subprocess.check_call(["lscpu"])

    # Print python version
    print("Python version:")
    subprocess.check_call(["python", "--version"])

    local_train(config_path)


@app.local_entrypoint()
def train(config: str):
    train_remote.remote(config)


@app.function(
    image=image,
    gpu="A100",
    volumes={
        MODEL_MNT_DIR: modal.Volume.from_name("model-output", create_if_missing=True),
    },
    timeout=240,
    retries=Retries(max_retries=0)
)
def sample_remote():
    sample(Path(MODEL_OUTPUT_DIR))


@app.local_entrypoint()
def predict():
    sample_remote.remote()
