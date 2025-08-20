from pathlib import Path

import modal

from training.sample import sample
from training.train import train as local_train

app = modal.App("image-latent-transformer")

MODEL_OUTPUT_DIR = "/output"

# Copy the entire project into the image
library_path = Path(__file__).parent.parent
image = (
    modal.Image.micromamba(python_version="3.12")
    .micromamba_install(
        "pycairo",
        "pygobject",
        "manimpango",
        channels=["conda-forge"],
    )
    .apt_install("git")
    .run_commands("mkdir -p /app/image_latent_transformer")
    .add_local_file(library_path / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .workdir("/app")
    .pip_install(".[train]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # turn on faster downloads from HF
    .add_local_dir(library_path / "image_latent_transformer", "/app/image_latent_transformer")
    .add_local_dir(library_path / "training", "/app/training")
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
        MODEL_OUTPUT_DIR: modal.Volume.from_name("model-output", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    ],
    timeout=3600,  # 1 hour
    retries=0  # Do not retry on failure
)
def train_remote(args: dict):
    import subprocess
    # Your real training script (kept in repo) with args from env if you want
    subprocess.check_call(["nvidia-smi"])

    local_train(args)

@app.local_entrypoint()
def train():
    args = [
        "--image_encoder_model_name_or_path", "WinKawaks/vit-tiny-patch16-224",
        "--bytes_encoder_model_name_or_path", "prajjwal1/bert-tiny",
        "--latent_transformer_model_name_or_path", "EleutherAI/pythia-70m",
        "--bytes_decoder_model_name_or_path", "sbintuitions/tiny-lm",
        "--dataset_name", "wikitext",
        "--dataset_config_name", "wikitext-2-raw-v1",
        "--remove_unused_columns", "False",
        "--per_device_train_batch_size", "16",
        "--per_device_eval_batch_size", "16",
        "--do_train", "True",
        # "--do_eval", "True",
        "--output_dir", MODEL_OUTPUT_DIR,
        "--logging_steps", "10",
        "--logging_strategy", "steps",
        "--max_steps", "5000",
        "--max_sequence_length", "128",
        "--max_word_length", "32",
        "--dataloader_num_workers", "4",
        "--include_tokens_per_second", "True",
        "--include_num_input_tokens_seen", "True",
        "--max_train_samples", "16",
    ]

    train_remote.remote(args)


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        MODEL_OUTPUT_DIR: modal.Volume.from_name("model-output", create_if_missing=True),
    },
    timeout=60,  # 60 seconds
    retries=0  # Do not retry on failure
)
def sample_remote():
    sample(Path(MODEL_OUTPUT_DIR))


@app.local_entrypoint()
def predict():
    sample_remote.remote()
