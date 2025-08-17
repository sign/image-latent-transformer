from pathlib import Path

import modal

from training.train import train as local_train

app = modal.App("image-latent-transformer")

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
    .add_local_file(library_path / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_dir(library_path / "image_latent_transformer", "/app/image_latent_transformer", copy=True)
    .workdir("/app")
    .pip_install(".[train]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # turn on faster downloads from HF
    .add_local_dir(library_path / "training", "/app/training")
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    ],
    timeout=600,  # 10 minutes
    retries=0 # Do not retry on failure
)
def train(args: dict):
    import subprocess
    # Your real training script (kept in repo) with args from env if you want
    subprocess.check_call(["nvidia-smi"])

    local_train(args)


@app.local_entrypoint()
def main():
    args = [
        "--image_encoder_model_name_or_path", "WinKawaks/vit-tiny-patch16-224",
        "--bytes_encoder_model_name_or_path", "prajjwal1/bert-tiny",
        "--latent_transformer_model_name_or_path", "EleutherAI/pythia-70m",
        "--bytes_decoder_model_name_or_path", "sbintuitions/tiny-lm",
        "--dataset_name", "wikitext",
        "--dataset_config_name", "wikitext-2-raw-v1",
        "--remove_unused_columns", "False",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--do_train", "True",
        "--do_eval", "True",
        "--output_dir", "/tmp/test-clm",
        "--logging_steps", "1",
        "--logging_strategy", "steps",
        "--max_steps", "100",
        "--max_sequence_length", "32",
        "--max_word_length", "8",
    ]

    train.remote(args)
