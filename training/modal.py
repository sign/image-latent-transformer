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
def train_remote(args: dict):
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

    local_train(args)


@app.local_entrypoint()
def train():
    # TODO convert to yaml
    args = [
        # Model args
        "--image_encoder_model_name_or_path", "WinKawaks/vit-tiny-patch16-224",
        "--bytes_encoder_model_name_or_path", "prajjwal1/bert-tiny",
        "--latent_transformer_model_name_or_path", "EleutherAI/pythia-70m",
        "--bytes_decoder_model_name_or_path", "sbintuitions/tiny-lm",
        # Align representations between pretrained models
        # "--load_pretrained", "True",
        # "--warmup_freeze_steps", "500",
        # Dataset args
        "--dataset_name", "Helsinki-NLP/opus-100",
        "--dataset_config_name", "en-he",
        "--dataset_text_template", "<en>\x0E{translation[en]}\x0F<he> {translation[he]}",
        "--remove_unused_columns", "False",
        # Dataloader args
        "--dataloader_num_workers", "8",
        "--dataloader_prefetch_factor", "4",
        "--dataloader_pin_memory", "True",
        "--dataloader_persistent_workers", "True",
        # Training args
        "--per_device_train_batch_size", "176",
        "--per_device_eval_batch_size", "176",
        "--max_sequence_length", "128",
        "--max_word_length", "16",
        "--auto_find_batch_size", "true",
        "--do_train", "True",
        # "--do_eval", "True", # TODO: fix eval
        "--output_dir", MODEL_OUTPUT_DIR,
        "--overwrite_output_dir", "True",
        "--logging_steps", "10",
        "--logging_strategy", "steps",
        "--max_steps", "30000",

        "--include_tokens_per_second", "True",
        "--include_num_input_tokens_seen", "True",
        "--learning_rate", "3e-4",
        "--dtype", "bfloat16",
        "--bf16", "True",
        "--optim", "adamw_torch_fused",
        "--report_to", "wandb",
    ]

    train_remote.remote(args)


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
