from functools import partial

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    enable_full_determinism,
    set_seed,
)

from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.tokenizer import ByteTokenizer
from image_latent_transformer.utils import collate_fn


def print_model_summary(name: str, model):
    """Print a summary of the model's architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    print(name, f"Total parameters: {total_params:,}")


def setup_model(
        image_encoder_name="WinKawaks/vit-tiny-patch16-224",
        byte_encoder_name="prajjwal1/bert-tiny",
        latent_transformer_name="EleutherAI/pythia-70m",
        byte_decoder_name="EleutherAI/pythia-70m",
        seed=42
):
    set_seed(seed, deterministic=True)
    enable_full_determinism(seed=seed)

    """Set up the ImageLatentTransformer model like in image_model.py."""
    # Image Encoder
    image_processor = AutoImageProcessor.from_pretrained(image_encoder_name, use_fast=True)
    image_model = AutoModelForImageClassification.from_pretrained(image_encoder_name)
    image_model.classifier = torch.nn.Identity()
    print_model_summary("Image Encoder", image_model)

    # Small Language Model (~106M parameters)
    tokenizer = ByteTokenizer()
    byte_lm = AutoModelForCausalLM.from_pretrained(byte_decoder_name)
    byte_lm.resize_token_embeddings(len(tokenizer))
    print_model_summary("Bytes LM", byte_lm)

    # Bytes Encoder (~111M parameters)
    byte_encoder = AutoModelForMaskedLM.from_pretrained(byte_encoder_name)
    byte_encoder.resize_token_embeddings(len(tokenizer))
    print_model_summary("Bytes MLM", byte_encoder)

    # Latent Transformer
    # latent_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    # Using a smaller transformer for the test
    latent_lm = AutoModelForCausalLM.from_pretrained(latent_transformer_name)
    latent_lm.resize_token_embeddings(0)
    print_model_summary("Latent LM", latent_lm)

    # Combine the models
    model = ImageLatentTransformer(
        image_encoder=image_model,
        bytes_encoder=byte_encoder,
        latent_transformer=latent_lm,
        bytes_decoder=byte_lm
    )
    print_model_summary("Final Model", model)

    collator = partial(collate_fn, pad_value=tokenizer.pad_token_type_id)

    return model, image_processor, tokenizer, collator
