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
from image_latent_transformer.processor import TextImageProcessor
from image_latent_transformer.tokenizer import ByteTokenizer
from image_latent_transformer.utils import collate_fn


def print_model_summary(name: str, model):
    """Print a summary of the model's architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    print(name, f"Total parameters: {total_params:,}")


def setup_model(
        image_encoder_name="WinKawaks/vit-tiny-patch16-224",
        bytes_encoder_name="prajjwal1/bert-tiny",
        latent_transformer_name="EleutherAI/pythia-70m",
        bytes_decoder_name="EleutherAI/pythia-70m",
        trust_remote_code=False,
        torch_dtype=None,
        seed=42
):
    set_seed(seed, deterministic=True)
    enable_full_determinism(seed=seed, warn_only=True)

    """Set up the ImageLatentTransformer model like in image_model.py."""
    from_args = dict(
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    # Image Encoder
    image_processor = AutoImageProcessor.from_pretrained(image_encoder_name, use_fast=True)
    image_model = AutoModelForImageClassification.from_pretrained(image_encoder_name, **from_args)
    image_model.classifier = torch.nn.Identity()
    print_model_summary("Image Encoder", image_model)

    # Small Language Model (~106M parameters)
    tokenizer = ByteTokenizer()
    byte_lm = AutoModelForCausalLM.from_pretrained(bytes_decoder_name, **from_args)
    byte_lm.resize_token_embeddings(len(tokenizer))
    print_model_summary("Bytes LM", byte_lm)

    # Bytes Encoder (~111M parameters)
    byte_encoder = AutoModelForMaskedLM.from_pretrained(bytes_encoder_name, **from_args)
    byte_encoder.resize_token_embeddings(len(tokenizer))
    byte_encoder.cls = torch.nn.Identity() # delete the decoder head
    print_model_summary("Bytes MLM", byte_encoder)

    # Latent Transformer
    # latent_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    # Using a smaller transformer for the test
    latent_lm = AutoModelForCausalLM.from_pretrained(latent_transformer_name, **from_args)
    latent_lm.resize_token_embeddings(0)
    print_model_summary("Latent LM", latent_lm)

    # Combine the models
    model = ImageLatentTransformer(
        image_encoder=image_model,
        bytes_encoder=byte_encoder,
        latent_transformer=latent_lm,
        bytes_decoder=byte_lm,
        padding_index=tokenizer.pad_token_id,
    )
    print_model_summary("Final Model", model)

    # Update config with tokenizer information
    model.config.update(dict(
        trust_remote_code=trust_remote_code,
        tokenizer_class=tokenizer.__class__.__name__,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
    ))

    processor = TextImageProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_seq_length=latent_lm.config.max_position_embeddings,
        max_word_length=byte_lm.config.max_position_embeddings,
    )

    collator = partial(collate_fn, pad_value=tokenizer.pad_token_type_id)

    return model, processor, collator
