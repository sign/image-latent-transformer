from functools import partial

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    enable_full_determinism,
    set_seed,
)

from image_latent_transformer.collator import collate_fn
from image_latent_transformer.config import ImageLatentTransformerConfig
from image_latent_transformer.model import ImageLatentTransformerForCausalLM
from image_latent_transformer.processor import TextImageProcessor
from image_latent_transformer.tokenizer.utf8 import UTF8Tokenizer


def print_model_summary(name: str, model):
    """Print a summary of the model's architecture."""
    if model is None:
        print(name, "is None")
        return
    total_params = sum(p.numel() for p in model.parameters())
    print(name, f"Total parameters: {total_params:,}")


def setup_model(
        image_encoder_name="WinKawaks/vit-tiny-patch16-224",
        bytes_encoder_name="prajjwal1/bert-tiny",
        latent_transformer_name="EleutherAI/pythia-70m",
        bytes_decoder_name="EleutherAI/pythia-70m",
        trust_remote_code=False,
        modality_dropout=0.15,
        dtype=torch.float32,
        seed=42,
        load_pretrained=True,
):
    set_seed(seed, deterministic=True)
    enable_full_determinism(seed=seed, warn_only=True)

    if image_encoder_name is not None:
        image_processor = AutoImageProcessor.from_pretrained(image_encoder_name, use_fast=True)
    else:
        image_processor = None

    tokenizer = UTF8Tokenizer()

    config = ImageLatentTransformerConfig(
        # All sub-configs are loaded from the respective model names
        image_encoder=AutoConfig.from_pretrained(image_encoder_name) if image_encoder_name else None,
        bytes_encoder=AutoConfig.from_pretrained(bytes_encoder_name) if bytes_encoder_name else None,
        latent_transformer=AutoConfig.from_pretrained(latent_transformer_name),
        bytes_decoder=AutoConfig.from_pretrained(bytes_decoder_name),
        # Other configuration parameters
        modality_dropout=modality_dropout,
        tokenizer_class=tokenizer.__class__.__name__,
        num_tokens=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )

    # Combine the models
    model = ImageLatentTransformerForCausalLM(config, load_pretrained=load_pretrained)
    print_model_summary("Image Encoder", model.image_encoder)
    print_model_summary("Bytes Encoder", model.bytes_encoder)
    print_model_summary("Latent Transformer", model.latent_transformer)
    print_model_summary("Bytes Decoder", model.bytes_decoder)
    print_model_summary("Final Model", model)

    max_seq_length = getattr(model.latent_transformer.config, "max_position_embeddings", 1024)
    max_word_length = getattr(model.bytes_decoder.config, "max_position_embeddings", 128)
    processor = TextImageProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_seq_length=max_seq_length,
        max_word_length=max_word_length,
    )

    collator = partial(collate_fn, pad_value=tokenizer.pad_token_type_id)

    return model, processor, collator
