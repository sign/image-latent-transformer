from functools import partial

import torch
from transformers import (
    AutoImageProcessor,
    enable_full_determinism,
    set_seed, AutoConfig,
)

from image_latent_transformer.config import ImageLatentTransformerConfig
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
        torch_dtype=torch.float32,
        seed=42
):
    set_seed(seed, deterministic=True)
    enable_full_determinism(seed=seed, warn_only=True)

    image_processor = AutoImageProcessor.from_pretrained(image_encoder_name, use_fast=True)
    tokenizer = ByteTokenizer()

    config = ImageLatentTransformerConfig(
        # All sub-configs are loaded from the respective model names
        image_encoder=AutoConfig.from_pretrained(image_encoder_name),
        bytes_encoder=AutoConfig.from_pretrained(bytes_encoder_name),
        latent_transformer=AutoConfig.from_pretrained(latent_transformer_name),
        bytes_decoder=AutoConfig.from_pretrained(bytes_decoder_name),
        # Other configuration parameters
        tokenizer_class=tokenizer.__class__.__name__,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    # Combine the models
    model = ImageLatentTransformer(config)
    print_model_summary("Image Encoder", model.image_encoder)
    print_model_summary("Bytes Encoder", model.bytes_encoder)
    print_model_summary("Latent Transformer", model.latent_transformer)
    print_model_summary("Bytes Decoder", model.bytes_decoder)
    print_model_summary("Final Model", model)

    processor = TextImageProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_seq_length=config.latent_transformer.max_position_embeddings,
        max_word_length=config.bytes_decoder.max_position_embeddings,
    )

    collator = partial(collate_fn, pad_value=tokenizer.pad_token_type_id)

    return model, processor, collator
