import importlib.util
from functools import partial

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    PretrainedConfig,
    enable_full_determinism,
    set_seed,
)

from image_latent_transformer.collator import collate_fn
from image_latent_transformer.config import ImageLatentTransformerConfig
from image_latent_transformer.model import ImageLatentTransformerForCausalLM, logger
from image_latent_transformer.processor import TextImageProcessor
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from image_latent_transformer.vision.navit import NaViTConfig


def print_model_summary(name: str, model):
    """Print a summary of the model's architecture."""
    if model is None:
        print(name, "is None")
        return
    total_params = sum(p.numel() for p in model.parameters())
    print(name, f"Total parameters: {total_params:,}")


def get_attn_implementation():
    if importlib.util.find_spec("flash_attn") is None:
        logger.warning("Flash Attention not available, using default attention")
        return None
    return "flash_attention_2"


CUSTOM_MODELS: dict[str, PretrainedConfig] = {
    "NaViT-tiny": NaViTConfig(
        patch_size=16,
        hidden_size=128,
        dim=64,
        depth=3,
        heads=4,
        mlp_dim=128,
        dropout=0.0,
        emb_dropout=0.0,
        token_dropout_prob=0.1,
    ),
    "NaViT-small": NaViTConfig(
        patch_size=16,
        hidden_size=512,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.0,
        emb_dropout=0.0,
        token_dropout_prob=0.1,
    )
}
CUSTOM_PROCESSORS_ALIAS: dict[str, str] = {
    "NaViT-tiny": "WinKawaks/vit-tiny-patch16-224",
    "NaViT-small": "WinKawaks/vit-tiny-patch16-224",
}


def get_model_config(model_name):
    if model_name in CUSTOM_MODELS:
        return CUSTOM_MODELS[model_name]
    return AutoConfig.from_pretrained(model_name)


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
        image_processor_name = CUSTOM_PROCESSORS_ALIAS.get(image_encoder_name, image_encoder_name)
        image_processor = AutoImageProcessor.from_pretrained(image_processor_name, use_fast=True)
    else:
        image_processor = None

    tokenizer = UTF8Tokenizer()

    config = ImageLatentTransformerConfig(
        # All sub-configs are loaded from the respective model names
        image_encoder=get_model_config(image_encoder_name) if image_encoder_name else None,
        bytes_encoder=get_model_config(bytes_encoder_name) if bytes_encoder_name else None,
        latent_transformer=get_model_config(latent_transformer_name),
        bytes_decoder=get_model_config(bytes_decoder_name),
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
    model = ImageLatentTransformerForCausalLM(config,
                                              load_pretrained=load_pretrained,
                                              attn_implementation=get_attn_implementation())
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
