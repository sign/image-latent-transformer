import pytest
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM, set_seed, enable_full_determinism,
)
from transformers.modeling_outputs import CausalLMOutput

from image_latent_transformer.dataset import TextImageDataset
from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.tokenizer import ByteTokenizer


def print_model_summary(name: str, model):
    """Print a summary of the model's architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    print(name, f"Total parameters: {total_params:,}")


def setup_model():
    """Set up the ImageLatentTransformer model like in image_model.py."""
    # Image Encoder
    image_model_name = "microsoft/swinv2-tiny-patch4-window16-256"
    image_processor = AutoImageProcessor.from_pretrained(image_model_name, use_fast=True)
    image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
    image_model.classifier = torch.nn.Identity()
    print_model_summary("Image Encoder", image_model)

    # Small Language Model (~106M parameters)
    tokenizer = ByteTokenizer()
    byte_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    byte_lm.resize_token_embeddings(len(tokenizer))
    print_model_summary("Bytes LM", byte_lm)

    # Bytes Encoder (~111M parameters)
    byte_encoder = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
    byte_encoder.resize_token_embeddings(len(tokenizer))
    print_model_summary("Bytes MLM", byte_encoder)

    # Latent Transformer
    # latent_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    # Using a smaller transformer for the test
    latent_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
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

    set_seed(42, deterministic=True)
    enable_full_determinism(seed=42)

    return model, image_processor, tokenizer




def make_dataset(texts: list[str], image_processor, tokenizer):
    """Create a dataset from a list of texts."""
    return TextImageDataset(
        texts_dataset=texts,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_seq_length=128,
        max_word_length=32
    )


def predict_dataset(texts: list[str], model, image_processor, tokenizer, collator):
    """Predict a dataset and return the logits."""
    dataset = make_dataset(texts, image_processor, tokenizer)

    # Compute losses for each sequence - process entire batch at once
    device = next(model.parameters()).device
    batch = collator([dataset[i] for i in range(len(dataset))])
    # Move batch to the same device as the model
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    labels = batch['labels_output']

    output_per_text = {}
    for i, text in enumerate(texts):
        # Compute cross entropy loss for this item
        item_loss = torch.nn.functional.cross_entropy(input=logits[i].reshape(-1, logits.size(-1)),
                                                      target=labels[i].reshape(-1),
                                                      ignore_index=tokenizer.pad_token_type_id,
                                                      reduction='mean')
        print(f"Loss for '{text}': {item_loss.item():.4f}")

        output_per_text[text] = CausalLMOutput(
            loss=item_loss,
            logits=logits[i],
            hidden_states=(outputs.hidden_states[0][i],),
            attentions=None
        )

    return output_per_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
