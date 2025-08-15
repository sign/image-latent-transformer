from functools import partial

import pytest
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    enable_full_determinism,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutput

from image_latent_transformer.dataset import TextImageDataset
from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.tokenizer import ByteTokenizer
from image_latent_transformer.utils import collate_fn


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

    collator = partial(collate_fn, pad_value=tokenizer.pad_token_type_id)

    return model, image_processor, tokenizer, collator


def make_dataset(texts: list[str], image_processor, tokenizer, max_word_length=32):
    """Create a dataset from a list of texts."""
    return TextImageDataset(
        texts_dataset=texts,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_seq_length=128,
        max_word_length=max_word_length
    )


def dataset_to_batch(model, collator, dataset):
    # Compute losses for each sequence - process entire batch at once
    device = next(model.parameters()).device
    batch = collator([dataset[i] for i in range(len(dataset))])
    # Move batch to the same device as the model
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def predict_dataset(texts: list[str], model, image_processor, tokenizer, collator, dataset_kwargs=None):
    """Predict a dataset and return the logits."""
    if dataset_kwargs is None:
        dataset_kwargs = {}
    dataset = make_dataset(texts, image_processor, tokenizer, **dataset_kwargs)

    batch = dataset_to_batch(model, collator, dataset)

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    labels = batch['labels_output']

    output_per_text = {}
    losses = {}
    for i, text in enumerate(texts):
        # Compute cross entropy loss for this item
        item_loss = torch.nn.functional.cross_entropy(input=logits[i].reshape(-1, logits.size(-1)),
                                                      target=labels[i].reshape(-1),
                                                      ignore_index=tokenizer.pad_token_type_id,
                                                      reduction='none')
        losses[text] = item_loss.mean().item()
        print(f"Loss for '{text}': {losses[text]:.4f}")

        output_per_text[text] = CausalLMOutput(
            loss=item_loss,
            logits=logits[i],
            hidden_states=(outputs.hidden_states[0][i],),
            attentions=None
        )

    return losses, output_per_text


def test_attention_no_look_ahead():
    """Test that attention does not look ahead - causal masking is working correctly."""
    model, image_processor, tokenizer, collator = setup_model()
    model.eval()

    # Test sequences that share prefixes
    texts = ["a b c", "a b d"]

    _, outputs = predict_dataset(texts, model, image_processor, tokenizer, collator, dataset_kwargs={
        # Force every word to predict a single byte (and EOS)
        # "a <eos>, b <eos>, c <eos>, <eos> <pad>" and "a <eos>, b <eos>, d <eos>, <eos> <pad>"
        "max_word_length": 1
    })
    for text in texts:
        print(f"Loss for '{text}':", outputs[text].loss.cpu().numpy())

    # Check that the first 4 tokens have identical losses
    for i in range(4):
        assert abs(outputs[texts[0]].loss[i] - outputs[texts[1]].loss[i]) < 1e-4, \
            f"Loss at position {i} should be identical: {outputs[texts[0]].loss[i]} vs {outputs[texts[1]].loss[i]}"


def test_attention_does_look_back():
    """Test that attention does look back - model uses previous context."""
    model, image_processor, tokenizer, collator = setup_model()
    model.eval()

    # Test sequences with shared suffix but different prefix
    texts = ["c b a", "d b a"]

    _, outputs = predict_dataset(texts, model, image_processor, tokenizer, collator, dataset_kwargs={
        # Force every word to predict a single byte (and EOS)
        # "c <eos>, b <eos>, a <eos>, <eos> <pad>" and "d <eos>, b <eos>, a <eos>, <eos> <pad>"
        "max_word_length": 1
    })
    for text in texts:
        print(f"Loss for '{text}':", outputs[text].loss.cpu().numpy())

    # Check that ALL positions have different losses due to different context
    for i in range(7):  # Check all 7 positions (excluding padding)
        loss_diff = abs(outputs[texts[0]].loss[i] - outputs[texts[1]].loss[i])
        assert loss_diff > 1e-4, \
            (f"Loss at position {i} should be different due to context: "
             f"{outputs[texts[0]].loss[i]} vs {outputs[texts[1]].loss[i]} (diff: {loss_diff})")


def test_loss_is_independent_of_batch():
    """Test that loss at first position is identical regardless of other items in batch."""
    model, image_processor, tokenizer, collator = setup_model()
    model.eval()

    # TODO: turn it back on once https://github.com/sign/image-latent-transformer/issues/1 is fixed
    model.image_encoder = None

    batches = [
        # Run first batch with just "a"
        ["a"],
        # Run second batch with "a" and additional text
        ["a", "2 w"],
        # Run third batch with "a" and additional longer text
        ["a", "two words"]
    ]
    outputs = [predict_dataset(batch, model, image_processor, tokenizer, collator)[1] for  batch in batches]

    # Get the loss for "a" from both batches
    losses = [outputs[i]["a"].loss[0].item() for i in range(len(outputs))]

    # Check that the loss at the first position (first token) is nearly identical
    # Note: losses[0] and losses[1] should be the same
    # since they're both predicting the same token with the same context
    # Small numerical differences are acceptable due to batching implementation details
    tolerance = 1e-3  # Relaxed tolerance for batch-dependent numerical precision
    assert abs(losses[0] - losses[1]) < tolerance, \
        f"Loss at first position should be nearly identical: {losses[0]} vs {losses[1]}"

    assert abs(losses[0] - losses[2]) < tolerance, \
        f"Loss at first position should be nearly identical: {losses[0]} vs {losses[2]}"

    print(f"✓ Loss at first position is batch-independent: {losses[0]:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
