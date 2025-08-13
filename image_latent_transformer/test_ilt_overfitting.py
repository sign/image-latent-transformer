import tempfile
from functools import partial

import pytest
import torch
from transformers import Trainer, TrainingArguments

from transformers.modeling_outputs import CausalLMOutput

from image_latent_transformer.dataset import TextImageDataset
from image_latent_transformer.test_model import setup_model, predict_dataset
from image_latent_transformer.utils import collate_fn


@pytest.fixture(scope="module")
def trained_model():
    """Train the model once and reuse for all tests."""

    # Setup
    model, image_processor, tokenizer = setup_model()
    collator = partial(collate_fn, pad_value=tokenizer.pad_token_type_id)

    def make_dataset(texts: list[str]):
        """Create a dataset from a list of texts."""
        return TextImageDataset(
            texts_dataset=texts,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_seq_length=128,
            max_word_length=32
        )

    # Create training data - all four texts for better testing
    train_texts = ["a b", "b a", "a cat", "a dog"]

    # Setup training arguments with more epochs for overfitting
    training_args = TrainingArguments(
        output_dir=tempfile.mktemp(),
        num_train_epochs=50,  # Much more epochs for better overfitting
        per_device_train_batch_size=len(train_texts),
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="no",  # Disable saving to avoid shared tensor issues
        remove_unused_columns=False,
        dataloader_drop_last=False,
        warmup_steps=0,  # No warmup for immediate learning
        weight_decay=0.0,  # No regularization for overfitting
        learning_rate=1e-4,
        lr_scheduler_type="constant",  # Keep learning rate constant
        use_mps_device=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=make_dataset(train_texts),
        data_collator=collator,
    )

    # Train the model
    print("Training model on texts:", train_texts)
    trainer.train()

    # Set to eval mode
    model.eval()

    return model, image_processor, tokenizer, collator


@pytest.fixture
def model_configuration(request, trained_model):
    """Configure model based on the test parameter."""
    model, image_processor, tokenizer, collator = trained_model
    config_name = request.param

    # Store original encoders
    original_bytes_encoder = model.bytes_encoder
    original_image_encoder = model.image_encoder

    # Apply configuration
    if config_name == "full_model":
        print("\n[Configuration: Full model with all encoders]")
        # No changes needed
    elif config_name == "no_bytes_encoder":
        print("\n[Configuration: Model without bytes encoder]")
        model.bytes_encoder = None
    elif config_name == "no_image_encoder":
        print("\n[Configuration: Model without image encoder]")
        model.image_encoder = None

    # Yield the configured model
    yield model, image_processor, tokenizer, collator

    # Restore original encoders after test
    model.bytes_encoder = original_bytes_encoder
    model.image_encoder = original_image_encoder


@pytest.mark.parametrize("model_configuration", [
    "full_model",
    "no_bytes_encoder",
    "no_image_encoder"
], indirect=True)
def test_character_level_conditioning(model_configuration):
    """Test 1: Character-level conditioning (a b vs a a, b a vs b b)"""
    model, image_processor, tokenizer, collator = model_configuration

    print("\n=== Test 1: Character-level conditioning ===")

    test_texts_char = ["a b", "b a", "a a", "b b"]
    predictions = predict_dataset(test_texts_char, model, image_processor, tokenizer, collator)
    losses = {text: pred.loss.item() for text, pred in predictions.items()}

    # Check conditioning: trained sequences should have lower loss
    assert losses['a b'] < losses['a a'], \
        f"'a b' should have lower loss than 'a a': {losses['a b']:.4f} vs {losses['a a']:.4f}"

    assert losses['b a'] < losses['b b'], \
        f"'b a' should have lower loss than 'b b': {losses['b a']:.4f} vs {losses['b b']:.4f}"

    print("✅ Character-level conditioning test passed!")


@pytest.mark.parametrize("model_configuration", [
    "full_model",
    "no_bytes_encoder",
    "no_image_encoder"
], indirect=True)
def test_word_level_conditioning(model_configuration):
    """Test 2: Word-level conditioning (a cat vs a dat, a dog vs a cog)"""
    model, image_processor, tokenizer, collator = model_configuration

    print("\n=== Test 2: Word-level conditioning ===")

    test_texts_word = ["a cat", "a dog", "a dat", "a cog", "a bat", "a fog"]
    predictions = predict_dataset(test_texts_word, model, image_processor, tokenizer, collator)
    word_losses = {text: pred.loss.item() for text, pred in predictions.items()}

    # Check conditioning: trained sequences should have lower loss
    assert word_losses['a cat'] < word_losses['a dat'], \
        f"'a cat' should have lower loss than 'a dat': {word_losses['a cat']:.4f} vs {word_losses['a dat']:.4f}"

    assert word_losses['a dog'] < word_losses['a cog'], \
        f"'a dog' should have lower loss than 'a cog': {word_losses['a dog']:.4f} vs {word_losses['a cog']:.4f}"

    print("✅ Word-level conditioning test passed!")


@pytest.mark.parametrize("model_configuration", [
    "full_model",
    "no_bytes_encoder",
    "no_image_encoder"
], indirect=True)
def test_byte_level_conditioning(model_configuration):
    """Test 3: Byte-level conditioning within words"""
    model, image_processor, tokenizer, collator = model_configuration

    print("\n=== Test 3: Byte-level conditioning within words ===")

    # For "a cat" and "a dog", after seeing "a c" or "a d", the model should be confident about the rest
    # This tests that the byte decoder is properly conditioned on previous bytes

    # Create a special test to check conditional probabilities
    test_conditional = ["a cat", "a cog", "a dog", "a dat"]
    predictions = predict_dataset(test_conditional, model, image_processor, tokenizer, collator)
    conditional_losses = {text: pred.loss.item() for text, pred in predictions.items()}

    # After 'a c', 'cat' should be more likely than 'cog'
    assert conditional_losses['a cat'] < conditional_losses['a cog'], \
        (f"'a cat' should have lower loss than 'a cog': "
         f"{conditional_losses['a cat']:.4f} vs {conditional_losses['a cog']:.4f}")

    # After 'a d', 'dog' should be more likely than 'dat'
    assert conditional_losses['a dog'] < conditional_losses['a dat'], \
        (f"'a dog' should have lower loss than 'a dat': "
         f"{conditional_losses['a dog']:.4f} vs {conditional_losses['a dat']:.4f}")

    print("✅ Byte-level conditioning test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
