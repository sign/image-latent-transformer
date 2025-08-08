from functools import partial

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    enable_full_determinism,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutput

from image_latent_transformer.dataset import TextImageDataset
from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.tokenizer import ByteTokenizer
from image_latent_transformer.utils import collate_fn


def setup_model():
    """Set up the ImageLatentTransformer model like in image_model.py."""
    # Image Encoder
    image_model_name = "microsoft/swinv2-tiny-patch4-window16-256"
    image_processor = AutoImageProcessor.from_pretrained(image_model_name, use_fast=True)
    image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
    image_model.classifier = torch.nn.Identity()

    # Small Language Model
    tokenizer = ByteTokenizer()
    byte_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    byte_lm.resize_token_embeddings(len(tokenizer))

    # Latent Transformer
    latent_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    latent_lm.resize_token_embeddings(0)

    # Combine the models
    model = ImageLatentTransformer(
        image_encoder=image_model,
        bytes_encoder=byte_lm,
        latent_transformer=latent_lm,
        bytes_decoder=byte_lm
    )

    return model, image_processor, tokenizer


def test_ilt_overfitting():
    """Test that the model can overfit on simple sequences and learn contextual patterns."""
    set_seed(42, deterministic=True)
    enable_full_determinism(seed=42)

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

    def predict_dataset(texts: list[str]):
        """Predict a dataset and return the logits."""
        dataset = make_dataset(texts)

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

    # Create training data - all four texts for better testing
    train_texts = ["a b", "b a", "a cat", "a dog"]

    # Setup training arguments with more epochs for overfitting
    training_args = TrainingArguments(
        num_train_epochs=100,  # Much more epochs for better overfitting
        per_device_train_batch_size=len(train_texts),
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="no",  # Disable saving to avoid shared tensor issues
        remove_unused_columns=False,
        dataloader_drop_last=False,
        warmup_steps=0,  # No warmup for immediate learning
        weight_decay=0.0,  # No regularization for overfitting
        learning_rate=5e-3,  # Higher learning rate for faster overfitting
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

    # Test contextual understanding
    model.eval()

    # Test 1: Character-level conditioning (a b vs a a, b a vs b b)
    print("\n=== Test 1: Character-level conditioning ===")

    test_texts_char = ["a b", "b a", "a a", "b b"]
    predictions = predict_dataset(test_texts_char)
    losses = {text: pred.loss.item() for text, pred in predictions.items()}

    # Check conditioning: trained sequences should have lower loss
    assert losses['a b'] < losses['a a'], \
        f"'a b' should have lower loss than 'a a': {losses['a b']:.4f} vs {losses['a a']:.4f}"

    assert losses['b a'] < losses['b b'], \
        f"'b a' should have lower loss than 'b b': {losses['b a']:.4f} vs {losses['b b']:.4f}"

    print("✅ Character-level conditioning test passed!")

    # Test 2: Word-level conditioning (a cat vs a dat, a dog vs a cog)
    print("\n=== Test 2: Word-level conditioning ===")

    test_texts_word = ["a cat", "a dog", "a dat", "a cog", "a bat", "a fog"]
    predictions = predict_dataset(test_texts_word)
    word_losses = {text: pred.loss.item() for text, pred in predictions.items()}

    # Check conditioning: trained sequences should have lower loss
    assert word_losses['a cat'] < word_losses['a dat'], \
        f"'a cat' should have lower loss than 'a dat': {word_losses['a cat']:.4f} vs {word_losses['a dat']:.4f}"

    assert word_losses['a dog'] < word_losses['a cog'], \
        f"'a dog' should have lower loss than 'a cog': {word_losses['a dog']:.4f} vs {word_losses['a cog']:.4f}"

    print("✅ Word-level conditioning test passed!")

    # Test 3: Check byte-level conditioning within words
    print("\n=== Test 3: Byte-level conditioning within words ===")

    # For "a cat" and "a dog", after seeing "a c" or "a d", the model should be confident about the rest
    # This tests that the byte decoder is properly conditioned on previous bytes

    # Create a special test to check conditional probabilities
    test_conditional = ["a cat", "a cog", "a dog", "a dat"]
    predictions = predict_dataset(test_conditional)
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

    print("\n🎉 All overfitting tests passed! Model shows proper contextual learning and conditioning.")

if __name__ == "__main__":
    test_ilt_overfitting()
