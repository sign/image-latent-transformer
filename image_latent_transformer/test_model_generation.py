from pathlib import Path

import pytest
import torch

from image_latent_transformer.ilt_generation import ImageLatentTransformerForTextGeneration
from image_latent_transformer.test_model import setup_model, make_dataset, dataset_to_batch


@pytest.fixture(scope="module")
def generation_model_setup():
    """Setup the generation model with trained weights."""
    trained_model_path = Path(__file__).parent / "trained_model"

    # Setup the base model
    model, image_processor, tokenizer, collator = setup_model()
    model.load_state_dict(torch.load(trained_model_path))

    # Create the generation model from the base model
    generation_model = ImageLatentTransformerForTextGeneration(
        image_encoder=model.image_encoder,
        bytes_encoder=model.bytes_encoder,
        latent_transformer=model.latent_transformer,
        bytes_decoder=model.bytes_decoder
    )

    # Set to eval mode
    generation_model.eval()

    return generation_model, image_processor, tokenizer, collator


def predict_texts(texts: list[str], generation_model, image_processor, tokenizer, collator):
    """Helper function to predict texts using the generation model."""

    dataset = make_dataset(texts, image_processor, tokenizer)
    batch = dataset_to_batch(generation_model, collator, dataset)

    with torch.no_grad():
        outputs = generation_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            input_pixels=batch["input_pixels"],
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_generated_words=5
        )

    for text, output in zip(texts, outputs):
        print(f"Generated for '{text}': {output}")
    return outputs


def test_batch_interference(generation_model_setup):
    """Test that generation of a batch does not interfere between texts."""
    generation_model, image_processor, tokenizer, collator = generation_model_setup

    generation_model.image_encoder = None # TODO: turn it back on once https://github.com/sign/image-latent-transformer/issues/1 is fixed

    print("\n=== Testing batch interference ===")
    batch_1 = predict_texts(["a", "a cat", ""], generation_model, image_processor, tokenizer, collator)
    batch_2 = predict_texts(["a", "b", "a_long_word"], generation_model, image_processor, tokenizer, collator)
    batch_3 = predict_texts(["a", "a"], generation_model, image_processor, tokenizer, collator)

    print(f"\nBatch 1 result for 'a': '{batch_1[0]}'")
    print(f"Batch 2 result for 'a': '{batch_2[0]}'")
    print(f"Batch 3 result for 'a': '{batch_3[0]}'")

    # Also test individual generation
    single_1 = predict_texts(["a"], generation_model, image_processor, tokenizer, collator)
    single_2 = predict_texts(["b"], generation_model, image_processor, tokenizer, collator)
    print(f"\nSingle result for 'a': '{single_1[0]}'")
    print(f"Single result for 'b': '{single_2[0]}'")

    # All results for 'a' should be the same
    all_a_results = [batch_1[0], batch_2[0], batch_3[0], batch_3[1], single_1[0]]
    print(f"\nAll results for 'a': {all_a_results}")

    # Check that all occurrences of "a" generate the same output
    assert all(result == single_1[0] for result in all_a_results), \
        f"Not all 'a' generations are equal. Expected all to be '{single_1[0]}', but got: {all_a_results}"

    print("✅ Generation test passed - no batch interference detected")


def test_same_text_in_batch(generation_model_setup):
    """Test that same text in batch returns same output."""
    generation_model, image_processor, tokenizer, collator = generation_model_setup

    print("\n=== Testing same text in batch returns same output ===")

    # Test with 4 times "a"
    batch_a = predict_texts(["a", "a", "a", "a"], generation_model, image_processor, tokenizer, collator)
    print(f"\nBatch with 4x 'a': {batch_a}")
    assert all(
        result == batch_a[0] for result in batch_a), f"Same text 'a' in batch produced different outputs: {batch_a}"
    print("✅ All 'a' inputs in batch produced same output")

    # Test with 4 times "b"
    batch_b = predict_texts(["b", "b", "b", "b"], generation_model, image_processor, tokenizer, collator)
    print(f"\nBatch with 4x 'b': {batch_b}")
    assert all(
        result == batch_b[0] for result in batch_b), f"Same text 'b' in batch produced different outputs: {batch_b}"
    print("✅ All 'b' inputs in batch produced same output")

    print("✅ Test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
