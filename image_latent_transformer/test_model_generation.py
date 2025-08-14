from pathlib import Path

import torch
from image_latent_transformer.ilt_generation import ImageLatentTransformerForTextGeneration
from image_latent_transformer.test_model import setup_model, make_dataset, dataset_to_batch


def test_model_generation():
    """Test that the model can generate expected sequences from trained_model state."""

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
    
    # Load the trained model state
    generation_model.eval()

    def predict_texts(texts: list[str]):
        """Predict texts using the generation model."""
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

    # Test that generation of a batch does not interfere between texts
    batch_1 = predict_texts(["a", "a cat", ""])
    batch_2 = predict_texts(["a", "b"])
    assert batch_1[0] == batch_2[0], "Batch generation should not interfere between texts"


if __name__ == "__main__":
    test_model_generation()