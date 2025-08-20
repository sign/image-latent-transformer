from pathlib import Path

from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.processor import TextImageProcessor


def sample(model_path: Path):
    last_checkpoint = get_last_checkpoint(model_path)
    model = ImageLatentTransformer.from_pretrained(last_checkpoint)
    processor = TextImageProcessor.from_pretrained(model_path)

    texts = [
        "The",
        "The game",
        "The game began",
        "The game began development in 2010",
        "Barack Obama is",
        "Helsinki is the capital of",
    ]

    inputs = processor(texts, collated=True)

    outputs = model.generate(
        input_pixels=inputs["input_pixels"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        max_generated_words=32,
        max_word_length=10,
        bytes_generation_config=GenerationConfig(num_beams=2)  # Sample with beam search, for example
    )
    for text, output in zip(texts, outputs):
        print(f"Generated for '{text}': {output}")


if __name__ == "__main__":
    model_path = Path(__file__).parent / "output"
    sample(model_path)
