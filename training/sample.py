from pathlib import Path

import torch
from transformers import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from image_latent_transformer.model import ImageLatentTransformerForCausalLM
from image_latent_transformer.processor import TextImageProcessor


@torch.no_grad()
def sample(model_path: Path):
    last_checkpoint = get_last_checkpoint(model_path)
    model = ImageLatentTransformerForCausalLM.from_pretrained(last_checkpoint)
    processor = TextImageProcessor.from_pretrained(model_path)

    model.eval()

    texts = [
        "<en>",  # Generate the english and hebrew
        # Texts from validation set
        "<en> What's wrong? <he>",
        "<en> YOu dOn't know the half Of it. <he>",  # look! mixed case
        "<en> - No, just said that you were acting... aggressive. <he>",
        "<en> -l'm a deputy. <he>",  # look! "l" instead of "I", no space.
        "<en> Well, the good news is Joe wasn't cheating on you. <he>",
        "<en> - Mm-hmm. No wonder women won't flirt with me. <he>",
        "<en> You understand you'll be working with your father. <he>",
        "<en> You wanted freedom, you got it. <he>",
        # Not from validation set, just wanted to try a long named entity
        "<en> Alexander Hamilton <he>",
    ]

    inputs = processor(texts, collated=True, packed=False)

    outputs = model.generate(
        input_pixels=inputs["input_pixels"],
        input_ids=inputs["input_ids"],
        input_attention_mask=inputs["input_attention_mask"],
        processor=processor,
        max_generated_words=32,
        max_word_length=10,
        bytes_generation_config=GenerationConfig(num_beams=2)  # Sample with beam search, for example
    )
    for text, output in zip(texts, outputs):
        print(f"Generated for '{text}': {output}")


if __name__ == "__main__":
    model_path = Path(__file__).parent / "output"
    sample(model_path)
