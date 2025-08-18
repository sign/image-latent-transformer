from pathlib import Path

from transformers import GenerationConfig

from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.processor import TextImageProcessor

MODEL_PATH = Path(__file__).parent / "output"

model = ImageLatentTransformer.from_pretrained(MODEL_PATH / "checkpoint-1500")
processor = TextImageProcessor.from_pretrained(MODEL_PATH)

texts = ["Barack Obama is the"]

inputs = processor(texts)

outputs = model.generate(
    input_pixels=inputs["input_pixels"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    tokenizer=processor.tokenizer,
    image_processor=processor.image_processor,
    max_generated_words=50,
    max_word_length=10,
    bytes_generation_config=GenerationConfig(num_beams=2)  # Sample with beam search, for example
)
for text, output in zip(texts, outputs):
    print(f"Generated for '{text}': {output}")
