import torch
import logging
from transformers import AutoImageProcessor, AutoModelForImageClassification, ByT5Tokenizer, AutoModelForCausalLM

from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.renderer import render_texts

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Image Encoder
image_model_name = "microsoft/swinv2-tiny-patch4-window16-256"
image_processor = AutoImageProcessor.from_pretrained(image_model_name, use_fast=True)
image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
# Remove the classifier head for feature extraction, to save on parameters
image_model.classifier = torch.nn.Identity()

# Small Language Model
tokenizer = ByT5Tokenizer(extra_ids=5)
byte_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
byte_lm.resize_token_embeddings(len(tokenizer))

# Latent Transformer
latent_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
latent_lm.resize_token_embeddings(0)

# Combine the models into a single Image Latent Transformer
model = ImageLatentTransformer(
    image_encoder=image_model,
    latent_transformer=latent_lm,
    bytes_decoder=byte_lm
)

example_string = "Built with HuggingFace 🤗!"
words = example_string.split()
logger.debug("Words: %s", words)

tokenized_words = tokenizer(words, return_tensors="pt", padding=True, max_length=32, truncation=True)
input_ids = tokenized_words.input_ids
attention_mask = tokenized_words.attention_mask
logger.debug("Input IDs: %s", input_ids)
logger.debug("Attention Mask: %s", attention_mask)

line_height, text_lines = render_texts(words)
image = image_processor(text_lines, return_tensors="pt", do_center_crop=False, do_resize=False).pixel_values
logger.debug("Image shape: %s", image.shape)

# Deconstruct the image, to one image per line
images = image.squeeze(0)\
    .unflatten(1, (len(words), line_height))\
    .permute(1, 0, 2, 3)  # [3, 4, 32, 128] -> [4, 3, 32, 128]

logger.debug("Images: %s", images.shape)

# Mimic a batch
input_ids = input_ids.unsqueeze(0)  # Add batch dimension
attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
images = images.unsqueeze(0)  # Add batch dimension

# Forward pass through the model
with torch.no_grad():
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_pixels=images
    )
    logger.debug("logits: %s", logits.shape)