import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, ByT5Tokenizer, AutoModelForCausalLM

from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.dataset import TextImageDataset

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
    bytes_encoder=byte_lm,
    latent_transformer=latent_lm,
    bytes_decoder=byte_lm
)

# Create dataset and dataloader
example_texts = ["Built with HuggingFace 🤗!"]
dataset = TextImageDataset(
    texts_dataset=example_texts,
    image_processor=image_processor,
    tokenizer=tokenizer,
    max_seq_length=128,
    max_word_length=32
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Get a batch from the dataloader
batch = next(iter(dataloader))
for key, value in batch.items():
    print(f"{key}: {value.shape}")

# Forward pass through the model
with torch.no_grad():
    logits = model(**batch)
    print(f"Logits shape: {logits.shape}")