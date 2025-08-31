from functools import cache

import pytest
import torch
from transformers import AutoConfig, AutoModelForImageClassification

from image_latent_transformer.batch_image_encoder import (
    encode_images,
    encode_images_batch,
    encode_images_sequentially,
    image_encoder_size,
    model_args_dict,
)

MODELS = {
    "WinKawaks/vit-tiny-patch16-224": 192,
    "microsoft/swinv2-tiny-patch4-window16-256": 768,
    "google/vit-base-patch16-224": 768,
    "microsoft/resnet-18": 512,
    "apple/mobilevit-xx-small": 80,
    # TODO: support once huggingface releases a new version (config loads, but no AutoModelForImageClassification)
    # "facebook/dinov3-vits16-pretrain-lvd1689m": 0,
    # "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": 0,
}

MODEL_NAMES = list(MODELS.keys())


@cache
def image_encoder(model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_config(config)
    model.eval()
    return model


def create_random_image(height, width, channels=3):
    """Create a random image tensor."""
    return torch.randn(channels, height, width)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_image_encoder_size(model_name):
    """Test that image_encoder_size returns the expected hidden size for each model."""
    model = image_encoder(model_name)
    expected_size = MODELS[model_name]
    actual_size = image_encoder_size(model)
    assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size} for {model_name}"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_single_image(model_name):
    """Test encode_images with a single image."""
    model = image_encoder(model_name)

    # Single batch with single image
    images = [[create_random_image(64, 64)]]

    embeddings = encode_images(model, images)

    assert embeddings.shape == (1, 1, embeddings.shape[2])


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_deterministic(model_name):
    model = image_encoder(model_name)

    images = [[create_random_image(64, 64)]]

    embeddings1 = encode_images(model, images)
    embeddings2 = encode_images(model, images)

    # Results should be identical
    assert torch.equal(embeddings1, embeddings2)


def test_encode_nested_tensor():
    model = image_encoder("WinKawaks/vit-tiny-patch16-224")

    tensors = [create_random_image(64, 64), create_random_image(64, 64)]
    image_input = torch.nested.nested_tensor(tensors)
    input_pixels = [image_input, image_input]

    embedding = encode_images(model, input_pixels)

    assert embedding.shape == (2, 2, 192)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_batched_or_sequential(model_name):
    """Make sure the batch implementation and the sequential implementation return the same result"""
    model = image_encoder(model_name)

    images = [create_random_image(64, 64), create_random_image(64, 64)]

    model_args = model_args_dict(model)
    embeddings1 = encode_images_batch(model, images, model_args)
    embeddings2 = encode_images_sequentially(model, images, model_args)

    assert embeddings1.shape == embeddings2.shape

    assert torch.equal(embeddings1, embeddings2)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_basic(model_name):
    """Test basic functionality of encode_images with different image sizes."""
    model = image_encoder(model_name)

    # Create images of different sizes
    images = [
        [create_random_image(32, 32), create_random_image(64, 64)],
        [create_random_image(32, 64), create_random_image(128, 32)]
    ]

    # Test encoding
    embeddings = encode_images(model, images)

    # Check output shape
    assert embeddings.shape[0] == 2  # batch size
    assert embeddings.shape[1] == 2  # sequence length (number of images per batch)
    assert embeddings.shape[2] > 0  # embedding dimension


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("image_size", [(32, 32), (32, 64), (32, 128), (64, 128)])
def test_encode_images_different_models_and_sizes(model_name, image_size):
    """Test encode_images with different model types and image sizes."""
    model = image_encoder(model_name)
    height, width = image_size

    # Create test image of specified size
    images = [[create_random_image(height, width)]]

    embeddings = encode_images(model, images)

    assert embeddings.shape[0] == 1  # batch size
    assert embeddings.shape[1] == 1  # sequence length
    assert embeddings.shape[2] > 0  # embedding dimension


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_batch_vs_individual(model_name):
    """Test that batch processing gives same results as individual processing for different models."""
    model = image_encoder(model_name)

    # Create test images of different sizes
    img1 = create_random_image(32, 32)
    img2 = create_random_image(64, 64)
    img3 = create_random_image(32, 128)
    img4 = create_random_image(64, 128)

    # Batch processing - all images at once
    batch_images = [[img1, img2, img3, img4]]
    batch_embeddings = encode_images(model, batch_images)

    # Individual processing - each image separately
    individual_embeddings = []
    for img in [img1, img2, img3, img4]:
        img_embeddings = encode_images(model, [[img]])
        individual_embeddings.append(img_embeddings.squeeze(0))

    for i, (individual_embedding, batch_embedding) in enumerate(zip(individual_embeddings, batch_embeddings[0])):
        print(individual_embedding.shape, batch_embedding.shape)
        # Compare results (allowing for small numerical differences)
        assert torch.allclose(batch_embedding, individual_embedding, atol=1e-3), \
            f"Batch vs individual results ({i}) differ for model {model_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
