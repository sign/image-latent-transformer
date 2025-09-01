from itertools import chain
from typing import Union

import torch
from transformers import AutoModelForImageClassification

from image_latent_transformer.collator import stack_pad_tensors
from image_latent_transformer.vision_utils import image_encoder_size, pool_hidden_dim, encode_images as utils_encode_images

ImagesNestedList = Union[list[list[torch.Tensor]], list[torch.nested.Tensor]]

def encode_images(image_encoder: AutoModelForImageClassification,
                  input_pixels: ImagesNestedList,
                  device: torch.device = None) -> torch.Tensor:
    """Image encoder should accept variable size images and return embeddings."""
    # Recreate as list of lists if input is a nested tensor
    input_pixels = [[img for img in inner] for inner in input_pixels]

    # Flatten images
    all_images = list(chain.from_iterable(input_pixels))

    _callable = encode_images_sequentially

    # Use batched implementation if all images are of the same size and no padding is needed
    first_image_shape = all_images[0].shape
    all_images_equal_size = all(image.shape == first_image_shape for image in all_images)
    if all_images_equal_size:
        _callable = encode_images_batch

    embeddings = _callable(image_encoder=image_encoder,
                           images=all_images,
                           device=device)

    hidden_size = image_encoder_size(image_encoder)

    # Re-arrange the embeddings to match the original batch structure
    B = len(input_pixels)  # noqa: N806
    L = max(len(inner) for inner in input_pixels)  # noqa: N806
    embeds = torch.zeros(B, L, hidden_size, device=device)

    # Vectorized split and assignment
    split_sizes = [len(inner) for inner in input_pixels]
    split_embeddings = torch.split(embeddings, split_sizes, dim=0)
    for i, split_embed in enumerate(split_embeddings):
        embeds[i, :len(split_embed)] = split_embed

    return embeds

def encode_images_batch(image_encoder: AutoModelForImageClassification,
                        images: list[torch.Tensor],
                        device: torch.device = None) -> torch.Tensor:
    input_pixels = stack_pad_tensors(images)
    if device is not None:
        input_pixels = input_pixels.to(device)

    # Encode images using the image encoder
    return utils_encode_images(image_encoder, input_pixels)


def encode_images_sequentially(image_encoder: AutoModelForImageClassification,
                               images: list[torch.Tensor],
                               device: torch.device = None) -> torch.Tensor:
    if device is not None:
        images = [image.to(device) for image in images]

    # Encode images using the image encoder
    encoded_images = [utils_encode_images(image_encoder, image.unsqueeze(0)) for image in images]

    return torch.cat(encoded_images, dim=0)
