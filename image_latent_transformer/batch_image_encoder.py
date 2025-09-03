from itertools import chain
from typing import Union

import torch
from transformers import AutoModelForImageClassification
from transformers.image_transforms import group_images_by_shape

from image_latent_transformer.collator import stack_pad_tensors
from image_latent_transformer.vision_utils import encode_images as utils_encode_images
from image_latent_transformer.vision_utils import image_encoder_size

ImagesNestedList = Union[list[list[torch.Tensor]], list[torch.nested.Tensor]]


def encode_images(image_encoder: AutoModelForImageClassification,
                  input_pixels: ImagesNestedList,
                  device: torch.device = None) -> torch.Tensor:
    """Image encoder should accept variable size images and return embeddings."""
    # Recreate as list of lists if input is a nested tensor
    input_pixels = [[img for img in inner] for inner in input_pixels]

    # Flatten images
    all_images = list(chain.from_iterable(input_pixels))

    embeddings = encode_images_group(image_encoder=image_encoder,
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
                        images: Union[list[torch.Tensor], torch.Tensor],
                        device: torch.device = None) -> torch.Tensor:
    if isinstance(images, list):
        images = stack_pad_tensors(images)

    if device is not None:
        images = images.to(device)

    # Encode images using the image encoder
    return utils_encode_images(image_encoder, images)


def encode_images_sequentially(image_encoder: AutoModelForImageClassification,
                               images: list[torch.Tensor],
                               device: torch.device = None) -> torch.Tensor:
    encoded_images = [encode_images_batch(image_encoder, image.unsqueeze(0), device=device) for image in images]
    return torch.cat(encoded_images, dim=0)


def encode_images_group(image_encoder: AutoModelForImageClassification,
                        images: list[torch.Tensor],
                        device: torch.device = None) -> torch.Tensor:
    grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=False)

    # Encode each group separately
    encoded_groups = {size: encode_images_batch(image_encoder=image_encoder, images=group, device=device)
                      for size, group in grouped_images.items()}
    # Re-arrange the encoded images to match the original order
    rearranged_images = [encoded_groups[size][i] for size, i in grouped_images_index.values()]

    return torch.stack(rearranged_images)
