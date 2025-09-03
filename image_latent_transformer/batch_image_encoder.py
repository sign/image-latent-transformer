from itertools import chain
from typing import Union

import torch
from transformers import AutoModelForImageClassification
from transformers.image_transforms import group_images_by_shape, reorder_images

from image_latent_transformer.collator import stack_pad_tensors
from image_latent_transformer.vision_utils import encode_images as utils_encode_images
from image_latent_transformer.vision_utils import image_encoder_size


def encode_images(image_encoder: AutoModelForImageClassification,
                  input_images: torch.Tensor,
                  input_images_dimensions: torch.Tensor,
                  device: torch.device = None) -> torch.Tensor:
    """Image encoder should accept variable size images and return consistent embeddings."""
    B, L, *_ = input_images.shape  # noqa: N806

    # Recreate as list of lists if input is a nested tensor, cropping padding from each image
    nested_images = [
        [img[:, :h, :w] for img, (h, w) in zip(images, dims) if h > 0 and w > 0]
        for images, dims in zip(input_images, input_images_dimensions)
    ]

    # Flatten images
    all_images = list(chain.from_iterable(nested_images))

    # Re-arrange the embeddings to match the original batch structure
    embeddings = encode_images_group(image_encoder=image_encoder,
                                     images=all_images,
                                     device=device)

    hidden_size = image_encoder_size(image_encoder)

    embeds = torch.zeros(B, L, hidden_size, device=device)

    # Vectorized split and assignment
    split_sizes = [len(inner) for inner in nested_images]
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
    rearranged_images = reorder_images(encoded_groups, grouped_images_index)

    return torch.stack(rearranged_images)
