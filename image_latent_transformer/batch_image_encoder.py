from functools import cache
from itertools import chain

import torch
from transformers import AutoModelForImageClassification

from image_latent_transformer.utils import accepts, collate_images, image_encoder_size


@cache
def model_args_dict(model: AutoModelForImageClassification) -> dict:
    """Get model arguments based on its capabilities."""
    args = {"output_hidden_states": True}
    if accepts(model.forward, 'interpolate_pos_encoding'):
        args['interpolate_pos_encoding'] = True
    return args


def encode_images(image_encoder: AutoModelForImageClassification,
                  input_pixels: list[list[torch.Tensor]],
                  device: torch.device = None) -> torch.Tensor:
    """Image encoder should accept variable size images and return embeddings."""
    model_args = model_args_dict(image_encoder)

    _callable = encode_images_sequentially

    # Use batched implementation if all images are of the same size and no padding is needed
    first_image_shape = input_pixels[0][0].shape
    all_images_equal_size = all(image.shape == first_image_shape for images in input_pixels for image in images)
    if all_images_equal_size:
        _callable = encode_images_batch

    return _callable(image_encoder=image_encoder,
                     input_pixels=input_pixels,
                     model_args=model_args,
                     device=device)

def pool_hidden_dim(tensor: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """
    Generic implementation to pool the hidden dimension of a tensor.
    This works both for ViT-like models (e.g. [B, N, H])
    and ConvNet-like models (e.g. [B, H, W, C] or [B, C, H, W]).
    """
    hidden_dim = next(i for i, s in enumerate(tensor.shape) if s == hidden_size and i != 0)
    non_hidden_dims = tuple(i for i in range(len(tensor.shape)) if i != hidden_dim and i != 0)
    return tensor.mean(dim=non_hidden_dims)

def encode_images_batch(image_encoder: AutoModelForImageClassification,
                        input_pixels: list[list[torch.Tensor]],
                        model_args: dict,
                        device: torch.device = None) -> torch.Tensor:
    input_pixels = collate_images(input_pixels)
    if device is not None:
        input_pixels = input_pixels.to(device)

    B, L, C, H, W = input_pixels.shape  # noqa: N806

    # Flatten batch and length dimensions
    input_pixels = input_pixels.view(B * L, C, H, W)

    # Encode images using the image encoder
    encoded_image = image_encoder(input_pixels, **model_args)
    image_embeds = encoded_image.hidden_states[-1]  # Use the last hidden state

    # Pool channels/patches dimension (e.g., mean pooling)
    hidden_size = image_encoder_size(image_encoder)
    image_embeds = pool_hidden_dim(image_embeds, hidden_size)
    return image_embeds.view(B, L, -1)

def encode_images_sequentially(image_encoder: AutoModelForImageClassification,
                               input_pixels: list[list[torch.Tensor]],
                               model_args: dict,
                               device: torch.device = None) -> torch.Tensor:
    images = list(chain.from_iterable(input_pixels))
    if device is not None:
        images = [image.to(device) for image in images]

    # Encode images using the image encoder
    encoded_images = [image_encoder(image.unsqueeze(0), **model_args) for image in images]
    hidden_states = [e.hidden_states[-1] for e in encoded_images]

    hidden_size = image_encoder_size(image_encoder)

    # Re-arrange the embeddings to match the original batch structure
    B = len(input_pixels) # noqa: N806
    L = max(len(inner) for inner in input_pixels) # noqa: N806
    embeds = torch.zeros(B, L, hidden_size, device=device)

    idx = 0
    for i, inner_images in enumerate(input_pixels):
        for j, _image in enumerate(inner_images):
            state = hidden_states[idx]
            embeds[i, j] = pool_hidden_dim(state, hidden_size)
            idx += 1

    return embeds
