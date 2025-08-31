import inspect
from functools import cache
from itertools import chain
from typing import Optional, Union

import torch
from transformers import AutoModelForImageClassification

from image_latent_transformer.collator import stack_pad_tensors

ImagesNestedList = Union[list[list[torch.Tensor]], list[torch.nested.Tensor]]


class UnknownImageEncoderError(ValueError):
    def __init__(self):
        super().__init__("Image encoder does not have a valid hidden size configuration.")


@cache
def image_encoder_size(image_encoder: Optional[AutoModelForImageClassification]) -> int:
    if image_encoder is None:
        return 0

    config = getattr(image_encoder, 'config', {})
    if hasattr(config, 'vision_config'):
        config = config.vision_config

    if hasattr(config, 'hidden_size'):
        return config.hidden_size

    # https://huggingface.co/docs/transformers/model_doc/mobilevit#transformers.MobileViTModel
    # If expand_output, the model will apply an additional 1x1 convolution to expand the output channels
    # from config.neck_hidden_sizes[5] to config.neck_hidden_sizes[6].
    if hasattr(config, 'neck_hidden_sizes'):
        if getattr(image_encoder, 'expand_output', False):
            return config.neck_hidden_sizes[-1]
        return config.neck_hidden_sizes[-2]

    if hasattr(config, 'hidden_sizes'):
        return config.hidden_sizes[-1]

    raise UnknownImageEncoderError()


@cache
def accepts(func, param_name: str) -> bool:
    sig = inspect.signature(func)
    return (
            param_name in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )


@cache
def model_args_dict(model: AutoModelForImageClassification) -> dict:
    """Get model arguments based on its capabilities."""
    args = {"output_hidden_states": True}
    if accepts(model.forward, 'interpolate_pos_encoding'):
        args['interpolate_pos_encoding'] = True
    return args


def encode_images(image_encoder: AutoModelForImageClassification,
                  input_pixels: ImagesNestedList,
                  device: torch.device = None) -> torch.Tensor:
    """Image encoder should accept variable size images and return embeddings."""
    model_args = model_args_dict(image_encoder)

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
                           model_args=model_args,
                           device=device)

    hidden_size = image_encoder_size(image_encoder)

    # Re-arrange the embeddings to match the original batch structure
    B = len(input_pixels)  # noqa: N806
    L = max(len(inner) for inner in input_pixels)  # noqa: N806
    embeds = torch.zeros(B, L, hidden_size, device=device)

    idx = 0
    for i, inner_images in enumerate(input_pixels):
        num_images = len(inner_images)
        embeds[i, range(num_images)] = embeddings[idx:idx + num_images]
        idx += num_images

    return embeds


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
                        images: list[torch.Tensor],
                        model_args: dict,
                        device: torch.device = None) -> torch.Tensor:
    input_pixels = stack_pad_tensors(images)
    if device is not None:
        input_pixels = input_pixels.to(device)

    # Encode images using the image encoder
    encoded_image = image_encoder(input_pixels, **model_args)
    image_embeds = encoded_image.hidden_states[-1]  # Use the last hidden state

    # Pool channels/patches dimension (e.g., mean pooling)
    hidden_size = image_encoder_size(image_encoder)
    image_embeds = pool_hidden_dim(image_embeds, hidden_size)
    return image_embeds


def encode_images_sequentially(image_encoder: AutoModelForImageClassification,
                               images: list[torch.Tensor],
                               model_args: dict,
                               device: torch.device = None) -> torch.Tensor:
    if device is not None:
        images = [image.to(device) for image in images]

    # Encode images using the image encoder
    encoded_images = [image_encoder(image.unsqueeze(0), **model_args) for image in images]
    hidden_states = [e.hidden_states[-1] for e in encoded_images]

    hidden_size = image_encoder_size(image_encoder)
    hidden_states = [pool_hidden_dim(state, hidden_size) for state in hidden_states]

    return torch.cat(hidden_states, dim=0)
