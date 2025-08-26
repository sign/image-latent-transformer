import inspect
from functools import cache
from typing import Optional

import torch
from transformers import AutoModelForImageClassification


def stack_pad_tensors(tensors, pad_value=0):
    """
    Vibe coded:
    Generic collate function that automatically pads mismatched dimensions.
    For each tensor field in the batch, finds dimensions that don't match
    and pads them with zeros to the maximum size.
    """

    # Early return if not tensors
    if not hasattr(tensors[0], 'shape'):
        return tensors

    # Check if all tensors have the same shape
    shapes = [tensor.shape for tensor in tensors]
    if len(set(shapes)) == 1:
        # All shapes are the same, just stack
        return torch.stack(tensors)
    else:
        # Find maximum size for each dimension
        max_shape = []
        ndim = max(len(shape) for shape in shapes)

        for dim in range(ndim):
            max_size = max(shape[dim] if dim < len(shape) else 1
                           for shape in shapes)
            max_shape.append(max_size)

        # Pad each tensor to max_shape
        padded_tensors = []
        for tensor in tensors:
            # Calculate padding needed for each dimension
            padding = []
            for dim in reversed(range(ndim)):  # padding goes from last dim to first
                if dim < len(tensor.shape):
                    pad_size = max_shape[dim] - tensor.shape[dim]
                    padding.extend([0, pad_size])
                else:
                    padding.extend([0, max_shape[dim]])

            if padding:
                padded_tensor = torch.nn.functional.pad(tensor, padding, value=pad_value)
            else:
                padded_tensor = tensor

            # If tensor has fewer dimensions than max, add singleton dimensions
            while len(padded_tensor.shape) < ndim:
                padded_tensor = padded_tensor.unsqueeze(0)

            padded_tensors.append(padded_tensor)

        return torch.stack(padded_tensors)


def collate_images(images: list[list[torch.Tensor]], pad_value=0) -> torch.Tensor:
    images = [stack_pad_tensors(images, pad_value=pad_value) for images in images]
    return stack_pad_tensors(images, pad_value=pad_value)


def collate_fn(batch, pad_value=0):
    if not batch:
        return batch

    # Get all keys from the first item
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        tensors = [item[key] for item in batch]

        if key == "input_pixels":
            collated[key] = tensors  # Not collated
        else:
            collated[key] = stack_pad_tensors(tensors, pad_value=pad_value)

    return collated


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
    # When loading the model, it does not expand_output=True.
    if hasattr(config, 'neck_hidden_sizes'):
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
