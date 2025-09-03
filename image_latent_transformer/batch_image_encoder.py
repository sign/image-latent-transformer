from itertools import chain
from typing import Union

import torch
from transformers import AutoModelForImageClassification
from transformers.image_transforms import group_images_by_shape, reorder_images

from image_latent_transformer.collator import stack_pad_tensors
from image_latent_transformer.vision_utils import encode_images as utils_encode_images
from image_latent_transformer.vision_utils import image_encoder_size


@torch.inference_mode()
def crop_nested_pixels(pixels: torch.Tensor) -> list[list[torch.Tensor]]:
    """
    pixels: (B, L, C, H, W) with zero-padding around the true content.
    Returns nested list of per-(b,l) crops with padding removed.
    """
    B, L, C, H, W = pixels.shape  # noqa: N806
    # Nonzero mask per (H,W) ignoring channels
    m = pixels.ne(0).any(dim=2).reshape(B * L, H, W)  # (N,H,W), bool
    non_empty = m.any(dim=(1, 2))  # (N,)
    if not bool(non_empty.any()):
        return [[] for _ in range(B)]

    m_sel = m[non_empty]  # (M,H,W)
    # Row/col occupancy
    row_any = m_sel.any(dim=2)  # (M,H)
    col_any = m_sel.any(dim=1)  # (M,W)

    # Tight bounds via argmax trick (first/last True)
    # Cast to int to avoid ambiguity warnings; works on CPU/GPU
    y1 = row_any.int().argmax(dim=1)  # (M,)
    y2 = (H - 1) - row_any.flip(dims=[1]).int().argmax(dim=1)  # (M,)
    x1 = col_any.int().argmax(dim=1)  # (M,)
    x2 = (W - 1) - col_any.flip(dims=[1]).int().argmax(dim=1)  # (M,)

    # Map flattened indices back to (b,l)
    idxs = non_empty.nonzero(as_tuple=False).squeeze(1)  # (M,)
    b_idx = torch.div(idxs, L, rounding_mode='floor')  # (M,)
    l_idx = idxs.remainder(L)  # (M,)

    crops: list[list[torch.Tensor]] = [[] for _ in range(B)]
    # Ragged outputs => loop over M non-empty positions
    for i in range(idxs.numel()):
        b = int(b_idx[i])  # noqa: N806
        l = int(l_idx[i])  # noqa: N806,E741
        rr0 = int(y1[i])
        rr1 = int(y2[i])
        cc0 = int(x1[i])
        cc1 = int(x2[i])
        crops[b].append(pixels[b, l, :, rr0:rr1 + 1, cc0:cc1 + 1].contiguous())
    return crops


def encode_images(image_encoder: AutoModelForImageClassification,
                  input_pixels: torch.Tensor,
                  device: torch.device = None) -> torch.Tensor:
    """Image encoder should accept variable size images and return consistent embeddings."""
    B, L, *_ = input_pixels.shape  # noqa: N806

    # Recreate as list of lists if input is a nested tensor, cropping padding from each image
    input_pixels = crop_nested_pixels(input_pixels)

    # Flatten images
    all_images = list(chain.from_iterable(input_pixels))

    # Re-arrange the embeddings to match the original batch structure
    embeddings = encode_images_group(image_encoder=image_encoder,
                                     images=all_images,
                                     device=device)

    hidden_size = image_encoder_size(image_encoder)

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
    rearranged_images = reorder_images(encoded_groups, grouped_images_index)

    return torch.stack(rearranged_images)
