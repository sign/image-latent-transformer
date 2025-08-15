import torch


def stack_pad_tensors(tensors, pad_value=0):
    """
    Vibe coded:
    Generic collate function that automatically pads mismatched dimensions.
    For each tensor field in the batch, finds dimensions that don't match
    and pads them with zeros to the maximum size.
    """

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


def collate_images(images: list[list[torch.Tensor]],
                   patch_size: int,
                   pad_value=0) -> tuple[torch.Tensor, torch.Tensor]:
    images = [stack_pad_tensors(images, pad_value=pad_value) for images in images]
    pixel_values = stack_pad_tensors(images, pad_value=pad_value)

    # Create a patches mask
    B, L, C, H, W = pixel_values.shape # noqa: N806
    num_patches_height = H // patch_size
    num_patches_width = W // patch_size
    bool_masked_pos = torch.zeros(B, L, num_patches_height, num_patches_width)
    for i, ims in enumerate(images):
        for j, im in enumerate(ims):
            c, h, w = im.shape  # noqa: N806
            # Create a boolean mask for the patches
            bool_masked_pos[i, j, :h//patch_size, :w//patch_size] = 1

    # Linearize the boolean mask
    bool_masked_pos = bool_masked_pos.reshape(B, L, -1)

    return pixel_values, bool_masked_pos


def collate_fn(batch, image_patch_size: int, pad_value=0):
    if not batch:
        return batch

    # Get all keys from the first item
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        tensors = [item[key] for item in batch]

        if key == "input_pixels":
            pixel_values, pixel_mask = collate_images(tensors, patch_size=image_patch_size, pad_value=pad_value)
            collated[key] = pixel_values
            collated[key + "_mask"] = pixel_mask
        else:
            collated[key] = stack_pad_tensors(tensors, pad_value=pad_value)

    return collated
