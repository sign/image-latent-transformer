import torch


def stack_pad_tensors(tensors, pad_value=0):
    """
    Vibe coded:
    Generic collate function that automatically pads mismatched dimensions.
    For each tensor field in the batch, finds dimensions that don't match
    and pads them with zeros to the maximum size.
    """
    # Early return if empty
    if len(tensors) == 0:
        return tensors

    # Early return if not tensors
    if not hasattr(tensors[0], 'shape'):
        return tensors

    # Early return if single tensor
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)

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


def collate_fn(batch: list, pad_value=0):
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
