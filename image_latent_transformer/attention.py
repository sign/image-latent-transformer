import torch


def get_attention_mask_for_packed_sequence(seq_lengths: list[int], words: list[str] = None) -> torch.Tensor:
    """
    Returns a 3D attention mask for a packed sequence. (1, seq_len, seq_len)
    The first dimension represents the head dimension, which is set to 1 for broadcasting.
    """
    total_length = sum(seq_lengths)

    mask = torch.zeros((1, total_length, total_length), dtype=torch.bool)

    current_position = 0
    for length in seq_lengths:
        tril = torch.tril(torch.ones((length, length), dtype=torch.bool))
        mask[0, current_position:current_position + length, current_position:current_position + length] = tril
        current_position += length

    # TODO: use words to find instruction tokens to attend between. something like <attention> text ... </attention>
    #      similar to prefixLM or MAS https://github.com/sign/image-latent-transformer/issues/9
    #      just in a generic way

    return mask


def get_position_ids_for_packed_sequence(seq_lengths: list[int]) -> torch.Tensor:
    total_length = sum(seq_lengths)

    positions = torch.zeros(total_length, dtype=torch.long)

    current_position = 0
    for length in seq_lengths:
        positions[current_position:current_position + length] = torch.arange(length, dtype=torch.long)
        current_position += length

    return positions
