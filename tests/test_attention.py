import pytest
import torch

from image_latent_transformer.attention import (
    get_attention_mask_for_packed_sequence,
    get_position_ids_for_packed_sequence,
)


def test_get_attention_mask_for_packed_sequence_single_sequence():
    seq_lengths = [3]
    mask = get_attention_mask_for_packed_sequence(seq_lengths)

    expected = torch.tensor([[
        [True, False, False],
        [True, True, False],
        [True, True, True]
    ]])

    assert torch.equal(mask, expected)
    assert mask.shape == (1, 3, 3)


def test_get_attention_mask_for_packed_sequence_two_sequences():
    seq_lengths = [2, 2]
    mask = get_attention_mask_for_packed_sequence(seq_lengths)

    expected = torch.tensor([[
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True]
    ]])

    assert torch.equal(mask, expected)
    assert mask.shape == (1, 4, 4)


def test_get_position_ids_for_packed_sequence_single_sequence():
    seq_lengths = [3]
    position_ids = get_position_ids_for_packed_sequence(seq_lengths)

    expected = torch.tensor([0, 1, 2])

    assert torch.equal(position_ids, expected)
    assert position_ids.shape == (3,)


def test_get_position_ids_for_packed_sequence_two_sequences():
    seq_lengths = [2, 2]
    position_ids = get_position_ids_for_packed_sequence(seq_lengths)

    expected = torch.tensor([0, 1, 0, 1])

    assert torch.equal(position_ids, expected)
    assert position_ids.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
