import pytest
import torch

from image_latent_transformer.collator import stack_pad_tensors_fast, stack_pad_tensors_slow


def test_stack_pad_tensors_consistency():
    """Test that stack_pad_tensors_fast and stack_pad_tensors_slow produce the same output."""
    a = torch.randn(2, 3, 4)
    b = torch.randn(3, 5, 4)
    c = torch.randn(1, 2, 3)
    d = torch.randn(2, 3, 1)

    tensors = [a, b, c, d]
    pad_value = -1

    result_fast = stack_pad_tensors_fast(tensors, pad_value=pad_value)
    result_slow = stack_pad_tensors_slow(tensors, pad_value=pad_value)

    assert torch.equal(result_fast, result_slow), "Fast and slow implementations should produce identical results"
    assert result_fast.shape == result_slow.shape, "Output shapes should be identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
