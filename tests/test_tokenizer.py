import tempfile

import pytest
import torch

from image_latent_transformer.tokenizer import UTF8Tokenizer


@pytest.fixture
def tokenizer():
    return UTF8Tokenizer()


def test_basic_tokenization(tokenizer):
    """Test basic tokenization without special tokens."""
    text = "hello"
    result = tokenizer(text, add_special_tokens=False)

    # Expected: [104, 101, 108, 108, 111] (UTF-8 bytes for "hello")
    expected_ids = [104, 101, 108, 108, 111]
    assert result.input_ids == expected_ids
    assert result.attention_mask == [1] * len(expected_ids)


def test_unicode_string(tokenizer):
    """Test tokenization of a unicode string."""
    text = "héllo עמית"
    result = tokenizer(text, add_special_tokens=False)

    decoded = tokenizer.decode(result.input_ids)
    assert decoded == text


def test_tokenization_with_special_tokens(tokenizer):
    """Test tokenization with BOS and EOS tokens."""
    text = "hello"
    result = tokenizer(text, add_special_tokens=True)

    # Expected: [2, 104, 101, 108, 108, 111, 3] (BOS + "hello" + EOS)
    expected_ids = [2, 104, 101, 108, 108, 111, 3]
    assert result.input_ids == expected_ids
    assert result.attention_mask == [1] * len(expected_ids)


def test_torch_method_basic(tokenizer):
    """Test torch method with basic functionality."""
    texts = ["hello", "world"]
    result = tokenizer.torch(texts, add_special_tokens=False)

    # Check shapes and types
    assert isinstance(result.input_ids, list)
    assert isinstance(result.attention_mask, list)
    assert len(result.input_ids) == 2
    assert len(result.attention_mask) == 2

    # Check individual tensors
    for i, text in enumerate(texts):
        expected_ids = [ord(c) for c in text]
        assert result.input_ids[i].tolist() == expected_ids
        assert result.input_ids[i].dtype == torch.long
        assert result.attention_mask[i].dtype == torch.long
        assert result.attention_mask[i].tolist() == [1] * len(expected_ids)


def test_torch_method_with_special_tokens(tokenizer):
    """Test torch method with special tokens."""
    texts = ["hello", "world"]
    result = tokenizer.torch(texts, add_special_tokens=True)

    # Check that BOS and EOS are added
    for i, text in enumerate(texts):
        expected_ids = [2] + [ord(c) for c in text] + [3]  # BOS + text + EOS
        assert result.input_ids[i].tolist() == expected_ids
        assert result.attention_mask[i].tolist() == [1] * len(expected_ids)


def test_torch_method_with_padding(tokenizer):
    """Test torch method with padding enabled."""
    texts = ["hi", "hello world"]  # Different lengths
    result = tokenizer.torch(texts, add_special_tokens=False, padding=True)

    # Check that tensors are padded to same length
    assert isinstance(result.input_ids, torch.Tensor)
    assert isinstance(result.attention_mask, torch.Tensor)

    max_len = max(len(text) for text in texts)
    assert result.input_ids.shape == (2, max_len)
    assert result.attention_mask.shape == (2, max_len)

    # Check padding values
    assert result.input_ids[0, -1] == 0  # Pad token
    assert result.attention_mask[0, -1] == 0  # Padded attention


def test_torch_method_with_truncation(tokenizer):
    """Test torch method with truncation."""
    texts = ["hello world this is a long text"]
    max_length = 5
    result = tokenizer.torch(texts, add_special_tokens=False, truncation=True, max_length=max_length)

    # Check truncation
    assert len(result.input_ids[0]) == max_length
    assert len(result.attention_mask[0]) == max_length

    # Verify truncated content matches first 5 characters
    expected_ids = [ord(c) for c in texts[0][:max_length]]
    assert result.input_ids[0].tolist() == expected_ids


def test_torch_method_special_tokens_with_padding(tokenizer):
    """Test torch method with both special tokens and padding."""
    texts = ["hi", "hello"]
    result = tokenizer.torch(texts, add_special_tokens=True, padding=True)

    # Check dimensions
    assert isinstance(result.input_ids, torch.Tensor)
    assert isinstance(result.attention_mask, torch.Tensor)

    # Both sequences should have BOS + text + EOS, padded to same length
    max_len = max(len(text) + 2 for text in texts)  # +2 for BOS/EOS
    assert result.input_ids.shape == (2, max_len)
    assert result.attention_mask.shape == (2, max_len)

    # Check first sequence: BOS + "hi" + EOS + padding
    expected_first = [2, ord('h'), ord('i'), 3, 0, 0, 0]
    assert result.input_ids[0].tolist() == expected_first
    assert result.attention_mask[0].tolist() == [1, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_comparison_tokenizer_vs_torch_method(add_special_tokens, tokenizer):
    """Test that tokenizer() and torch() methods produce compatible results."""
    texts = ["test string"]

    result1 = tokenizer(texts, add_special_tokens=add_special_tokens, return_tensors="pt")
    result2 = tokenizer.torch(texts, add_special_tokens=add_special_tokens, padding=False)

    assert torch.equal(result1.input_ids[0], result2.input_ids[0])
    assert torch.equal(result1.attention_mask[0], result2.attention_mask[0])


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_comparison_tokenizer_vs_torch_method_max_length(add_special_tokens, tokenizer):
    texts = ["test string"]

    params = dict(
        add_special_tokens=add_special_tokens,
        max_length=1,
        truncation=True
    )
    result1 = tokenizer(texts, return_tensors="pt", **params)
    result2 = tokenizer.torch(texts, **params)

    assert torch.equal(result1.input_ids[0], result2.input_ids[0])
    assert torch.equal(result1.attention_mask[0], result2.attention_mask[0])


def test_comparison_tokenizer_vs_torch_method_multiple_strings(tokenizer):
    texts = ["test string", "shorter"]

    result1 = tokenizer(texts, padding=True, return_tensors="pt")
    result2 = tokenizer.torch(texts, padding=True)

    assert result1.input_ids[0].dtype == result2.input_ids[0].dtype
    assert result1.attention_mask[0].dtype == result2.attention_mask[0].dtype

    assert torch.equal(result1.input_ids[0], result2.input_ids[0])
    assert torch.equal(result1.attention_mask[0], result2.attention_mask[0])


def test_empty_string(tokenizer):
    """Test tokenization of empty string."""
    result = tokenizer("", add_special_tokens=False)
    assert result.input_ids == []
    assert result.attention_mask == []

    result_with_special = tokenizer("", add_special_tokens=True)
    assert result_with_special.input_ids == [2, 3]  # BOS + EOS
    assert result_with_special.attention_mask == [1, 1]


def test_special_characters(tokenizer):
    """Test tokenization with special characters and unicode."""
    text = "hello\nworld\t!"
    result = tokenizer(text, add_special_tokens=False)

    expected_ids = [ord(c) for c in text]
    assert result.input_ids == expected_ids
    assert result.attention_mask == [1] * len(expected_ids)


def test_dtype_consistency(tokenizer):
    """Test that dtypes are consistent across different methods."""
    texts = ["hello", "world"]
    result = tokenizer.torch(texts, padding=True)

    assert result.input_ids.dtype == torch.long
    assert result.attention_mask.dtype == torch.long


def test_batch_consistency(tokenizer):
    """Test that batch processing produces consistent results."""
    texts = ["hello", "world", "test"]

    # Process as batch
    batch_result = tokenizer.torch(texts, add_special_tokens=True, padding=True)

    # Process individually
    individual_results = []
    for text in texts:
        result = tokenizer.torch([text], add_special_tokens=True, padding=False)
        individual_results.append(result.input_ids[0])

    # Check that unpadded portions match
    for i, individual_ids in enumerate(individual_results):
        batch_ids = batch_result.input_ids[i][:len(individual_ids)]
        assert batch_ids.tolist() == individual_ids.tolist()


def test_vocab_properties(tokenizer):
    """Test tokenizer vocabulary properties."""
    assert tokenizer.vocab_size == 256  # 2^8
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3

    vocab = tokenizer.get_vocab()
    assert len(vocab) == 256
    assert vocab[chr(0)] == 0
    assert vocab[chr(2)] == 2
    assert vocab[chr(3)] == 3


def test_tokenizer_save_and_load():
    """Test saving and loading the tokenizer."""
    original_tokenizer = UTF8Tokenizer()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the tokenizer
        original_tokenizer.save_pretrained(temp_dir)

        # Load the tokenizer
        UTF8Tokenizer.from_pretrained(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
