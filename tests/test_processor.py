import tempfile

import pytest
import torch

from image_latent_transformer.processor import TextImageProcessor
from tests.test_model import setup_tiny_model


@pytest.fixture(scope="module")
def processor():
    model, processor, collator = setup_tiny_model()
    return processor

expected_tensor_keys = ["input_ids", "attention_mask", "labels_input", "labels_attention_mask", "labels_output"]
expected_keys = expected_tensor_keys + ["input_pixels"]


def test_processor_save_and_load_works(processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_pretrained(save_directory=temp_dir, push_to_hub=False)
        new_processor = TextImageProcessor.from_pretrained(temp_dir)

        assert new_processor.tokenizer is not None
        assert new_processor.image_processor is not None


def test_processor_single_text_collated(processor):
    text = "example text for testing"
    inputs = processor(text, collated=True)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)


def test_processor_single_text_not_collated(processor):
    text = "example text for testing"
    inputs = processor(text)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], list) and len(inputs[key]) == 1 for key in expected_tensor_keys)


def test_processor_list_format_collated(processor):
    text = "example text for testing"
    inputs = processor([text], collated=True)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)


def test_processor_object_format_collated(processor):
    text = "example text for testing"
    inputs = processor({"text": text}, collated=True)
    assert all(key in inputs for key in expected_keys)
    assert all(isinstance(inputs[key], torch.Tensor) for key in expected_tensor_keys)


def test_processor_packed_vs_unpacked_labels(processor):
    text = "hello world test"

    # Test packed=True (default)
    inputs_packed = processor(text, collated=True, packed=True)

    # Test packed=False
    inputs_unpacked = processor(text, collated=True, packed=False)

    # Both should have same structure
    assert all(key in inputs_packed for key in expected_keys)
    assert all(key in inputs_unpacked for key in expected_keys)

    # Input tokens should be the same
    assert torch.equal(inputs_packed["input_ids"], inputs_unpacked["input_ids"])
    assert torch.equal(inputs_packed["attention_mask"], inputs_unpacked["attention_mask"])

    # Labels should be different due to different packing strategies
    assert not torch.equal(inputs_packed["labels_input"], inputs_unpacked["labels_input"])
    assert not torch.equal(inputs_packed["labels_output"], inputs_unpacked["labels_output"])


def test_processor_packed_true_default_behavior(processor):
    text = "example text for testing"

    # Default should be packed=True
    inputs_default = processor(text, collated=True)
    inputs_explicit_packed = processor(text, collated=True, packed=True)

    # Should be identical
    assert torch.equal(inputs_default["labels_input"], inputs_explicit_packed["labels_input"])
    assert torch.equal(inputs_default["labels_output"], inputs_explicit_packed["labels_output"])


def test_get_words_and_labels_packed_vs_unpacked(processor):
    text = "hello world test"

    # Test packed=True
    words_packed, labels_packed = processor.get_words_and_labels(text, pack=True)

    # Test packed=False
    words_unpacked, labels_unpacked = processor.get_words_and_labels(text, pack=False)

    # Words should be the same
    assert words_packed == words_unpacked

    # Labels should be different
    assert labels_packed != labels_unpacked

    assert labels_packed == ['hello world test', 'world test', 'test', '']
    assert labels_unpacked == ['hello ', 'world ', 'test ', '']

def test_get_words_and_labels_packed_vs_unpacked_respect_max_word_length(processor):
    text = "this is a long-test"

    new_processor = TextImageProcessor(
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        max_word_length=3
    )

    # Test packed=True
    words_packed, labels_packed = new_processor.get_words_and_labels(text, pack=True)

    # Test packed=False
    words_unpacked, labels_unpacked = new_processor.get_words_and_labels(text, pack=False)

    assert labels_packed == ['thi', 'is', 'a l', 'lon', '']
    assert labels_unpacked == ['thi', 'is ', 'a ', 'lon', '']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
