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
        TextImageProcessor.from_pretrained(temp_dir)


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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
