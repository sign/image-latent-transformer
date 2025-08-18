import tempfile

import pytest

from image_latent_transformer.processor import TextImageProcessor
from tests.test_model import setup_tiny_model


def test_processor_save_and_load_works():
    model, processor, collator = setup_tiny_model()

    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_pretrained(save_directory=temp_dir, push_to_hub=False)
        TextImageProcessor.from_pretrained(temp_dir)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
