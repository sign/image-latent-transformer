import tempfile

import pytest
from transformers import AutoImageProcessor

from welt.noop import NoopImageProcessor


def test_image_processor_save_load():
    processor = NoopImageProcessor()
    assert isinstance(processor, NoopImageProcessor)

    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_pretrained(save_directory=temp_dir, push_to_hub=False)
        new_processor = AutoImageProcessor.from_pretrained(temp_dir)
        assert isinstance(new_processor, NoopImageProcessor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
