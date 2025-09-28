import difflib
import tempfile

import pytest

from tests.test_model import setup_tiny_model
from welt.config import WordLatentTransformerConfig


def test_config_save_and_load_equal():
    model, processor, collator = setup_tiny_model()

    with tempfile.TemporaryDirectory() as temp_dir:
        model.config.save_pretrained(save_directory=temp_dir, push_to_hub=False)

        new_config = WordLatentTransformerConfig.from_pretrained(temp_dir)

        old_config_text = model.config.to_json_string()
        new_config_text = new_config.to_json_string()

        if old_config_text != new_config_text:
            diff = "\n".join(
                difflib.unified_diff(
                    old_config_text.splitlines(),
                    new_config_text.splitlines(),
                    fromfile="model.config (before save/load)",
                    tofile="new_config (after save/load)",
                    lineterm="",
                )
            )
            pytest.fail(f"Config changed after save/load:\n{diff}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
