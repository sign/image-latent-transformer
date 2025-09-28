import tempfile

import pytest
import torch

from tests.test_model import num_model_params
from welt.vision.navit import NaViTConfig, NaViTModel


@pytest.fixture(scope="module")
def config():
    return NaViTConfig(
        image_size=256,
        patch_size=16,
        hidden_size=512,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.0,
        emb_dropout=0.0,
        token_dropout_prob=0.1,
    )


@pytest.fixture(scope="module")
def model(config):
    return NaViTModel(config)


def test_model_forward(model):
    images = [
        torch.randn(3, 256, 256),
        torch.randn(3, 128, 128),
        torch.randn(3, 128, 256),
        torch.randn(3, 256, 128),
        torch.randn(3, 64, 256)
    ]

    out = model(images=images)
    assert out.pooler_output.shape == (5, 512)


def test_model_from_pretrained_works(model, config):
    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        config.save_pretrained(temp_dir)
        original_num_parameters = num_model_params(model)

        new_model = NaViTModel.from_pretrained(temp_dir)
        loaded_num_parameters = num_model_params(new_model)
        assert original_num_parameters == loaded_num_parameters, \
            f"Number of parameters mismatch: {original_num_parameters:,} vs {loaded_num_parameters:,}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
