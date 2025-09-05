import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

from image_latent_transformer.embeddings import (
    PatchedBitEmbeddings,
    join_embedding_layers,
    patch_embedding_layers,
    unpack_bits,
)


class TestEmbeddings:
    @pytest.fixture
    def model(self):
        model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
        model.resize_token_embeddings(256)
        return model

    @pytest.fixture
    def sample_input(self):
        B, L = 2, 16# noqa: N806
        return torch.randint(0, 256, (B, L), dtype=torch.long)

    @pytest.fixture
    def attention_mask(self):
        B, L = 2, 16 # noqa: N806
        return torch.ones(B, L, dtype=torch.long)

    def test_unpack_bits(self):
        x = torch.tensor([255, 0, 128], dtype=torch.uint8)
        bits = unpack_bits(x)

        assert bits.shape == (3, 8)
        assert torch.allclose(bits[0], torch.tensor([1., 1., 1., 1., 1., 1., 1., 1.]))
        assert torch.allclose(bits[1], torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.]))
        assert torch.allclose(bits[2], torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]))

    def test_embedding_setup(self, model):
        original_embeddings = model.get_input_embeddings()
        assert isinstance(original_embeddings, nn.Embedding)
        assert original_embeddings.num_embeddings == 256

        patch_embedding_layers(model)
        patched_embeddings = model.get_input_embeddings()
        assert isinstance(patched_embeddings, PatchedBitEmbeddings)
        assert isinstance(patched_embeddings.embeddings, nn.Embedding)
        assert patched_embeddings.embeddings.num_embeddings == 256

    def test_model_callable_after_patching(self, model, sample_input, attention_mask):
        patch_embedding_layers(model)

        with torch.inference_mode():
            output = model(input_ids=sample_input, attention_mask=attention_mask)

        assert hasattr(output, 'logits')
        assert output.logits.shape == (2, 16, 256)

    def test_parameter_count_preservation(self, model):
        original_param_count = sum(p.numel() for p in model.parameters())
        original_embedding_params = model.get_input_embeddings().weight.numel()

        patch_embedding_layers(model)
        patched_embeddings = model.get_input_embeddings()
        patched_param_count = sum(p.numel() for p in model.parameters())

        join_embedding_layers(model)
        final_param_count = sum(p.numel() for p in model.parameters())
        final_embedding_params = model.get_input_embeddings().weight.numel()

        # The embedding layer parameter count should be preserved
        assert original_embedding_params == final_embedding_params
        # The patched version should have additional parameters (bit projection)
        assert patched_param_count > original_param_count
        # The total parameter difference includes bit projection + combined weight
        embedding_dim = patched_embeddings.embeddings.embedding_dim
        expected_extra_params = 8 * embedding_dim + 256 * embedding_dim  # bit_proj + combined weight
        assert patched_param_count - original_param_count == expected_extra_params
        # After joining, we should be back to the original count
        assert final_param_count == original_param_count

    def test_embedding_weight_preservation(self, model, sample_input):
        original_embeddings = model.get_input_embeddings()
        original_weight = original_embeddings.weight.clone()

        with torch.inference_mode():
            original_output = original_embeddings(sample_input)

        patch_embedding_layers(model)
        join_embedding_layers(model)

        final_embeddings = model.get_input_embeddings()
        final_weight = final_embeddings.weight

        with torch.inference_mode():
            final_output = final_embeddings(sample_input)

        assert torch.allclose(original_weight, final_weight, atol=1e-6)
        assert torch.allclose(original_output, final_output, atol=1e-6)

    def test_patched_bit_embeddings_properties(self, model):
        patch_embedding_layers(model)
        patched_embeddings = model.get_input_embeddings()

        assert hasattr(patched_embeddings, 'weight')
        assert hasattr(patched_embeddings, 'embeddings')
        assert hasattr(patched_embeddings, 'bit_proj_w')

        weight = patched_embeddings.weight
        assert weight.shape == (256, patched_embeddings.embeddings.embedding_dim)

    def test_bit_projection_initialization(self, model):
        patch_embedding_layers(model)
        patched_embeddings = model.get_input_embeddings()

        weight = patched_embeddings.bit_proj_w.data
        assert torch.allclose(weight, torch.zeros_like(weight))

    def test_forward_pass_consistency(self, model, sample_input):
        original_embeddings = model.get_input_embeddings()

        with torch.inference_mode():
            original_embedded = original_embeddings(sample_input)

        patch_embedding_layers(model)
        patched_embeddings = model.get_input_embeddings()

        with torch.inference_mode():
            patched_embedded = patched_embeddings(sample_input)

        assert torch.allclose(original_embedded, patched_embedded, atol=1e-6)

    def test_embedding_weight_consistency_after_patch(self, model):
        original_embeddings = model.get_input_embeddings()
        original_weight = original_embeddings.weight.clone()

        patch_embedding_layers(model)
        patched_embeddings = model.get_input_embeddings()

        assert torch.allclose(original_weight, patched_embeddings.weight, atol=1e-6)

    def test_bit_projection_training(self, model, sample_input):
        patch_embedding_layers(model)
        patched_embeddings = model.get_input_embeddings()

        initial_embedding_weight = patched_embeddings.embeddings.weight.data.clone()

        # Verify bit_proj.weight starts as zeros
        initial_bit_weight = patched_embeddings.bit_proj_w.data.clone()
        assert torch.allclose(initial_bit_weight, torch.zeros_like(initial_bit_weight))

        # Set up a simple training objective: make embeddings closer to 1
        optimizer = torch.optim.SGD(patched_embeddings.parameters(), lr=0.1)
        target = torch.ones_like(patched_embeddings(sample_input))

        # One training step
        optimizer.zero_grad()
        embedded = patched_embeddings(sample_input)
        loss = torch.nn.functional.mse_loss(embedded, target)
        loss.backward()
        optimizer.step()

        # Verify bit_proj.weight is no longer zeros
        final_bit_weight = patched_embeddings.bit_proj_w.data
        assert not torch.allclose(final_bit_weight, torch.zeros_like(final_bit_weight), atol=1e-6)

        # Verify base embeddings have changed too
        final_embedding_weight = patched_embeddings.embeddings.weight.data
        assert not torch.allclose(final_embedding_weight, initial_embedding_weight, atol=1e-6)
