import pytest
import torch
from torch import nn
from transformers import AutoModel, AutoModelForImageClassification

from welt.vision.masked_vit_patcher import (
    is_vit_model,
    maybe_patch_vit_model,
)
from welt.vision.vision_utils import encode_images


class TestIsVitModel:
    def test_is_vit_model_with_vit_base(self):
        """Test is_vit_model returns True for ViT base model"""
        model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        assert is_vit_model(model) is True

    def test_is_vit_model_with_vit_classification(self):
        """Test is_vit_model returns True for ViT classification model"""
        model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        assert is_vit_model(model) is True

    def test_is_vit_model_with_mobilenet(self):
        """Test is_vit_model returns False for MobileNet"""
        model = AutoModel.from_pretrained("google/mobilenet_v2_1.0_224")
        assert is_vit_model(model) is False

    def test_is_vit_model_with_swin_transformer(self):
        """Test is_vit_model returns False for Swin Transformer"""
        model = AutoModel.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256")
        assert is_vit_model(model) is False


class TestMaybePatchVitModel:
    def test_maybe_patch_vit_model_patches_vit(self):
        """Test maybe_patch_vit_model applies patch to ViT model"""
        model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.eval()

        # Check model is not patched initially
        assert not getattr(model, "_zero_pad_mask_patched", False)

        # Apply patch
        maybe_patch_vit_model(model)

        # Check model is now patched
        assert getattr(model, "_zero_pad_mask_patched", False)

    def test_maybe_patch_vit_model_with_classification_model(self):
        """Test maybe_patch_vit_model works with ViTForImageClassification"""
        model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.eval()

        # Check vit submodel is not patched initially
        assert not getattr(model.vit, "_zero_pad_mask_patched", False)

        # Apply patch
        maybe_patch_vit_model(model)

        # Check vit submodel is now patched
        assert getattr(model.vit, "_zero_pad_mask_patched", False)

    def test_maybe_patch_vit_model_idempotent(self):
        """Test maybe_patch_vit_model can be called multiple times safely"""
        model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.eval()

        # Apply patch twice
        maybe_patch_vit_model(model)
        maybe_patch_vit_model(model)

        # Should still be patched
        assert getattr(model, "_zero_pad_mask_patched", False)


class TestEncodeImagesWithPatching:
    def test_encode_images_padding_invariance(self):
        """Test that patched model produces same output for padded and unpadded images"""
        model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.eval()
        torch.manual_seed(0)  # For determinism

        # Create test images
        example_image = torch.randn(1, 3, 64, 64)
        example_image_padded = torch.zeros(1, 3, 128, 128)
        example_image_padded[:, :, :64, :64] = example_image

        # Test without patching - should be different
        output_no_pad_before = encode_images(model, example_image)
        output_pad_before = encode_images(model, example_image_padded)

        # Outputs should be different before patching
        assert not torch.allclose(output_no_pad_before, output_pad_before, atol=1e-3)

        # Apply patch
        maybe_patch_vit_model(model)

        # Test with patching - should be the same
        output_no_pad_after = encode_images(model, example_image)
        output_pad_after = encode_images(model, example_image_padded)

        # Outputs should be the same after patching
        assert torch.allclose(output_no_pad_after, output_pad_after, atol=1e-3)

    def test_encode_images_deterministic_output(self):
        """Test that encode_images produces consistent output with same seed"""
        model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.eval()
        maybe_patch_vit_model(model)

        example_image = torch.randn(1, 3, 64, 64)

        # Set seed and encode
        torch.manual_seed(42)
        output1 = encode_images(model, example_image)

        # Reset seed and encode again
        torch.manual_seed(42)
        output2 = encode_images(model, example_image)

        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-8)

    def test_encode_zeros_should_return_zeros(self):
        """Test that encode_images produces consistent output with same seed"""
        model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.eval()
        maybe_patch_vit_model(model)

        example_image = torch.zeros(1, 3, 64, 64)

        # Set seed and encode
        torch.manual_seed(42)
        output = encode_images(model, example_image)

        # Should be identical
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-8)

    def test_encode_multiple_vs_individual(self):
        """Test that batch processing gives same results as individual processing for different models."""
        model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.eval()
        maybe_patch_vit_model(model)

        img1 = torch.randn(1, 3, 64, 64)
        img2 = torch.randn(1, 3, 32, 32)
        img2_padded = torch.zeros(1, 3, 64, 64)
        img2_padded[:, :, :32, :32] = img2

        img_batch = torch.cat([img1, img2_padded], dim=0)
        out_batch = encode_images(model, img_batch)

        out_1 = encode_images(model, img1)
        out_2 = encode_images(model, img2)

        assert torch.allclose(out_batch[0], out_1, atol=1e-4)
        assert torch.allclose(out_batch[1], out_2, atol=1e-4)


def _make_padded_batch(shapes, device):
    """
    shapes: list of (C,H,W) tuples (batch dimension implied by list length)
    Returns: pixel_values (B,C,Hmax,Wmax), mask list (per-sample valid (H,W) areas)
    """
    C = shapes[0][0] # noqa: N806
    h_max = max(h for _, h, _ in shapes)
    w_max = max(w for _, _, w in shapes)
    batch = []
    for _, H, W in shapes: # noqa: N806
        x = torch.randn(C, H, W, device=device)
        canvas = torch.zeros(C, h_max, w_max, device=device)
        canvas[:, :H, :W] = x
        batch.append(canvas)
    return torch.stack(batch, dim=0)  # (B,C,Hmax,Wmax)


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


@pytest.mark.parametrize("device", DEVICES)
def test_one_train_step_vit_base(device):
    """Single training step through ViTModel + small head with padded batch."""
    torch.manual_seed(0)
    model = AutoModel.from_pretrained("WinKawaks/vit-tiny-patch16-224").to(device)
    model.train()
    maybe_patch_vit_model(model)  # patch invariance

    # Tiny projection head on [CLS]
    head = nn.Sequential(
        nn.LayerNorm(model.config.hidden_size),
        nn.Linear(model.config.hidden_size, 8),
    ).to(device)

    # Build padded batch: different spatial sizes
    pixel_values = _make_padded_batch([(3, 64, 64), (3, 32, 32), (3, 96, 48)], device)  # (B,3,Hm,Wm)

    opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    opt.zero_grad(set_to_none=True)

    out = model(pixel_values=pixel_values, output_hidden_states=False)
    cls = out.last_hidden_state[:, 0, :]  # (B,D)
    logits = head(cls)  # (B,8)

    # Simple target: zeros
    target = torch.zeros_like(logits)
    loss = nn.functional.mse_loss(logits, target)

    assert torch.isfinite(loss).item(), "Loss is not finite before backward"
    loss.backward()
    opt.step()

    assert torch.isfinite(loss).item(), "Loss is not finite after step"


@pytest.mark.parametrize("device", DEVICES)
def test_one_train_step_vit_for_classification(device):
    """Single training step through ViTForImageClassification with padded batch."""
    torch.manual_seed(0)
    clf = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224").to(device)
    clf.train()
    maybe_patch_vit_model(clf)  # handles .vit

    pixel_values = _make_padded_batch([(3, 64, 64), (3, 128, 64)], device)  # (B,3,Hm,Wm)
    B = pixel_values.size(0) # noqa: N806
    num_labels = clf.config.num_labels or 1000

    # Random labels from the valid range
    labels = torch.randint(0, num_labels, (B,), device=device)

    opt = torch.optim.AdamW(clf.parameters(), lr=5e-4)
    opt.zero_grad(set_to_none=True)

    out = clf(pixel_values=pixel_values, labels=labels)
    loss = out.loss
    assert torch.isfinite(loss).item(), "Loss is not finite before backward"
    loss.backward()
    opt.step()

    assert torch.isfinite(loss).item(), "Loss is not finite after step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
