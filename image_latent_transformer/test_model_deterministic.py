import torch

from image_latent_transformer.test_model import predict_dataset, setup_tiny_model, setup_model
from image_latent_transformer.test_model_overfitting import train_model


def test_model_setup_is_deterministic():
    """Test that model setup is deterministic - creates identical models when called twice."""
    model1, image_processor1, tokenizer1, collator1 = setup_tiny_model()
    model2, image_processor2, tokenizer2, collator2 = setup_tiny_model()

    # Compare all parameters between the two models
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()

    # Check that both models have the same parameter names
    assert set(model1_state_dict.keys()) == set(model2_state_dict.keys()), \
        "Models have different parameter names"

    # Check that all parameters are identical
    for param_name in model1_state_dict:
        param1 = model1_state_dict[param_name]
        param2 = model2_state_dict[param_name]

        # Check shapes match
        assert param1.shape == param2.shape, \
            f"Parameter {param_name} has different shapes: {param1.shape} vs {param2.shape}"

        # Check values are identical
        assert torch.allclose(param1, param2, rtol=1e-7, atol=1e-7), \
            f"Parameter {param_name} has different values. Max diff: {(param1 - param2).abs().max().item()}"

    print(f"✓ Model setup is deterministic: {len(model1_state_dict)} parameters verified")


def test_train_model_is_deterministic():
    print("Setting up models for training determinism test...")
    models = [train_model(setup_tiny_model, num_epochs=50) for _ in range(2)]

    print("Predicting losses for test texts using both models...")
    test_texts = ["a b", "b a", "a cat", "a dog"]
    losses = [predict_dataset(test_texts, model, image_processor, tokenizer, collator)[0]
              for model, image_processor, tokenizer, collator in models]

    # Compare losses - they should be identical
    tolerance = 1e-4
    for text in test_texts:
        assert abs(losses[0][text] - losses[1][text]) < tolerance, \
            f"Loss mismatch for '{text}': {losses[0][text]:.6f} vs {losses[1][text]:.6f}"

    print("✅ Training determinism test passed - both models produced identical losses!")
