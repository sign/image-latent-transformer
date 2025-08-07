import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification, ByT5Tokenizer, AutoModelForCausalLM, \
    set_seed, Trainer, TrainingArguments

from image_latent_transformer.ilt import ImageLatentTransformer
from image_latent_transformer.dataset import TextImageDataset
from image_latent_transformer.utils import collate_fn


def setup_model():
    """Set up the ImageLatentTransformer model like in image_model.py."""
    # Image Encoder
    image_model_name = "microsoft/swinv2-tiny-patch4-window16-256"
    image_processor = AutoImageProcessor.from_pretrained(image_model_name, use_fast=True)
    image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
    image_model.classifier = torch.nn.Identity()

    # Small Language Model
    tokenizer = ByT5Tokenizer(extra_ids=5)
    byte_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    byte_lm.resize_token_embeddings(len(tokenizer))

    # Latent Transformer
    latent_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    latent_lm.resize_token_embeddings(0)

    # Combine the models
    model = ImageLatentTransformer(
        image_encoder=image_model,
        bytes_encoder=byte_lm,
        latent_transformer=latent_lm,
        bytes_decoder=byte_lm
    )
    
    return model, image_processor, tokenizer


def compute_sequence_likelihood(model, batch):
    """Compute the likelihood of sequences in a batch using loss (lower loss = higher likelihood)."""
    with torch.no_grad():
        outputs = model(**batch)
        # Return negative loss as likelihood proxy (lower loss = higher likelihood)
        return -outputs.loss.item()


def test_ilt_overfitting():
    set_seed(42)

    """Test that the model can overfit on simple sequences and learn contextual patterns."""
    # Setup
    model, image_processor, tokenizer = setup_model()
    
    # Create simple training data - just "a b" and "b a"
    train_texts = ["a b", "b a"]
    train_dataset = TextImageDataset(
        texts_dataset=train_texts,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_seq_length=128,
        max_word_length=32
    )
    
    
    # Setup training arguments with logging every step
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=30,
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        logging_steps=1,
        logging_strategy="steps",
        save_steps=10,
        eval_steps=10,
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    # Train the model
    trainer.train()
    
    # Test contextual understanding
    model.eval()
    
    # Create test sequences
    test_texts = ["a b", "b a", "a a", "b b"]
    test_dataset = TextImageDataset(
        texts_dataset=test_texts,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_seq_length=128,
        max_word_length=32
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=len(test_texts), 
        shuffle=False, 
        collate_fn=collate_fn
    )

    with torch.no_grad():
        test_batch = next(iter(test_dataloader))
        outputs = model(**test_batch)
        initial_loss = outputs.loss
        print(initial_loss.shape)
    
    # Compute likelihoods for each test sequence individually
    ab_batch = next(iter(DataLoader(Subset(test_dataset, [0]), batch_size=1, collate_fn=collate_fn)))
    ba_batch = next(iter(DataLoader(Subset(test_dataset, [1]), batch_size=1, collate_fn=collate_fn)))
    aa_batch = next(iter(DataLoader(Subset(test_dataset, [2]), batch_size=1, collate_fn=collate_fn)))
    bb_batch = next(iter(DataLoader(Subset(test_dataset, [3]), batch_size=1, collate_fn=collate_fn)))
    
    ab_likelihood = compute_sequence_likelihood(model, ab_batch)  # "a b"
    ba_likelihood = compute_sequence_likelihood(model, ba_batch)  # "b a"
    aa_likelihood = compute_sequence_likelihood(model, aa_batch)  # "a a"
    bb_likelihood = compute_sequence_likelihood(model, bb_batch)  # "b b"
    
    print(f"Likelihoods:")
    print(f"'a b': {ab_likelihood:.4f}")
    print(f"'b a': {ba_likelihood:.4f}")
    print(f"'a a': {aa_likelihood:.4f}")
    print(f"'b b': {bb_likelihood:.4f}")
    
    # Assertions
    # 1. Loss should decrease during training
    final_batch = next(iter(test_dataloader))
    final_outputs = model(**final_batch)
    final_loss = final_outputs.loss.item()
    
    assert final_loss < initial_loss, f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    
    # 2. Model should prefer trained sequences over similar untrained ones
    assert ab_likelihood > aa_likelihood, \
        f"'a b' should have higher likelihood than 'a a': {ab_likelihood:.4f} vs {aa_likelihood:.4f}"
    
    assert ba_likelihood > bb_likelihood, \
        f"'b a' should have higher likelihood than 'b b': {ba_likelihood:.4f} vs {bb_likelihood:.4f}"
    
    # 3. Both "a b" and "b a" should have reasonable likelihoods (not -inf)
    assert ab_likelihood > float('-inf'), "a b likelihood should be finite"
    assert ba_likelihood > float('-inf'), "b a likelihood should be finite"
    
    print("✅ Overfitting test passed! Model shows contextual learning.")


if __name__ == "__main__":
    test_ilt_overfitting()