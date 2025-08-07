import re

import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List
from transformers import AutoImageProcessor, ByT5Tokenizer

from image_latent_transformer.renderer import render_texts


class TextImageDataset(Dataset):
    """
    PyTorch dataset that converts text data into input_ids, attention_mask, and input_pixels.
    
    Args:
        texts_dataset: Dataset or list of texts to process
        image_processor: HuggingFace image processor for converting rendered text to pixels
        tokenizer: Tokenizer for converting text to input_ids and attention_mask
        max_seq_length: Maximum sequence length for tokenization
        max_word_length: Maximum length of each word for tokenization
    """

    def __init__(self,
                 texts_dataset: List[str],
                 image_processor: AutoImageProcessor,
                 tokenizer: ByT5Tokenizer,
                 max_seq_length: int = 128,
                 max_word_length: int = 32):
        self.texts_dataset = texts_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts_dataset)

    def get_words_and_labels(self, text: str) -> tuple[list[str], list[str]]:
        words = re.findall(r'\S+\s*', text)
        words = words[:self.max_seq_length]  # Limit to max sequence length

        labels = []
        label_idx = 0
        for word in words:
            label_idx += len(word)
            # For efficiency, we don't just use the next word as label, but a longer token string
            label = text[label_idx:label_idx+self.max_word_length].strip()
            labels.append(label)
        return words, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                - input_ids: (LENGTH, TOKENS)
                - attention_mask: (LENGTH, TOKENS) 
                - input_pixels: (LENGTH, CHANNELS, HEIGHT, WIDTH)
        """
        text = self.texts_dataset[idx]
        words, labels = self.get_words_and_labels(text)

        # Tokenize words with BOS and EOS tokens
        tokenized = self.tokenizer(
            words,
            return_tensors="pt",
            padding=True,
            max_length=self.max_word_length,
            truncation=True
        )

        # Render text to images
        line_height, text_lines = render_texts(words)

        # Process images
        image = self.image_processor(
            text_lines,
            return_tensors="pt",
            do_center_crop=False,
            do_resize=False
        ).pixel_values

        # Deconstruct the image to one image per line/word
        images = image.squeeze(0) \
            .unflatten(1, (len(words), line_height)) \
            .permute(1, 0, 2, 3)  # [3, N, 32, 128] -> [N, 3, 32, 128]

        # Tokenize labels with BOS and EOS tokens for bytes decoder
        tokenized_labels = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            max_length=self.max_word_length,
            truncation=True,
            add_special_tokens=True  # Add BOS/EOS tokens
        )

        input_ids = tokenized.input_ids.squeeze(0)  # Remove batch dim
        attention_mask = tokenized.attention_mask.squeeze(0)  # Remove batch dim
        
        labels_input_ids = tokenized_labels.input_ids.squeeze(0)  # Remove batch dim
        labels_attention_mask = tokenized_labels.attention_mask.squeeze(0)  # Remove batch dim

        # Create labels_input (for bytes decoder input) - right-shifted with BOS
        labels_input = labels_input_ids.clone()
        
        # Create labels_output (for loss computation) - left-shifted, removing BOS
        labels_output = labels_input_ids.clone()
        labels_output[:, :-1] = labels_input_ids[:, 1:]  # Shift left
        labels_output[:, -1] = -100  # Last position has no next token
        # Set padded positions to -100 (ignored in loss)
        labels_output[labels_attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_pixels": images,
            "labels_input": labels_input,
            "labels_attention_mask": labels_attention_mask,
            "labels_output": labels_output
        }
