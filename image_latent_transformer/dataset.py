import re

import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from image_latent_transformer.renderer import render_texts_torch
from image_latent_transformer.tokenizer import ByteTokenizer


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
                 texts_dataset: list[str],
                 image_processor: AutoImageProcessor,
                 tokenizer: ByteTokenizer,
                 max_seq_length: int = 128,
                 max_word_length: int = 32):
        self.texts_dataset = texts_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length

        assert tokenizer.bos_token_id is not None, "Tokenizer must have a BOS token"
        assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"

    def __len__(self) -> int:
        return len(self.texts_dataset)

    def get_words_and_labels(self, text: str) -> tuple[list[str], list[str]]:
        text = ("<> " + text).strip()  # Add BOS token "<>" at the start

        # TODO: Ensure all texts end with a space. this is a model quirk and needs to be handled generally
        #  if the text does not end with a space, the model should continue generating the last word directly
        text += " "

        words = re.findall(r'\S+\s*', text)  # Split text into words, keeping spaces
        words = words[:self.max_seq_length]  # Limit to max sequence length

        labels = []
        label_idx = 0
        for word in words:
            label_idx += len(word)
            # For efficiency, we don't just use the next word as label, but a longer token string
            label = text[label_idx:label_idx + self.max_word_length].strip()
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
            truncation=True,
            add_special_tokens=False  # get_words_and_labels adds BOS already
        )

        # Render images independently for torch
        images = render_texts_torch(words, image_processor=self.image_processor)

        # Tokenize labels with BOS and EOS tokens for bytes decoder
        tokenized_labels = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            max_length=self.max_word_length,
            truncation=True,
            add_special_tokens=True  # Add BOS/EOS tokens
        )

        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "input_pixels": images,
            "labels_input": tokenized_labels.input_ids[:, :-1],  # Remove EOS token from input labels
            "labels_attention_mask": tokenized_labels.attention_mask[:, :-1],  # Remove EOS token from attention mask
            "labels_output": tokenized_labels.input_ids[:, 1:]  # Remove BOS token from output labels
        }
