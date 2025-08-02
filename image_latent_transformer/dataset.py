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
        max_length: Maximum sequence length for tokenization
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                - input_ids: (LENGTH, TOKENS)
                - attention_mask: (LENGTH, TOKENS) 
                - input_pixels: (LENGTH, CHANNELS, HEIGHT, WIDTH)
        """
        text = self.texts_dataset[idx]
        words = text.split()
        words = words[:self.max_word_length]
        
        # Tokenize words
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
        images = image.squeeze(0)\
            .unflatten(1, (len(words), line_height))\
            .permute(1, 0, 2, 3)  # [3, N, 32, 128] -> [N, 3, 32, 128]
        
        return {
            "input_ids": tokenized.input_ids.squeeze(0),  # Remove batch dim
            "attention_mask": tokenized.attention_mask.squeeze(0),  # Remove batch dim
            "input_pixels": images
        }