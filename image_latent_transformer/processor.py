import re
from typing import Union

import torch
from transformers import AutoImageProcessor, ProcessorMixin

from image_latent_transformer.collator import collate_fn
from image_latent_transformer.renderer import render_texts_torch
from image_latent_transformer.tokenizer import ByteTokenizer


class TextImageProcessor(ProcessorMixin):
    name = "text-image-processor"

    attributes = ["tokenizer", "image_processor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "ByteTokenizer"
    optional_attributes = ["max_seq_length", "max_word_length"]

    def __init__(self,
                 tokenizer: ByteTokenizer,
                 image_processor: AutoImageProcessor,
                 max_seq_length: int = 128,
                 max_word_length: int = 32):
        super().__init__(tokenizer=tokenizer,
                         image_processor=image_processor,
                         max_seq_length=max_seq_length,
                         max_word_length=max_word_length)

        assert tokenizer.bos_token_id is not None, "Tokenizer must have a BOS token"
        assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length

        # Not used in this processor, but necessary for "from_pretrained" compatibility
        self.chat_template = None
        self.audio_tokenizer = None

    def get_words_and_labels(self, text: str, pack=True) -> tuple[list[str], list[str]]:
        text = ("<> " + text.lstrip()).strip()  # Add BOS token "<>" at the start

        # TODO: Ensure all texts end with a space. this is a model quirk and needs to be handled generally
        #  if the text does not end with a space, the model should continue generating the last word directly
        #  https://github.com/sign/image-latent-transformer/issues/2
        text += " "

        words = re.findall(r'\S+\s*', text)  # Split text into words, keeping spaces
        words = words[:self.max_seq_length]  # Limit to max sequence length

        if pack:
            # Next several characters as label, last word has no label
            labels = []
            label_idx = 0
            for word in words:
                label_idx += len(word)
                # For efficiency, we don't just use the next word as label, but a longer token string
                label = text[label_idx:label_idx + self.max_word_length].strip()
                labels.append(label)
        else:
            # Next word as label, last word has no label
            labels = words[1:] + [""]
            # Truncate labels to max_word_length
            labels = [label[:self.max_word_length] for label in labels]

        return words, labels

    def process_single_example(self, text: str, pack=True):
        words, labels = self.get_words_and_labels(text, pack=pack)

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

    def __call__(self,
                 batch: Union[dict[str, list[str]], str, list[str]],
                 collated=False,
                 packed=True) -> dict[str, torch.Tensor]:
        if isinstance(batch, str):
            batch = {"text": [batch]}

        if isinstance(batch, list):
            batch = {"text": batch}

        dicts = [self.process_single_example(t, pack=packed) for t in batch["text"]]

        if collated:
            return collate_fn(dicts)

        new_batch = {}
        for key in dicts[0].keys():
            new_batch[key] = [d[key] for d in dicts]

        return new_batch
