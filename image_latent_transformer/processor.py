import re
from typing import Union

import torch
from datasets import Dataset
from transformers import AutoImageProcessor, ProcessorMixin

from image_latent_transformer.attention import (
    get_attention_mask_for_packed_sequence,
    get_position_ids_for_packed_sequence,
)
from image_latent_transformer.collator import collate_fn
from image_latent_transformer.renderer import render_texts_torch
from image_latent_transformer.tokenizer import ByteTokenizer


class TextImageProcessor(ProcessorMixin):
    name = "text-image-processor"

    attributes = ["tokenizer", "image_processor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "ByteTokenizer"

    def __init__(self,
                 tokenizer: ByteTokenizer,
                 image_processor: AutoImageProcessor,
                 max_seq_length: int = 128,
                 max_word_length: int = 32):
        super().__init__(tokenizer=tokenizer,
                         image_processor=image_processor)

        assert tokenizer.bos_token_id is not None, "Tokenizer must have a BOS token"
        assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length

    def pretokenize(self, text: str) -> list[str]:
        # Add BOS token at the start
        text = self.tokenizer.bos_token + " " + text.strip()

        # TODO: Ensure all texts end with a space. this is a model quirk and needs to be handled generally
        #  if the text does not end with a space, the model should continue generating the last word directly
        #  https://github.com/sign/image-latent-transformer/issues/2
        text += " "

        # Split text into words, keeping spaces
        words = re.findall(r'\S+\s*', text)

        return words

    def pretokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Pretokenize a dataset in place, adding a 'words' column."""

        def tokenize_example(example):
            example["words"] = self.pretokenize(example["text"])
            return example

        return dataset.map(tokenize_example, batched=False, remove_columns=["text"])

    def get_sequence_labels(self, words: list[str], seq_lengths: list[int] = None, pack=True) -> list[str]:
        if seq_lengths is None:
            seq_lengths = [len(words)]

        labels = []

        # Process each sequence separately, to support efficient packing
        offset = 0
        for length in seq_lengths:
            if pack:
                # Next several characters as label, last word has no label
                segment_words = words[offset:offset + length]
                text = "".join(segment_words)
                label_idx = 0
                for word in segment_words:
                    label_idx += len(word)
                    # For efficiency, we don't just use the next word as label, but a longer token string
                    label = text[label_idx:label_idx + self.max_word_length]
                    # TODO: remove once https://github.com/sign/image-latent-transformer/issues/2 is solved
                    label = label.rstrip()  # Remove trailing spaces to avoid generating them
                    labels.append(label)
            else:
                # Next word as label, last word has no label
                raw_labels = words[offset + 1:offset + length] + [""]
                # Truncate labels to max_word_length
                labels += [label[:self.max_word_length] for label in raw_labels]
                # TODO: remove once https://github.com/sign/image-latent-transformer/issues/2 is solved
                labels[-2] = labels[-2].rstrip()  # Remove last trailing space to avoid generating it

            offset += length

        return labels

    def process_single_example(self, words: list[str], seq_lengths: list[int], pack=True):
        labels = self.get_sequence_labels(words, seq_lengths, pack=pack)

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
            "input_attention_mask": tokenized.attention_mask, # Attention within each word
            "attention_mask": get_attention_mask_for_packed_sequence(seq_lengths, words=words), # Attention across words
            "position_ids": get_position_ids_for_packed_sequence(seq_lengths),
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

        if "text" in batch and "words" not in batch:
            words = [self.pretokenize(t) for t in batch["text"]]
            # Create a similar batch object to a packed dataset
            batch = {"words": words, "seq_lengths": [[len(w)] for w in words]}

        dicts = [self.process_single_example(words, seq_lengths, pack=packed)
                 for words, seq_lengths in zip(batch["words"], batch["seq_lengths"])]

        if collated:
            return collate_fn(dicts)

        new_batch = {}
        for key in dicts[0].keys():
            new_batch[key] = [d[key] for d in dicts]

        return new_batch
