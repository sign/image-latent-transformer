import re
from functools import lru_cache, partial
from typing import Union

import torch
from datasets import Dataset
from transformers import AutoImageProcessor, ProcessorMixin

from image_latent_transformer.attention import (
    get_attention_mask_for_packed_sequence,
    get_position_ids_for_packed_sequence,
)
from image_latent_transformer.collator import collate_fn
from image_latent_transformer.renderer import render_text_torch
from image_latent_transformer.tokenizer.control import CONTROl_TOKENS_PATTERN
from image_latent_transformer.tokenizer.utf8 import UTF8Tokenizer

# Consider three classes of tokens:
TOKEN_PATTERN = (
    rf"[{CONTROl_TOKENS_PATTERN}]"  # # 1. Control tokens
    rf"|[^\s{CONTROl_TOKENS_PATTERN}]+\s?"  # 2. Words - no whitespace / control (with optional 1 trailing space)
    r"|\s+"  # 3) whitespace runs
)


class TextImageProcessor(ProcessorMixin):
    name = "text-image-processor"

    attributes = ["tokenizer", "image_processor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "UTF8Tokenizer"

    def __init__(self,
                 tokenizer: UTF8Tokenizer,
                 image_processor: AutoImageProcessor,
                 max_seq_length: int = 128,
                 max_word_length: int = 32,
                 cache_size: int = 10000):
        super().__init__(tokenizer=tokenizer,
                         image_processor=image_processor)

        assert tokenizer.bos_token_id is not None, "Tokenizer must have a BOS token"
        assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length
        self.cache_size = cache_size

        self.render_text = self._cached_renderer()

    def _cached_renderer(self):
        # Bind a cached version of the method
        render_text_fn = partial(render_text_torch, image_processor=self.image_processor)
        # TODO: replace with a multithreaded cache, to avoid caching the same text multiple times in different threads
        #       so that we can make the cache size much larger
        return lru_cache(maxsize=self.cache_size)(render_text_fn)

    def pretokenize(self, text: str) -> list[str]:
        # Add BOS token at the start
        text = self.tokenizer.bos_token + text.strip()

        # TODO: Ensure all texts end with a space. this is a model quirk and needs to be handled generally
        #  if the text does not end with a space, the model should continue generating the last word directly
        #  https://github.com/sign/image-latent-transformer/issues/2
        text += " "

        return re.findall(TOKEN_PATTERN, text)

    def pretokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Pretokenize a dataset in place, adding a 'words' column."""

        def tokenize_example(example):
            example["words"] = self.pretokenize(example["text"])
            return example

        return dataset.map(tokenize_example,
                           batched=False,
                           remove_columns=["text"],
                           desc="Pretokenizing texts into 'words'")

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
                    # max_word_length characters, not bytes, will be trimmed by the tokenizer later
                    label = text[label_idx:label_idx + self.max_word_length]
                    # TODO: remove once https://github.com/sign/image-latent-transformer/issues/2 is solved
                    label = label.rstrip()  # Remove trailing spaces to avoid generating them
                    labels.append(label)
            else:
                # Next word as label, last word has no label
                raw_labels = words[offset + 1:offset + length] + [""]
                # Truncate labels to max_word_length (characters, not bytes)
                labels += [label[:self.max_word_length] for label in raw_labels]
                # TODO: remove once https://github.com/sign/image-latent-transformer/issues/2 is solved
                labels[-2] = labels[-2].rstrip()  # Remove last trailing space to avoid generating it

            offset += length

        return labels

    def tokenize_words(self, words: list[str], device=None):
        return self.tokenizer.torch(
            words,
            padding=True,
            max_length=self.max_word_length,
            truncation=True,
            add_special_tokens=True,
            device=device
        )

    def process_single_example(self, words: list[str], seq_lengths: list[int], pack=True):
        labels = self.get_sequence_labels(words, seq_lengths, pack=pack)

        # Tokenize words with BOS and EOS tokens
        tokenized = self.tokenize_words(words)  # Tokenized inputs
        tokenized_labels = self.tokenize_words(labels)  # Tokenized outputs

        # Render images independently for torch
        images = [self.render_text(word) for word in words]
        # Pack images to a single tensor, to make transfer to the main process more efficient
        # TODO: remove jagged once https://github.com/sign/image-latent-transformer/issues/1 is efficient
        input_pixels = torch.nested.nested_tensor(images, layout=torch.jagged)

        return {
            "input_ids": tokenized.input_ids,
            "input_attention_mask": tokenized.attention_mask,  # Attention within each word
            # Attention across words
            "attention_mask": get_attention_mask_for_packed_sequence(seq_lengths, words=words),
            "position_ids": get_position_ids_for_packed_sequence(seq_lengths),
            "input_pixels": input_pixels,
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

        dicts = [self.process_single_example(words=words, seq_lengths=seq_lengths, pack=packed)
                 for words, seq_lengths in zip(batch["words"], batch["seq_lengths"])]

        if collated:
            return collate_fn(dicts)

        new_batch = {}
        for key in dicts[0].keys():
            new_batch[key] = [d[key] for d in dicts]

        return new_batch

    def __getstate__(self):
        # Remove the lru_cache decorated method for pickling
        state = self.__dict__.copy()
        # Remove the unpickleable render_text method
        del state['render_text']
        return state

    def __setstate__(self, state):
        # Restore the object state and recreate the lru_cache method
        self.__dict__.update(state)
        # Recreate the cached render_text method
        self.render_text = self._cached_renderer()
