import warnings
from collections import namedtuple
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import TextInput

from image_latent_transformer.pretokenizer.control import ControlTokens


def tokenize_ids(text: str, errors="strict"):
    return list(text.encode('utf-8', errors=errors))


TokenizerResult = namedtuple("TokenizerResult", ["input_ids", "attention_mask"])


class UTF8Tokenizer(PreTrainedTokenizer):
    """
    Custom UTF8 Byte Level Tokenizer implementation,
    extending PreTrainedTokenizer for basic Hugging Face ecosystem support.

    This tokenizer only supports exactly 256 tokens, with no support for "special tokens".
    See README.md to learn more how this works with control tokens instead.

    Additionally, exposes a `.torch` method, which fuses and skips unnecessary ops,
    to achieve a ~3x speedup over `__call__` for training purposes.
    """

    def __init__(self, **kwargs):
        # Pad token
        kwargs["pad_token"] = getattr(kwargs, "pad_token", ControlTokens.Null)
        kwargs["pad_token_id"] = ord(kwargs["pad_token"])
        # BOS token
        kwargs["bos_token"] = getattr(kwargs, "bos_token", ControlTokens.StartOfText)
        kwargs["bos_token_id"] = ord(kwargs["bos_token"])
        # EOS token
        kwargs["eos_token"] = getattr(kwargs, "eos_token", ControlTokens.EndOfText)
        kwargs["eos_token_id"] = ord(kwargs["eos_token"])

        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return 2 ** 8

    def add_tokens(self, *args, **kwargs):
        raise NotImplementedError("UTF8Tokenizer does not support adding tokens")

    def get_vocab(self):
        return {chr(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: TextInput, errors="strict", **kwargs):
        return [chr(c) for c in tokenize_ids(text, errors=errors)]

    def tokenize(self, text: TextInput, errors="strict", **kwargs):
        return self._tokenize(text, errors=errors, **kwargs)

    def _encode_plus(self, text: TextInput, **kwargs):
        return self.prepare_for_model(tokenize_ids(text), **kwargs)

    def _convert_token_to_id(self, token: str):
        return ord(token)

    def _convert_id_to_token(self, index: int):
        return chr(index)

    def convert_tokens_to_string(self, tokens: list[str]):
        """Converts a sequence of tokens (string) in a single string."""
        _bytes = bytes(ord(token) for token in tokens)
        return _bytes.decode("utf-8", errors="ignore")

    def build_inputs_with_special_tokens(self,
                                         token_ids_0: list[int],
                                         token_ids_1: Optional[list[int]] = None) -> list[int]:
        assert token_ids_1 is None, "UTF8Tokenizer only supports single sequence"
        # Experimentally, the fastest way to add BOS/EOS
        token_ids_0.append(self.eos_token_id)  # EOS
        token_ids_0.insert(0, self.bos_token_id)  # BOS
        return token_ids_0

    def torch(self,
              texts: list[TextInput],
              add_special_tokens: bool = True,
              padding: bool = False,
              truncation: bool = None,
              max_length: Optional[int] = None,
              device: Optional[torch.device] = None):

        input_ids = [tokenize_ids(text) for text in texts]

        if truncation is not None:
            if max_length is None:
                warnings.warn(
                    "Asking to truncate to max_length but no maximum length is provided and the model has "
                    "no predefined maximum length. Default to no truncation.",
                    stacklevel=2)
            else:
                corrected_max_length = max_length - 2 if add_special_tokens else max_length
                if corrected_max_length < 0:
                    warnings.warn(
                        "We need to remove more tokens than exist. Default to no truncation.",
                        stacklevel=2)
                else:
                    input_ids = [ids[:max_length] for ids in input_ids]

        if add_special_tokens:
            # Faster to manipulate strings than lists of ints
            input_ids = [self.build_inputs_with_special_tokens(ids) for ids in input_ids]

        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [torch.ones(len(ids), dtype=torch.long) for ids in input_ids]

        if padding:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        return TokenizerResult(input_ids=input_ids, attention_mask=attention_mask)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()


AutoTokenizer.register(UTF8Tokenizer, slow_tokenizer_class=UTF8Tokenizer)
