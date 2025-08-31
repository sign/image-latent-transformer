import re
from itertools import chain

import regex as reg
import torch
from transformers import PreTrainedTokenizer, StoppingCriteria, add_start_docstrings
from transformers.generation.stopping_criteria import STOPPING_CRITERIA_INPUTS_DOCSTRING

from image_latent_transformer.tokenizer.control import CONTROl_TOKENS_PATTERN

# Consider three classes of tokens:
_TOKEN_PATTERN = (
    rf"[{CONTROl_TOKENS_PATTERN}]"  # # 1. Control tokens
    rf"|[^\s{CONTROl_TOKENS_PATTERN}]+\s?"  # 2. Words - no whitespace / control (with optional 1 trailing space)
    r"|\s+"  # 3) whitespace runs
)
_COMPLETE_WORD_PATTERNS = [
    rf"[{CONTROl_TOKENS_PATTERN}]",  # Control tokens are always complete
    rf"[^\s{CONTROl_TOKENS_PATTERN}]+\s",  # Words with trailing space are complete
]
_COMPILED_TOKEN_PATTERN = re.compile(_TOKEN_PATTERN)
_COMPILED_GRAPHEME_PATTERN = reg.compile(r"\X")


def text_to_words(text: str, max_bytes: int) -> list[str]:
    words = _COMPILED_TOKEN_PATTERN.findall(text)
    chunks = (utf8_chunks_grapheme_safe(word, max_bytes=max_bytes) for word in words)
    return list(chain.from_iterable(chunks))


def utf8_chunks_grapheme_safe(text: str, max_bytes: int = 16) -> iter:
    """
    Split a string into chunks of at most max_bytes bytes, without splitting grapheme clusters.
    Except, if there is a single grapheme cluster longer than max_bytes, it will be in its own chunk. 👩‍👩‍👧‍👦
    """
    text_bytes = text.encode("utf-8")
    if len(text_bytes) <= max_bytes:
        yield text
        return

    clusters = _COMPILED_GRAPHEME_PATTERN.findall(text)
    if len(clusters) == 1:
        yield text
        return

    curr = []
    curr_bytes = 0
    for cluster in clusters:
        cluster_bytes = len(cluster.encode("utf-8"))
        if curr_bytes + cluster_bytes > max_bytes:
            if curr:
                yield "".join(curr)
            curr = [cluster]
            curr_bytes = cluster_bytes
        else:
            curr.append(cluster)
            curr_bytes += cluster_bytes
    if curr:
        yield "".join(curr)


def is_word_complete(text: str) -> bool:
    for pattern in _COMPLETE_WORD_PATTERNS:
        if re.fullmatch(pattern, text):
            return True

    # TODO: not clear how to know if a word full of whitespaces is complete
    #       maybe if _TOKEN_PATTERN is not a full match, but then need to "delete" the last token.
    return False


class WordStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        texts = [self.tokenizer.decode(ids) for ids in input_ids]
        is_done = [is_word_complete(text) for text in texts]
        return torch.BoolTensor(is_done)
