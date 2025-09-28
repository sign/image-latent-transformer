import math
import re
from collections.abc import Iterable
from itertools import chain

import regex
import torch
from transformers import PreTrainedTokenizer, StoppingCriteria, add_start_docstrings
from transformers.generation.stopping_criteria import STOPPING_CRITERIA_INPUTS_DOCSTRING
from utf8_tokenizer.control import CONTROl_TOKENS_PATTERN

from welt.pretokenizer.languages import segment_text

_COMPILED_GRAPHEME_PATTERN = regex.compile(r"\X")
_COMPLETE_WORD_PATTERNS = [
    rf"[{CONTROl_TOKENS_PATTERN}]",  # Control tokens are always complete
    rf"[^\s{CONTROl_TOKENS_PATTERN}]+\s",  # Words with trailing space are complete
]

def words_to_text(words: Iterable[str]) -> str:
    return ''.join(words)


def text_to_words(text: str, max_bytes: int = math.inf) -> list[str]:
    words = chain.from_iterable(segment_text(text))

    if max_bytes == math.inf:
        return list(words)

    chunks = (utf8_chunks_grapheme_safe(word, max_bytes=max_bytes) for word in words)
    return list(chain.from_iterable(chunks))


def utf8_chunks_grapheme_safe(text: str, max_bytes: int = 16) -> Iterable[str]:
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
