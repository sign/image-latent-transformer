"""
Script-aware segmentation with per-language callbacks.
- Uses Unicode Script_Extensions (scx) to segment Han/Hiragana/Katakana, etc.
- Falls back to a Default branch that avoids those scripts.
- Each non-default segment is passed to its language-specific callback.
"""

from functools import cache
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypedDict

import regex

from image_latent_transformer.pretokenizer.control import CONTROl_TOKENS_PATTERN
from image_latent_transformer.pretokenizer.chinese import segment_chinese
from image_latent_transformer.pretokenizer.japanese import segment_japanese

# Three classes of tokens inside the Default branch:
# 1) Control tokens (always atomic)
# 2) "Words" = runs of non-space, non-control + optional trailing single space
# 3) Whitespace runs
_TOKEN_PATTERN = (
    rf"[{CONTROl_TOKENS_PATTERN}]"  # 1) Control tokens
    rf"|[^\s{CONTROl_TOKENS_PATTERN}]+\s?"  # 2) Word (+ optional trailing space)
    r"|\s+"  # 3) Whitespace runs
)
_COMPILED_TOKEN_PATTERN = regex.compile(_TOKEN_PATTERN)


def text_to_unbound_words(text: str) -> List[str]:
    """Tokenize a non-scripted span using the token rules above."""
    return _COMPILED_TOKEN_PATTERN.findall(text)


class LanguageSpec(TypedDict):
    scripts: Tuple[str, ...]  # e.g., ("Han",) or ("Han", "Hiragana", "Katakana")
    callback: Callable[[str], Any]  # called with the matched span


LANGUAGE_SPECS: Dict[str, LanguageSpec] = {
    "Chinese": {
        "scripts": ("Han",),
        "callback": segment_chinese,
    },
    "Japanese": {
        "scripts": ("Han", "Hiragana", "Katakana"),
        "callback": segment_japanese,
    },
    "Default": {
        "scripts": tuple(),
        "callback": text_to_unbound_words,
    },
}


def _union_scx(scripts: Tuple[str, ...]) -> str:
    """Create a non-capturing alternation for a set of Script_Extensions."""
    parts = [fr"\p{{scx={s}}}" for s in scripts]
    return "(?:" + "|".join(parts) + ")"


@cache
def build_regex_from_languages() -> regex.Pattern:
    """
    Compile the master regex with named groups for each language plus Default.

    Precedence: dict order in LANGUAGE_SPECS — first match wins if script sets overlap.
    Default branch: consumes runs that do NOT begin with any of the listed scripts.
    """
    # Explicit language branches (skip Default — it has no 'scripts')
    branches: List[str] = []
    for name, spec in LANGUAGE_SPECS.items():
        if spec["scripts"]:
            branches.append(fr"(?P<{name}>{_union_scx(spec['scripts'])}+)")

    # Default: refuse any char that begins one of the explicit-script branches
    all_scripts = tuple(sorted({s for spec in LANGUAGE_SPECS.values() for s in spec.get("scripts", ())}))
    forbidden = _union_scx(all_scripts) if all_scripts else r"$a"  # impossible atom if no scripts exist
    default_branch = fr"(?P<Default>(?:(?!{forbidden})\X)+)"

    # Combined pattern (verbose mode for readability)
    pattern = r"(?x)(?:" + "|".join(branches + [default_branch]) + r")"
    return regex.compile(pattern)


def segment_text(text: str) -> Iterable[Any]:
    """
    Iterate over callback results for each matched span.
    - Non-Default groups call their language callback.
    - Default group calls its callback if present in LANGUAGE_SPECS.
    """
    pat = build_regex_from_languages()
    for m in pat.finditer(text):
        group_name = m.lastgroup
        spec = LANGUAGE_SPECS.get(group_name)
        yield spec["callback"](m.group(0))


if __name__ == "__main__":
    sample = "東京abcかなカナ漢字123 אני אחד私は学生です"
    # Stream results as produced by callbacks
    print(list(chain.from_iterable(segment_text(sample))))
