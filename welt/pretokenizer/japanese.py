"""
Japanese text pretokenization utilities.

This module provides functions for detecting and segmenting Japanese text using the fugashi
library with MeCab for morphological analysis.
"""

from functools import cache

import regex


def has_japanese(text: str) -> bool:
    """
    Check if the given text contains Japanese characters.

    Detects Japanese writing systems including:
    - Hiragana (U+3040-U+309F): Phonetic script for native words
    - Katakana (U+30A0-U+30FF): Phonetic script for foreign words
    - Kanji (Han ideographs): Chinese characters used in Japanese
    - Half-width Katakana (U+FF65-U+FF9F): Narrow katakana variants

    Args:
        text: The input text to check for Japanese characters

    Returns:
        True if Japanese characters are found, False otherwise
    """
    # Match Hiragana, Katakana, or Han ideographs (Kanji)
    return bool(regex.search(r'[\p{Hiragana}\p{Katakana}\p{Han}]', text))


@cache
def get_japanese_tagger():
    """
    Get a cached instance of the fugashi Japanese morphological analyzer.

    Fugashi is a Python wrapper for MeCab, a morphological analyzer for Japanese.
    The tagger is configured with the '-Owakati' option to output space-separated
    words without part-of-speech information. The instance is cached to avoid
    repeated initialization overhead.

    Returns:
        fugashi.Tagger instance configured for word segmentation

    Raises:
        ImportError: If the fugashi library or unidic-lite dictionary is not installed
    """
    try:
        from fugashi import Tagger
    except ImportError:
        print("Error: fugashi library not found. Please install it with: pip install 'fugashi[unidic-lite]'")
        raise

    # -Owakati: Output format that produces space-separated words only
    return Tagger('-Owakati')


def segment_japanese(text: str) -> list[str]:
    """
    Segment Japanese text into space-separated words using morphological analysis.

    Uses MeCab via fugashi to perform morphological analysis and word segmentation
    of Japanese text. This handles the complex task of word boundary detection in
    Japanese, which doesn't use spaces between words.

    Args:
        text: The Japanese text to segment

    Returns:
        List of Japanese words

    Example:
        >>> segment_japanese("私は学生です")
        "私 は 学生 です"
    """
    tagger = get_japanese_tagger()
    # Parse the text and return space-separated morphemes
    return [str(word) for word in tagger(text)]
