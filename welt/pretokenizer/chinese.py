"""
Chinese text pretokenization utilities.

This module provides functions for detecting and segmenting Chinese text using the jieba
library for word segmentation.
"""

from functools import cache

import regex


def has_chinese(text: str) -> bool:
    """
    Check if the given text contains Chinese characters.

    Uses Unicode Han ideograph property to detect Chinese characters including:
    - CJK Unified Ideographs (U+4E00-U+9FFF)
    - CJK Extension A, B, C, D, E, F, and G
    - CJK Compatibility Ideographs

    Args:
        text: The input text to check for Chinese characters

    Returns:
        True if Chinese characters are found, False otherwise
    """
    # Match any Han ideograph using Unicode property
    return bool(regex.search(r'[\p{Han}]', text))


@cache
def get_chinese_segmenter():
    """
    Get a cached instance of the jieba Chinese word segmenter.

    Jieba is a popular Chinese text segmentation library that uses a combination of
    dictionary-based matching and statistical models to segment Chinese text into words.
    The segmenter is cached to avoid repeated initialization overhead.

    Returns:
        jieba module instance for text segmentation

    Raises:
        ImportError: If the jieba library is not installed
    """
    try:
        import jieba
    except ImportError:
        print("Error: jieba library not found. Please install it with: pip install jieba")
        raise

    return jieba


def segment_chinese(text: str) -> list[str]:
    """
    Segment Chinese text into space-separated words.

    Uses jieba's precise segmentation mode to break Chinese text into individual words,
    then joins them with spaces. This preprocessing step helps the tokenizer better
    understand Chinese text structure.

    Args:
        text: The Chinese text to segment

    Returns:
        List of Chinese words

    Example:
        >>> segment_chinese("我爱北京天安门")
        "我 爱 北京 天安门"
    """
    jieba = get_chinese_segmenter()
    # Use jieba.cut() for precise segmentation and join with spaces
    segments = jieba.cut(text)
    # Filter out empty segments and join with single spaces
    return list(segments)
