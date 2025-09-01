import pytest

from image_latent_transformer.pretokenizer.chinese import segment_chinese
from image_latent_transformer.pretokenizer.japanese import has_japanese, segment_japanese


def test_has_japanese_hiragana():
    """Test has_japanese with Hiragana characters."""
    assert has_japanese("こんにちは")
    assert has_japanese("ひらがな")
    assert has_japanese("あいうえお")


def test_has_japanese_katakana():
    """Test has_japanese with Katakana characters."""
    assert has_japanese("カタカナ")
    assert has_japanese("コンピューター")
    assert has_japanese("アメリカ")


def test_has_japanese_kanji():
    """Test has_japanese with Kanji characters."""
    assert has_japanese("漢字")
    assert has_japanese("日本語")
    assert has_japanese("学生")


def test_has_japanese_mixed_content():
    """Test has_japanese with mixed Japanese and other characters."""
    assert has_japanese("hello こんにちは")
    assert has_japanese("私は学生です。")
    assert has_japanese("123 カタカナ abc")
    assert has_japanese("English 日本語 混合")


def test_has_japanese_no_japanese():
    """Test has_japanese with non-Japanese text."""
    assert not has_japanese("hello")
    assert not has_japanese("English text")
    assert not has_japanese("123456")
    assert not has_japanese("!@#$%^&*()")
    assert not has_japanese("עברית")
    assert not has_japanese("العربية")


def test_has_japanese_empty_string():
    """Test has_japanese with empty string."""
    assert not has_japanese("")


def test_has_japanese_whitespace_only():
    """Test has_japanese with whitespace only."""
    assert not has_japanese(" ")
    assert not has_japanese("\n\t")
    assert not has_japanese("   ")


def test_segment_japanese_simple():
    """Test segment_japanese with simple Japanese text."""
    result = segment_japanese("こんにちは")
    assert result == "こんにちは"


def test_segment_japanese_mixed():
    """Test segment_japanese with mixed Japanese and English."""
    result = segment_japanese("hello 私は学生です。 world")
    assert result == "hello 私 は 学生 です 。 world"


def test_segment_japanese_english_only():
    """Test segment_japanese with English-only text."""
    result = segment_japanese("hello world")
    assert result == "hello world"


def test_segment_japanese_empty():
    """Test segment_japanese with empty string."""
    result = segment_japanese("")
    assert result == ""


def test_segment_japanese_complex():
    """Test segment_japanese with complex Japanese sentence."""
    result = segment_japanese("私は東京大学の学生です。")
    assert result == "私 は 東京 大学 の 学生 です 。"


def test_segment_japanese_katakana():
    """Test segment_japanese with Katakana text."""
    result = segment_japanese("コンピューター")
    assert result == "コンピューター"


def test_japanese_does_not_interfere_with_chinese():
    """Test that Japanese segmentation does not interfere with Chinese segmentation on Han text."""
    han_text = "中国学生"

    # Segment with Chinese first
    chinese_result = segment_chinese(han_text)

    # Then segment the Chinese result with Japanese
    japanese_after_chinese = segment_japanese(chinese_result)

    # They should be the same - Japanese should not re-segment Chinese output
    assert chinese_result == japanese_after_chinese
    assert chinese_result == "中国 学生"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
