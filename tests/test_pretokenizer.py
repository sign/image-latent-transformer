import pytest

from welt.pretokenizer.pretokenizer import (
    is_word_complete,
    text_to_words,
    utf8_chunks_grapheme_safe,
)


def test_utf8_chunks_english():
    """Test utf8_chunks_grapheme_safe with English text."""
    text = "hello world"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=5))
    assert chunks == ["hello", " worl", "d"]
    assert len(chunks) == 3
    assert all(len(chunk.encode('utf-8')) <= 5 for chunk in chunks)


def test_utf8_chunks_hebrew():
    """Test utf8_chunks_grapheme_safe with Hebrew text."""
    text = "עמית מוריוסף"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=8))
    assert chunks == ['עמית', ' מור', 'יוסף']
    assert "".join(chunks) == text
    assert len(chunks) == 3
    assert all(len(chunk.encode('utf-8')) <= 8 for chunk in chunks)


def test_utf8_chunks_emoji():
    """Test utf8_chunks_grapheme_safe with basic emoji."""
    text = "hello 😀 world"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=8))
    assert chunks == ['hello ', '😀 wor', 'ld']
    assert "".join(chunks) == text
    assert len(chunks) == 3
    assert all(len(chunk.encode('utf-8')) <= 8 for chunk in chunks)


def test_utf8_chunks_long_emoji_cluster():
    """Test utf8_chunks_grapheme_safe with complex emoji cluster."""
    text = "👩‍👩‍👧‍👦"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=5))
    assert chunks == [text]
    assert len(chunks) == 1
    assert len(text.encode('utf-8')) > 5


def test_utf8_chunks_single_grapheme():
    """Test special case of single grapheme cluster."""
    text = "a"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=16))
    assert chunks == ["a"]


def test_utf8_chunks_mixed_content():
    """Test with mixed English, Hebrew, and emoji."""
    text = "hello עמית 👋"
    chunks = list(utf8_chunks_grapheme_safe(text, max_bytes=10))
    assert "".join(chunks) == text


def test_text_to_words_json():
    """Test text_to_words with JSON string."""
    json_text = '{"name": "test", "value": 123}'
    words = text_to_words(json_text, max_bytes=10)
    assert words == ['{"name": ', '"test", ', '"value": ', "123}"]
    assert "".join(words) == json_text
    assert len(words) == 4


def test_text_to_words_long_string():
    """Test text_to_words with long string."""
    long_text = ("This is a very long string that should be split into multiple chunks "
                 "when processed by the text_to_words function with appropriate byte limits.")
    words = text_to_words(long_text, max_bytes=8)
    assert "".join(words) == long_text


def test_text_to_words_short_string():
    """Test text_to_words with short string."""
    short_text = "hi"
    words = text_to_words(short_text, max_bytes=16)
    assert words == ["hi"]


def test_text_to_words_multiline_code():
    """Test similar to the processor test example."""
    text = """
    def foo():
        return "bar"
    """.strip()
    words = text_to_words(text, max_bytes=10)
    assert words == ['def ', 'foo():\n', '        ', 'return ', '"bar"']
    assert "".join(words) == text
    assert len(words) == 5
    assert 'def ' in words
    assert '        ' in words


def test_text_to_words_whitespace():
    """Test proper whitespace handling."""
    text = "hello    world"
    words = text_to_words(text, max_bytes=8)
    assert words == ["hello ", "   ", "world"]
    assert "".join(words) == text
    assert len(words) == 3
    assert "   " in words


def test_text_to_words_mixed_unicode():
    """Test with mixed content including unicode."""
    text = "hello עמית! 🌟 {'key': 'value'}"
    words = text_to_words(text, max_bytes=10)
    assert "".join(words) == text


def test_text_to_words_empty():
    """Test with empty string."""
    words = text_to_words("", max_bytes=16)
    assert words == []


def test_text_to_words_only_whitespace():
    """Test with only whitespace."""
    text = "   \n\t  "
    words = text_to_words(text, max_bytes=16)
    assert text == words[0]
    assert "".join(words) == text
    assert len(words) == 1
    assert all(c.isspace() for c in words[0])


def test_text_to_words_json_unicode():
    """Test JSON containing unicode characters."""
    json_text = '{"message":"שלום world 🌍","count": 42}'
    words = text_to_words(json_text, max_bytes=6)
    assert "".join(words) == json_text
    assert len(words) > 5
    assert any("🌍" in word for word in words)


def test_is_word_complete_control_tokens():
    """Test is_word_complete with control tokens."""
    assert is_word_complete("\x01")
    assert is_word_complete("\x02")
    assert is_word_complete("\x03")
    assert is_word_complete("\x08")
    assert is_word_complete("\x7F")


def test_is_word_complete_words_with_space():
    """Test is_word_complete with words that have trailing space."""
    assert is_word_complete("hello ")
    assert is_word_complete("world ")
    assert is_word_complete("test ")
    assert is_word_complete("עמית ")
    assert is_word_complete("🌟 ")


def test_is_word_complete_incomplete_words():
    """Test is_word_complete with incomplete words (no trailing space)."""
    assert not is_word_complete("hello")
    assert not is_word_complete("world")
    assert not is_word_complete("test")
    assert not is_word_complete("עמית")
    assert not is_word_complete("🌟")


def test_is_word_complete_whitespace_only():
    """Test is_word_complete with whitespace-only strings."""
    assert not is_word_complete(" ")
    assert not is_word_complete("  ")
    assert not is_word_complete("\n")
    assert not is_word_complete("\t")
    assert not is_word_complete("   \n\t  ")


def test_is_word_complete_empty_string():
    """Test is_word_complete with empty string."""
    assert not is_word_complete("")

def test_is_word_complete_unicode_with_space():
    """Test is_word_complete with unicode characters and trailing space."""
    assert is_word_complete("שלום ")
    assert is_word_complete("مرحبا ")
    assert is_word_complete("こんにちは ")
    assert not is_word_complete("שלום")
    assert not is_word_complete("مرحبا")
    assert not is_word_complete("こんにちは")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
