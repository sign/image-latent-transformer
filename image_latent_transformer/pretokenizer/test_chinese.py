import pytest

from image_latent_transformer.pretokenizer.chinese import has_chinese, segment_chinese


def test_has_chinese_simple():
    """Test has_chinese with simple Chinese characters."""
    assert has_chinese("你好")
    assert has_chinese("中文")
    assert has_chinese("我来到北京清华大学")


def test_has_chinese_mixed_content():
    """Test has_chinese with mixed Chinese and other characters."""
    assert has_chinese("hello 你好")
    assert has_chinese("mixed 你好")
    assert has_chinese("123 中文 abc")
    assert has_chinese("English 中文 混合")


def test_has_chinese_no_chinese():
    """Test has_chinese with non-Chinese text."""
    assert not has_chinese("hello")
    assert not has_chinese("English text")
    assert not has_chinese("123456")
    assert not has_chinese("!@#$%^&*()")
    assert not has_chinese("こんにちは")  # Japanese
    assert not has_chinese("עברית")  # Hebrew


def test_has_chinese_empty_string():
    """Test has_chinese with empty string."""
    assert not has_chinese("")


def test_has_chinese_whitespace_only():
    """Test has_chinese with whitespace only."""
    assert not has_chinese(" ")
    assert not has_chinese("\n\t")
    assert not has_chinese("   ")


def test_segment_chinese_simple():
    """Test segment_chinese with simple Chinese text."""
    result = segment_chinese("你好")
    assert result == "你好"


def test_segment_chinese_mixed():
    """Test segment_chinese with mixed Chinese and English."""
    result = segment_chinese("hello 我来到北京清华大学 world")
    assert result == "hello 我 来到 北京 清华大学 world"


def test_segment_chinese_english_only():
    """Test segment_chinese with English-only text."""
    result = segment_chinese("hello world")
    assert result == "hello world"


def test_segment_chinese_empty():
    """Test segment_chinese with empty string."""
    result = segment_chinese("")
    assert result == ""


def test_segment_chinese_complex():
    """Test segment_chinese with complex Chinese sentence."""
    result = segment_chinese("小明硕士毕业于中国科学院计算所")
    assert result == "小明 硕士 毕业 于 中国科学院 计算所"


def test_segment_chinese_compound_words():
    """Test segment_chinese with compound words."""
    result = segment_chinese("中文分词测试")
    assert result == "中文 分词 测试"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
