"""Text normalisation and CER: the definition every evaluation number depends on."""

from __future__ import annotations

import pytest

from sichuan_speech.text import (
    CerStats,
    cer,
    char_error_stats,
    corpus_cer,
    edit_distance,
    normalize_text,
    strip_special_tokens,
)


def test_strip_special_tokens_removes_rich_transcription_markers():
    assert strip_special_tokens("<|zh|><|NEUTRAL|>今天天气好") == "今天天气好"
    assert strip_special_tokens("<|woitn|>") == ""


def test_normalize_text_drops_punctuation_whitespace_and_zero_width():
    assert normalize_text("今天天气好，我们出去耍一哈嘛。") == "今天天气好我们出去耍一哈嘛"
    assert normalize_text(" a b\tc\n") == "abc"
    assert normalize_text("a\u200bb\ufeffc") == "abc"
    assert normalize_text("（四）和（十）") == "四和十"
    assert normalize_text("（四）十") == "四十"


def test_cer_identical_is_zero():
    assert cer("今天天气好", "今天天气好") == 0.0


def test_cer_counts_one_substitution_over_length():
    assert cer("今天天气好", "今天天汽好") == pytest.approx(1 / 5)


def test_cer_handles_insertion_and_deletion():
    assert cer("今天好", "今天很好") == pytest.approx(1 / 3)
    assert cer("今天很好", "今天好") == pytest.approx(1 / 4)


def test_cer_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "有字") == 1.0
    assert cer("，。", "") == 0.0


def test_cer_can_skip_normalisation():
    assert cer("你好，", "你好", normalize=False) == pytest.approx(1 / 3)


def test_char_error_stats_reports_ref_length_after_normalisation():
    stats = char_error_stats("今天天气好！", "今天天气好")
    assert isinstance(stats, CerStats)
    assert stats.errors == 0
    assert stats.ref_chars == 5
    assert stats.cer == 0.0


def test_corpus_cer_is_micro_averaged():
    pairs = [("今天天气好", "今天天汽好"), ("你好", "你好")]
    # 1 error over 5 + 2 = 7 characters
    assert corpus_cer(pairs) == pytest.approx(1 / 7)


def test_edit_distance_is_symmetric_for_equal_lengths():
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("abd", "abc") == 1
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3


def test_edit_distance_uses_linear_memory_not_quadratic():
    # A 20k-character pair would allocate 400M cells with a full matrix; this must finish.
    reference = "今天天气好安逸哦我们出去耍一哈嘛" * 500
    assert edit_distance(reference, reference) == 0
