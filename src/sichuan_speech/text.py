"""Transcript normalisation and character error rate (CER).

The evaluation protocol is strict about tokenisation: punctuation, whitespace and
SenseVoice's ``<|...|>`` control tokens never count as characters. Every reported
number in ``results/`` was produced with exactly this normalisation, so changing
it changes the numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# SenseVoice emits control tokens such as <|zh|>, <|Speech|>, <|NEUTRAL|>, <|woitn|>.
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")

# Punctuation + whitespace stripped before scoring. The character class is the one
# used for every CER number in this repository; keep it byte-for-byte stable.
PUNCTUATION_RE = re.compile(
    r"[\s，。、？！；：,.?!;:\"'“”‘’（）()《》<>·—\-\u200b\u200c\u200d\ufeff]"
)


def strip_special_tokens(text: str) -> str:
    """Remove ``<|...|>`` control tokens emitted by SenseVoice."""
    return SPECIAL_TOKEN_RE.sub("", text)


def normalize_text(text: str) -> str:
    """Normalise a transcript for CER scoring.

    Removes control tokens, punctuation, whitespace and zero-width characters.
    Case is preserved (Chinese text has no case; mixed Latin text is rare and
    case-sensitive comparison is the conservative choice).
    """
    return PUNCTUATION_RE.sub("", strip_special_tokens(text))


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance over Unicode code points, O(len(hyp)) memory."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_char in enumerate(hypothesis, start=1):
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + (ref_char != hyp_char),  # substitution
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class CerStats:
    """Edit operations and reference length for one utterance."""

    errors: int
    ref_chars: int

    @property
    def cer(self) -> float:
        if self.ref_chars == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.ref_chars


def char_error_stats(reference: str, hypothesis: str, *, normalize: bool = True) -> CerStats:
    """Return edit counts for a single utterance."""
    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    return CerStats(errors=edit_distance(ref, hyp), ref_chars=len(ref))


def cer(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """Character error rate for one utterance.

    An empty reference yields 0.0 for an empty hypothesis and 1.0 otherwise
    (the historical scripts divided by ``max(1, len(ref))``; this makes the
    degenerate case explicit without changing any non-empty result).
    """
    return char_error_stats(reference, hypothesis, normalize=normalize).cer


def corpus_cer(pairs: list[tuple[str, str]], *, normalize: bool = True) -> float:
    """Micro-averaged CER over ``(reference, hypothesis)`` pairs.

    Micro-averaging (sum of edits / sum of reference characters) is what the
    evaluation scripts report; it weights long utterances more than short ones.
    """
    total_errors = 0
    total_chars = 0
    for reference, hypothesis in pairs:
        stats = char_error_stats(reference, hypothesis, normalize=normalize)
        total_errors += stats.errors
        total_chars += stats.ref_chars
    if total_chars == 0:
        return 0.0 if total_errors == 0 else 1.0
    return total_errors / total_chars
