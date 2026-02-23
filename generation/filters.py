"""Rule-based post-generation quality filters for Phase 1 stories.

All filters are deterministic string/regex operations — no LLM calls.
Each filter returns ``(passed: bool, reason: str | None)``.
``run_all_filters`` aggregates all checks and returns overall pass status
plus a list of failure reasons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from datasketch import MinHash, MinHashLSH

    _DATASKETCH_AVAILABLE = True
except ImportError:
    _DATASKETCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_WORDS = 150
_MAX_WORDS = 250

# Structural markers / placeholders that should not appear in clean output.
_MARKER_PATTERNS = [
    re.compile(r"\["),
    re.compile(r"\]"),
    re.compile(r"\*"),
    re.compile(r"#"),
    re.compile(r"---"),
    re.compile(r"(?i)\bTitle\s*:"),
    re.compile(r"(?i)\bStory\s*:"),
    re.compile(r"(?i)\bThe\s+End\b"),
]

# Quotation mark characters that indicate dialogue.
# Single curly quotes (\u2018, \u2019) are intentionally excluded — they also
# appear in contractions (it's, didn't) and possessives (dog's).
_QUOTE_CHARS = {'"', '\u201c', '\u201d'}  # straight " and curly " "

# Dialogue verbs — when these appear before a comma or period and are
# immediately followed (within the same sentence) by a capital letter word,
# they signal speech attribution.
_DIALOGUE_VERBS = {
    "said",
    "asked",
    "replied",
    "whispered",
    "shouted",
    "exclaimed",
    "cried",
    "answered",
    "called",
}

# Pattern: dialogue verb followed by a comma and then a capital letter word
# (speech attribution pattern like: "said, She" / "whispered, He").
_DIALOGUE_VERB_PATTERN = re.compile(
    r"\b(" + "|".join(_DIALOGUE_VERBS) + r")\b\s*,\s*[A-Z]",
    re.UNICODE,
)

# MinHash parameters.
_MINHASH_NUM_PERM = 128
_MINHASH_SIMILARITY_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------


def check_word_count(text: str) -> tuple[bool, Optional[str]]:
    """Reject if word count is outside [150, 250]."""
    count = len(text.split())
    if count < _MIN_WORDS:
        return False, f"word_count_too_low:{count}"
    if count > _MAX_WORDS:
        return False, f"word_count_too_high:{count}"
    return True, None


def _stem_present(word: str, text_lower: str) -> bool:
    """Return True if ``word`` (or a stem of it) appears in ``text_lower``.

    Uses a simple prefix match: the seed word must appear as a prefix of some
    whitespace-delimited token (handles common English inflections like
    ``climb`` → ``climbed``, ``run`` → ``running``).
    """
    # Exact whole-word match first (fastest path).
    pattern = re.compile(r"\b" + re.escape(word) + r"\w*", re.IGNORECASE)
    return bool(pattern.search(text_lower))


def check_triplet_words(text: str, triplet: dict[str, str]) -> tuple[bool, Optional[str]]:
    """Reject if any seed word from the triplet is absent (stem-matched)."""
    text_lower = text.lower()
    missing = []
    for key in ("noun", "verb", "adj"):
        word = triplet[key].lower()
        if not _stem_present(word, text_lower):
            missing.append(f"{key}={triplet[key]!r}")
    if missing:
        return False, "missing_triplet_words:" + ",".join(missing)
    return True, None


def check_no_dialogue(text: str) -> tuple[bool, Optional[str]]:
    """Reject if the story contains dialogue.

    Checks for:
    - Any quotation mark character (straight or curly).
    - Dialogue verb followed by comma + capital letter (speech attribution).
    """
    # Check for quote characters.
    for char in text:
        if char in _QUOTE_CHARS:
            return False, "dialogue:quote_characters"

    # Check for dialogue verb attribution pattern.
    if _DIALOGUE_VERB_PATTERN.search(text):
        return False, "dialogue:attribution_pattern"

    return True, None


def check_no_markers(text: str) -> tuple[bool, Optional[str]]:
    """Reject if structural markers or placeholders are present."""
    for pattern in _MARKER_PATTERNS:
        if pattern.search(text):
            return False, f"marker:{pattern.pattern!r}"
    return True, None


def _make_minhash(text: str) -> "MinHash":  # type: ignore[name-defined]
    """Create a MinHash signature from the text (character 3-grams)."""
    mh = MinHash(num_perm=_MINHASH_NUM_PERM)
    text_lower = text.lower()
    for i in range(len(text_lower) - 2):
        mh.update(text_lower[i : i + 3].encode("utf-8"))
    return mh


def check_near_duplicate(
    text: str,
    minhash_index: Optional["MinHashLSH"],  # type: ignore[name-defined]
) -> tuple[bool, Optional[str]]:
    """Reject if MinHash similarity > 0.8 against any story in ``minhash_index``.

    If ``datasketch`` is not installed or ``minhash_index`` is None, this
    filter is skipped (passes by default).

    Args:
        text: The story text to check.
        minhash_index: A ``MinHashLSH`` instance pre-populated with accepted
            stories, or ``None`` to skip the check.

    Returns:
        ``(True, None)`` if no near-duplicate found, or ``(False, reason)`` if
        a near-duplicate exists.
    """
    if not _DATASKETCH_AVAILABLE or minhash_index is None:
        return True, None

    mh = _make_minhash(text)
    results = minhash_index.query(mh)
    if results:
        return False, f"near_duplicate:matches={results[:3]}"
    return True, None


def add_to_minhash_index(
    text: str,
    story_id: str,
    minhash_index: "MinHashLSH",  # type: ignore[name-defined]
) -> None:
    """Insert a story into the MinHash index after it has been accepted.

    Args:
        text: The story text.
        story_id: A unique identifier string for this story.
        minhash_index: The ``MinHashLSH`` instance to update in-place.
    """
    if not _DATASKETCH_AVAILABLE:
        return
    mh = _make_minhash(text)
    minhash_index.insert(story_id, mh)


def create_minhash_index() -> Optional["MinHashLSH"]:  # type: ignore[name-defined]
    """Create and return a new empty ``MinHashLSH`` index.

    Returns ``None`` if ``datasketch`` is not installed.
    """
    if not _DATASKETCH_AVAILABLE:
        return None
    return MinHashLSH(threshold=_MINHASH_SIMILARITY_THRESHOLD, num_perm=_MINHASH_NUM_PERM)


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def run_all_filters(
    text: str,
    triplet: dict[str, str],
    minhash_index: Optional[object] = None,
) -> tuple[bool, list[str]]:
    """Run all filters against a story.

    Args:
        text: Generated story text.
        triplet: The word triplet used to generate this story (keys: noun, verb, adj).
        minhash_index: Optional ``MinHashLSH`` instance for near-duplicate detection.

    Returns:
        ``(passed, reasons)`` where ``passed`` is ``True`` only if all filters
        pass, and ``reasons`` is a list of failure reason strings (empty on pass).
    """
    checks = [
        check_word_count(text),
        check_triplet_words(text, triplet),
        check_no_dialogue(text),
        check_no_markers(text),
        check_near_duplicate(text, minhash_index),
    ]

    reasons = [reason for passed, reason in checks if not passed and reason is not None]
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Filter statistics tracker
# ---------------------------------------------------------------------------


@dataclass
class FilterStats:
    """Tracks pass/fail counts per filter across a generation run."""

    total: int = 0
    passed: int = 0

    # Per-filter failure counts keyed by the filter name prefix.
    failures: dict[str, int] = field(default_factory=dict)

    def record(self, passed: bool, reasons: list[str]) -> None:
        """Record the result of ``run_all_filters`` for one story.

        Args:
            passed: Whether the story passed all filters.
            reasons: List of failure reason strings from ``run_all_filters``.
        """
        self.total += 1
        if passed:
            self.passed += 1
        else:
            for reason in reasons:
                # Use the prefix before the first colon as the filter key.
                key = reason.split(":")[0]
                self.failures[key] = self.failures.get(key, 0) + 1

    @property
    def rejected(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Total: {self.total}  Passed: {self.passed}  Rejected: {self.rejected}  "
            f"Pass rate: {self.pass_rate:.1%}"
        ]
        if self.failures:
            lines.append("Rejection breakdown:")
            for key, count in sorted(self.failures.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {key}: {count}")
        return "\n".join(lines)
