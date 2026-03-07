"""Rule-based post-generation quality filters for Phase 1 descriptors.

All filters are deterministic string/regex operations — no LLM calls.
Each filter returns ``(passed: bool, reason: str | None)``.
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

_DESC_MIN_WORDS = 5
_DESC_MAX_WORDS = 60
_SCENE_MIN_WORDS = 40
_SCENE_MAX_WORDS = 250

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


def _stem_present(word: str, text_lower: str) -> bool:
    """Return True if ``word`` (or a stem of it) appears in ``text_lower``.

    Uses a simple prefix match: the seed word must appear as a prefix of some
    whitespace-delimited token (handles common English inflections like
    ``climb`` → ``climbed``, ``run`` → ``running``).

    Also handles y→ies plurals (``butterfly`` → ``butterflies``).
    """
    # Exact prefix match first (fastest path).
    pattern = re.compile(r"\b" + re.escape(word) + r"\w*", re.IGNORECASE)
    if pattern.search(text_lower):
        return True
    # Handle y→ies plurals: "butterfly" should match "butterflies".
    if word.endswith("y"):
        stem = word[:-1]
        pattern_ies = re.compile(r"\b" + re.escape(stem) + r"ies\b", re.IGNORECASE)
        if pattern_ies.search(text_lower):
            return True
    return False


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


def strip_markdown(text: str) -> str:
    """Remove common markdown formatting artifacts from generated text.

    Strips thinking blocks (``<think>...</think>``), bold, italic, and
    heading markers while preserving the underlying content.
    """
    # Strip <think>...</think> reasoning blocks (e.g. Qwen3, DeepSeek).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also strip an unclosed <think> block (model hit max tokens mid-thought).
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    # Bold: **word** or __word__
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Italic: *word* or _word_ (single, not inside a word)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Heading markers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text



# ---------------------------------------------------------------------------
# Descriptor filters
# ---------------------------------------------------------------------------

# Narrative indicators — descriptors should NOT read like stories.
_NARRATIVE_PATTERNS = [
    re.compile(r"(?i)\bonce upon\b"),
    re.compile(r"(?i)\bone day\b"),
    re.compile(r"(?i)\bone morning\b"),
    re.compile(r"(?i)\bone night\b"),
    re.compile(r"(?i)\blong ago\b"),
    re.compile(r"(?i)\bthere was\b"),
    re.compile(r"(?i)\bthere were\b"),
    re.compile(r"(?i)\bthere lived\b"),
    re.compile(r"(?i)\bnamed\s+[A-Z]"),  # character names
    re.compile(r"(?i)\bcalled\s+[A-Z]"),
]


def check_descriptor_word_count(text: str) -> tuple[bool, Optional[str]]:
    """Reject if word count is outside [5, 60] for descriptors."""
    count = len(text.split())
    if count < _DESC_MIN_WORDS:
        return False, f"word_count_too_low:{count}"
    if count > _DESC_MAX_WORDS:
        return False, f"word_count_too_high:{count}"
    return True, None


def check_concept_mention(text: str, noun: str) -> tuple[bool, Optional[str]]:
    """Reject if the target noun does not appear in the text (stem-matched)."""
    if not _stem_present(noun.lower(), text.lower()):
        return False, f"missing_concept:{noun}"
    return True, None


def check_no_narrative(text: str) -> tuple[bool, Optional[str]]:
    """Reject if the text contains story/narrative indicators."""
    for pattern in _NARRATIVE_PATTERNS:
        if pattern.search(text):
            return False, f"narrative:{pattern.pattern!r}"
    return True, None


def check_factual_grounding(text: str, value: str) -> tuple[bool, Optional[str]]:
    """Reject if the relation value does not appear in the text (stem-matched).

    For multi-word values, checks if any significant word (>3 chars) appears.
    """
    text_lower = text.lower()
    value_lower = value.lower()
    # For short single-word values, do direct stem match.
    words = value_lower.split()
    if len(words) == 1:
        if _stem_present(value_lower, text_lower):
            return True, None
        return False, f"missing_value:{value}"

    # For multi-word values, require at least one significant word to appear.
    significant = [w for w in words if len(w) > 3]
    if not significant:
        # All short words — check the full phrase via substring.
        if value_lower in text_lower:
            return True, None
        return False, f"missing_value:{value}"

    for word in significant:
        if _stem_present(word, text_lower):
            return True, None
    return False, f"missing_value:{value}"


def run_descriptor_filters(
    text: str,
    noun: str,
    relation: str,
    value: str,
    minhash_index: Optional[object] = None,
) -> tuple[bool, list[str], str]:
    """Run all filters against a descriptor.

    Args:
        text: Generated descriptor text.
        noun: The target concept noun.
        relation: The relation type (e.g. "IsA").
        value: The relation value (e.g. "animal").
        minhash_index: Optional MinHashLSH for near-duplicate detection.

    Returns:
        ``(passed, reasons, cleaned_text)``
    """
    text = strip_markdown(text)

    checks = [
        check_descriptor_word_count(text),
        check_concept_mention(text, noun),
        check_no_dialogue(text),
        check_no_markers(text),
        check_no_narrative(text),
        check_factual_grounding(text, value),
        check_near_duplicate(text, minhash_index),
    ]

    reasons = [reason for passed, reason in checks if not passed and reason is not None]
    return len(reasons) == 0, reasons, text


# ---------------------------------------------------------------------------
# Scene filters (Phase 2)
# ---------------------------------------------------------------------------


def check_scene_word_count(text: str) -> tuple[bool, Optional[str]]:
    """Reject if word count is outside [40, 250] for scenes."""
    count = len(text.split())
    if count < _SCENE_MIN_WORDS:
        return False, f"word_count_too_low:{count}"
    if count > _SCENE_MAX_WORDS:
        return False, f"word_count_too_high:{count}"
    return True, None


def check_concept_mentions(
    text: str, nouns: list[str],
) -> tuple[bool, Optional[str]]:
    """Reject if fewer than half of the target nouns appear in the text."""
    text_lower = text.lower()
    found = sum(1 for noun in nouns if _stem_present(noun.lower(), text_lower))
    # Require at least half of the target nouns (rounded up).
    required = (len(nouns) + 1) // 2
    if found < required:
        missing = [n for n in nouns if not _stem_present(n.lower(), text_lower)]
        return False, f"missing_concepts:{','.join(missing)}"
    return True, None


def run_scene_filters(
    text: str,
    concept_pairs: list[tuple[str, str, str]],
    minhash_index: Optional[object] = None,
) -> tuple[bool, list[str], str]:
    """Run all filters against a Phase 2 everyday scene.

    Args:
        text: Generated scene text.
        concept_pairs: List of (noun, relation, value) triples targeted.
        minhash_index: Optional MinHashLSH for near-duplicate detection.

    Returns:
        ``(passed, reasons, cleaned_text)``
    """
    text = strip_markdown(text)

    nouns = list({noun for noun, _, _ in concept_pairs})

    checks = [
        check_scene_word_count(text),
        check_concept_mentions(text, nouns),
        check_no_dialogue(text),
        check_no_markers(text),
        check_no_narrative(text),
        check_near_duplicate(text, minhash_index),
    ]

    reasons = [reason for passed, reason in checks if not passed and reason is not None]
    return len(reasons) == 0, reasons, text


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
        """Record the result of filtering one descriptor.

        Args:
            passed: Whether the descriptor passed all filters.
            reasons: List of failure reason strings from ``run_descriptor_filters``.
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
