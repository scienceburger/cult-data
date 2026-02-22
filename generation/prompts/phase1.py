"""Phase 1 prompt template and narrative feature definitions."""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Narrative features
# ---------------------------------------------------------------------------

NARRATIVE_FEATURES: dict[str, str] = {
    "plot_twist": "an unexpected turn of events that surprises the reader",
    "moral": "a clear lesson or moral that the main character learns",
    "foreshadowing": "hints early in the story about what will happen later",
    "bad_ending": "an ending where things don't work out well for the main character",
    "conflict": "a conflict or disagreement between two characters that drives the story",
}

# Keep a stable ordered list for reproducible random selection.
_FEATURE_KEYS: list[str] = list(NARRATIVE_FEATURES.keys())

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_TEMPLATE = """\
Write a short story for young children (ages 3-5).

The story must include these three words: {noun}, {verb}, {adjective}.

The story should feature: {narrative_feature_description}.

Rules:
- Use simple words that a young child would understand
- Write in third person (he, she, they — not "I")
- Do not include any dialogue or quoted speech
- Use complete sentences with correct grammar
- Keep the story between 150 and 250 words
- Use past tense
- Do not include a title, headers, markers, or placeholders

Write only the story, with no title or commentary.\
"""

PROMPT_TEMPLATE_VERSION = "p1-v1"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def random_feature(rng: random.Random | None = None) -> str:
    """Return a randomly selected narrative feature key (uniform distribution).

    Args:
        rng: Optional ``random.Random`` instance for reproducibility.

    Returns:
        One of the keys from ``NARRATIVE_FEATURES``.
    """
    _rng = rng or random
    return _rng.choice(_FEATURE_KEYS)


def build_prompt(triplet: dict[str, str], feature: str) -> str:
    """Fill the Phase 1 prompt template with a word triplet and narrative feature.

    Args:
        triplet: Dict with keys ``"noun"``, ``"verb"``, ``"adj"``.
        feature: A key from ``NARRATIVE_FEATURES``.

    Returns:
        The fully formatted prompt string ready to send to the model.

    Raises:
        KeyError: If ``feature`` is not a valid narrative feature key.
        KeyError: If ``triplet`` is missing required keys.
    """
    if feature not in NARRATIVE_FEATURES:
        raise KeyError(
            f"Unknown narrative feature: {feature!r}. "
            f"Valid features: {list(NARRATIVE_FEATURES)}"
        )

    return _TEMPLATE.format(
        noun=triplet["noun"],
        verb=triplet["verb"],
        adjective=triplet["adj"],
        narrative_feature_description=NARRATIVE_FEATURES[feature],
    )
