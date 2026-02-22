"""Word triplet sampling for story generation."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

_VOCAB_DIR = Path(__file__).parent / "vocab"

_PHASE_FILES: dict[int, dict[str, str]] = {
    1: {
        "nouns": "phase1_nouns.txt",
        "verbs": "phase1_verbs.txt",
        "adjs": "phase1_adjs.txt",
    },
}

# Track which triplets have been sampled for coverage analysis.
_triplet_usage: dict[tuple[str, str, str], int] = defaultdict(int)


def load_vocab(phase: int) -> dict[str, list[str]]:
    """Load vocabulary lists for the given phase.

    Args:
        phase: The curriculum phase number (e.g. 1).

    Returns:
        A dict with keys "nouns", "verbs", "adjs", each mapping to a list of words.

    Raises:
        ValueError: If the phase is not supported.
        FileNotFoundError: If a vocab file is missing.
    """
    if phase not in _PHASE_FILES:
        raise ValueError(f"Unsupported phase: {phase}. Supported phases: {list(_PHASE_FILES)}")

    vocab: dict[str, list[str]] = {}
    for key, filename in _PHASE_FILES[phase].items():
        path = _VOCAB_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Vocab file not found: {path}")
        words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        vocab[key] = words

    return vocab


def sample_triplet(
    vocab: dict[str, list[str]] | None = None,
    phase: int = 1,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Randomly sample one noun, one verb, and one adjective.

    Sampling is with replacement — the vocab is reused across many stories.
    Each sampled triplet is recorded in the module-level usage tracker.

    Args:
        vocab: Pre-loaded vocab dict (from ``load_vocab``). If None, loads phase vocab.
        phase: Phase to load vocab for (only used when ``vocab`` is None).
        rng: Optional ``random.Random`` instance for reproducibility.

    Returns:
        A dict ``{"noun": str, "verb": str, "adj": str}``.
    """
    if vocab is None:
        vocab = load_vocab(phase)

    _rng = rng or random

    noun = _rng.choice(vocab["nouns"])
    verb = _rng.choice(vocab["verbs"])
    adj = _rng.choice(vocab["adjs"])

    triplet = (noun, verb, adj)
    _triplet_usage[triplet] += 1

    return {"noun": noun, "verb": verb, "adj": adj}


def get_triplet_coverage() -> dict[str, int | float]:
    """Return coverage statistics for sampled triplets.

    Returns:
        A dict with:
        - ``total_samples``: total number of triplets sampled so far.
        - ``unique_triplets``: number of distinct (noun, verb, adj) combinations seen.
        - ``max_reuse``: highest reuse count for any single triplet.
    """
    if not _triplet_usage:
        return {"total_samples": 0, "unique_triplets": 0, "max_reuse": 0}

    total = sum(_triplet_usage.values())
    unique = len(_triplet_usage)
    max_reuse = max(_triplet_usage.values())

    return {
        "total_samples": total,
        "unique_triplets": unique,
        "max_reuse": max_reuse,
    }


def reset_triplet_coverage() -> None:
    """Clear the triplet usage tracker (useful between runs or in tests)."""
    _triplet_usage.clear()
