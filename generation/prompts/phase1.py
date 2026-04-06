"""Phase 1 prompt templates for descriptor generation.

Generates factual sentences about concept × relation triples for young children.
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Relation verb mappings — converts relation keys to natural language
# ---------------------------------------------------------------------------

RELATION_VERBS: dict[str, str] = {
    "IsA": "is a",
    "Does": "",  # value is already a verb phrase — conjugated by _conjugate_s()
    "Has": "has",
    "Likes": "likes",
    "AtLocation": "can be found at",
    "Sound": "makes the sound",
    "Size": "is",
    "Color": "can be",
    "Taste": "tastes",
    "Shape": "is shaped like a",
}

# Nouns that should use no article or a plural form ("boots are", not "a boots is").
_PLURAL_NOUNS: set[str] = {
    "boots", "gloves", "socks", "pants", "teeth", "blocks",
}

# Nouns that are uncountable or abstract — use no article.
_UNCOUNTABLE_NOUNS: set[str] = {
    "ice", "mud", "snow", "fog", "rain", "water", "grass", "rice", "corn",
    "ice cream", "bread", "cheese", "milk", "juice", "soup", "pasta", "honey",
    "lightning", "thunder", "wind", "morning", "night", "bedtime",
    "breakfast", "lunch", "dinner", "snack",
    "sleeping", "reading", "running", "dancing", "singing", "playing",
    "swimming", "cooking", "shopping", "drawing", "painting", "helping",
    "sharing", "eating", "brushing teeth", "taking a bath",
    "happy", "sad", "tired", "hungry", "scared", "angry", "surprised",
    "red", "blue", "green", "yellow", "pink", "purple",
    "christmas", "halloween",
}


def _subject_phrase(noun: str) -> str:
    """Build a grammatical subject phrase for a noun.

    Returns e.g. "A dog", "Boots", "Ice", "A birthday".
    """
    lower = noun.lower()
    if lower in _PLURAL_NOUNS:
        return noun.capitalize()
    if lower in _UNCOUNTABLE_NOUNS:
        return noun[0].upper() + noun[1:]
    return f"A {noun}"


def _conjugate_s(verb_phrase: str, plural: bool) -> str:
    """Add third-person singular -s/-es to the first verb when *not* plural.

    Handles common patterns: run→runs, fly→flies, catch→catches, go→goes.
    Skips phrases that already look conjugated or start with "can"/"will"/etc.
    """
    if plural:
        return verb_phrase  # plural subject: "Boots keep your feet warm"
    parts = verb_phrase.split(None, 1)
    if not parts:
        return verb_phrase
    verb = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    # Don't touch modals or already-conjugated verbs.
    if verb in ("can", "will", "could", "would", "should", "may", "might", "must"):
        return verb_phrase
    if verb.endswith("s") and not verb.endswith("ss"):
        return verb_phrase  # likely already conjugated
    # Conjugation rules.
    if verb.endswith(("sh", "ch", "x", "z", "ss", "o")):
        conjugated = verb + "es"
    elif verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        conjugated = verb[:-1] + "ies"
    else:
        conjugated = verb + "s"
    return f"{conjugated} {rest}".strip()


def _prep_for_location(value: str) -> str:
    """Choose a natural preposition + article for AtLocation values.

    Returns e.g. "in a park", "at home", "on a leaf".
    """
    lower = value.lower().strip()
    # Places that need no article.
    no_article = {"home", "school", "outside", "inside", "work"}
    if lower in no_article:
        return f"at {lower}"
    # Common "in" locations.
    in_locations = {
        "water", "forest", "garden", "ocean", "river", "lake", "pond",
        "mud", "snow", "bathroom", "kitchen", "bedroom", "house",
        "hospital", "library", "store", "zoo", "farm", "park",
        "neighborhood", "dining room", "building", "sky",
    }
    on_locations = {"leaf", "foot", "leg", "tree", "ground", "table"}
    if lower in in_locations:
        return f"in a {lower}"
    if lower in on_locations:
        return f"on a {lower}"
    return f"at a {lower}"

# ---------------------------------------------------------------------------
# Descriptor prompt templates
# ---------------------------------------------------------------------------

_SENTENCE_STARTERS: list[str] = [
    "Start your sentences in different ways.",
    "Try starting with the fact itself, not the concept name.",
    "Try beginning with \"Did you know\" or \"Some\" or \"Most\".",
    "Try starting with where, when, or how.",
    "Begin by describing what you can see or hear.",
    "Start with what makes this special or interesting.",
]

_DESCRIPTOR_TEMPLATES: dict[str, str] = {
    "p1-desc-v1": """\
Write simple factual sentences about the following concept for young children (ages 3-5).

Concept: {noun}
Fact: {fact_sentence}

Rules:
- Write 1-3 short, simple sentences stating this fact
- Use simple words a young child would understand
- Do not use any narrative or storytelling
- Do not include dialogue
- Use present tense for general facts
- {variation_hint}
- Do not use markdown formatting
- Write only the sentences, nothing else\
""",
    "p1-desc-v2": """\
Write simple factual sentences about the following concept for young children (ages 3-5).

Concept: {noun}
Fact: {fact_sentence}

Rules:
- Write 1-3 short, simple sentences stating this fact
- Use simple words a young child would understand
- Do not use any narrative or storytelling
- Do not include dialogue or quoted speech
- Use present tense
- {variation_hint}
- Do not use markdown formatting — no asterisks, bold, or italics
- Do not include headers, titles, or labels
- Write only the sentences, nothing else\
""",
    "p1-desc-v3": """\
Tell a young child (age 3-5) a simple fact.

Concept: {noun}
Fact: {fact_sentence}

Write 1-3 short sentences about this fact. Use simple words. Do not tell a story. Do not use dialogue. Use present tense. No markdown. {variation_hint} Write only the sentences.\
""",
}

DEFAULT_DESCRIPTOR_VERSION = "p1-desc-v1"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def relation_to_sentence(noun: str, relation: str, value: str) -> str:
    """Convert a (noun, relation, value) triple into a natural-language fact sentence.

    Examples:
        >>> relation_to_sentence("dog", "IsA", "animal")
        'A dog is an animal'
        >>> relation_to_sentence("dog", "Does", "bark")
        'A dog barks'
        >>> relation_to_sentence("boots", "AtLocation", "foot")
        'Boots can be found on a foot'
    """
    subject = _subject_phrase(noun)
    plural = noun.lower() in _PLURAL_NOUNS
    verb = RELATION_VERBS.get(relation, relation.lower())

    if relation == "Does":
        conjugated = _conjugate_s(value, plural)
        return f"{subject} {conjugated}"
    if relation == "AtLocation":
        loc = _prep_for_location(value)
        return f"{subject} can be found {loc}"
    if relation == "IsA":
        article = "an" if value and value[0].lower() in "aeiou" else "a"
        return f"{subject} is {article} {value}"
    return f"{subject} {verb} {value}"


def get_descriptor_template(version: str | None = None) -> tuple[str, str]:
    """Return (template_string, version) for a descriptor prompt.

    Raises:
        KeyError: If the version is unknown.
    """
    v = version or DEFAULT_DESCRIPTOR_VERSION
    if v not in _DESCRIPTOR_TEMPLATES:
        raise KeyError(
            f"Unknown descriptor prompt version: {v!r}. "
            f"Valid versions: {list(_DESCRIPTOR_TEMPLATES)}"
        )
    return _DESCRIPTOR_TEMPLATES[v], v


def build_descriptor_prompt(
    noun: str,
    relation: str,
    value: str,
    version: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Build a descriptor prompt from a (noun, relation, value) triple.

    Args:
        noun: The target concept (e.g. "dog").
        relation: The relation type (e.g. "IsA", "Has", "Does").
        value: The relation value (e.g. "animal", "four legs", "bark").
        version: Prompt template version. Defaults to DEFAULT_DESCRIPTOR_VERSION.
        rng: Random instance for selecting variation hints.

    Returns:
        The fully formatted prompt string.
    """
    template, _ = get_descriptor_template(version)
    fact_sentence = relation_to_sentence(noun, relation, value)
    r = rng or random.Random()
    hint = r.choice(_SENTENCE_STARTERS)
    return template.format(noun=noun, fact_sentence=fact_sentence, variation_hint=hint)
