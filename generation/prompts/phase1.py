"""Phase 1 prompt templates for descriptor generation.

Generates factual sentences about concept × relation triples for young children.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Relation verb mappings — converts relation keys to natural language
# ---------------------------------------------------------------------------

RELATION_VERBS: dict[str, str] = {
    "IsA": "is a",
    "Does": "",  # value is already a verb phrase
    "Has": "has",
    "Likes": "likes",
    "AtLocation": "can be found at",
    "Sound": "makes the sound",
    "Size": "is",
    "Color": "can be",
    "Taste": "tastes",
    "Shape": "is shaped like a",
}

# ---------------------------------------------------------------------------
# Descriptor prompt templates
# ---------------------------------------------------------------------------

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
- Vary sentence structure (don't always start with "A {noun}...")
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
- Try different ways to start your sentences
- Do not use markdown formatting — no asterisks, bold, or italics
- Do not include headers, titles, or labels
- Write only the sentences, nothing else\
""",
    "p1-desc-v3": """\
Tell a young child (age 3-5) a simple fact.

Concept: {noun}
Fact: {fact_sentence}

Write 1-3 short sentences about this fact. Use simple words. Do not tell a story. Do not use dialogue. Use present tense. No markdown. Write only the sentences.\
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
        'A dog is a animal'
        >>> relation_to_sentence("dog", "Does", "bark")
        'A dog barks'
        >>> relation_to_sentence("dog", "Has", "four legs")
        'A dog has four legs'
    """
    verb = RELATION_VERBS.get(relation, relation.lower())
    if relation == "Does":
        return f"A {noun} {value}"
    return f"A {noun} {verb} {value}"


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
) -> str:
    """Build a descriptor prompt from a (noun, relation, value) triple.

    Args:
        noun: The target concept (e.g. "dog").
        relation: The relation type (e.g. "IsA", "Has", "Does").
        value: The relation value (e.g. "animal", "four legs", "bark").
        version: Prompt template version. Defaults to DEFAULT_DESCRIPTOR_VERSION.

    Returns:
        The fully formatted prompt string.
    """
    template, _ = get_descriptor_template(version)
    fact_sentence = relation_to_sentence(noun, relation, value)
    return template.format(noun=noun, fact_sentence=fact_sentence)
