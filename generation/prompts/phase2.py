"""Phase 2 prompt templates for everyday event / scene generation.

Generates short everyday scenes (~100-200 words) exercising 2-4 concept-relation
pairs, grounded in recognizable routines. No dialogue, no narrative arcs.
"""

from __future__ import annotations

import random
from typing import Any

from .phase1 import relation_to_sentence

# ---------------------------------------------------------------------------
# Child name pool — culturally diverse, gender-balanced, simple
# ---------------------------------------------------------------------------

CHILD_NAMES: list[str] = [
    # English / British
    "Tom", "Sam", "Kate", "Jack", "Rose", "Ben", "Amy", "Max", "Nora", "Joe",
    "Finn", "Grace", "Will", "Nell", "Rory", "Beth", "Cal", "Dot", "Kit", "Bea",
    # North American
    "Avery", "River", "Sky", "Quinn", "Blake", "Casey", "Sage", "Wren", "Drew",
    "Chase", "Morgan", "Robin", "Lane", "Rue", "Dale",
    # Latin American / Hispanic
    "Carlos", "Sofia", "Ana", "Diego", "Lucia", "Marco", "Valentina", "Rafael",
    "Camila", "Mateo", "Isabela", "Tomas", "Renata", "Emilio", "Paloma",
    "Luca", "Esperanza", "Nico", "Dulce", "Belen",
    # Brazilian / Portuguese
    "Davi", "Beatriz", "Gael", "Vitoria", "Enzo", "Lorena", "Murilo", "Yasmin",
    # West African
    "Kofi", "Amara", "Kwame", "Nia", "Seun", "Yaw", "Abena", "Kojo", "Efua",
    "Chidi", "Ngozi", "Emeka", "Adaeze", "Tunde", "Funmi",
    # East African
    "Amani", "Baraka", "Imara", "Zawadi", "Jabari", "Zuri", "Kamau", "Wanjiru",
    "Aisha", "Tariq", "Fatuma", "Hamisi",
    # Southern African
    "Thabo", "Lindiwe", "Sipho", "Zanele", "Bongani", "Nomsa",
    # North African / Arab
    "Omar", "Leila", "Nadia", "Youssef", "Hana", "Kareem", "Salma", "Tarek",
    "Nour", "Rania", "Jad", "Lara", "Sami", "Dina",
    # Persian / Iranian
    "Dara", "Shirin", "Kian", "Nasrin", "Aryan", "Azadeh",
    # Turkish
    "Deniz", "Ece", "Berk", "Elif", "Kerem", "Ceren",
    # South Asian / Indian
    "Arjun", "Priya", "Rohan", "Anaya", "Leela", "Ravi", "Kavya", "Arnav",
    "Diya", "Veer", "Meera", "Aditya", "Ishaan", "Pooja", "Kabir",
    # Pakistani / Bangladeshi
    "Bilal", "Zainab", "Hamza", "Fatima", "Usman", "Mehak",
    # East Asian / Chinese
    "Wei", "Mei", "Lin", "Fang", "Jun", "Xiao", "Lan", "Bo", "Hui", "Ping",
    # Japanese
    "Yuki", "Kenji", "Hana", "Riku", "Sora", "Akira", "Yuna", "Ren", "Mio",
    # Korean
    "Ji", "Yuna", "Seo", "Minjun", "Jia", "Dohyun", "Yejin",
    # Southeast Asian / Vietnamese
    "Linh", "Minh", "Tuan", "Ngoc", "Khanh", "Thu",
    # Filipino
    "Bea", "Carlo", "Pia", "Niko", "Lena", "Rico",
    # Thai / Indonesian / Malay
    "Chai", "Nong", "Dian", "Rizky", "Sari", "Nurul",
    # Eastern European / Slavic
    "Marta", "Ivan", "Anya", "Luka", "Vera", "Toma", "Mila", "Borya",
    "Katya", "Pavel", "Zoya", "Danila", "Nadya",
    # Scandinavian / Nordic
    "Elsa", "Erik", "Astrid", "Lars", "Sigrid", "Bjorn", "Inga", "Leif",
    # Mediterranean / Southern European
    "Marco", "Giulia", "Nikos", "Elena", "Mateus", "Ines", "Adria",
    # Indigenous / First Nations
    "Kaya", "Tala", "Ama", "Nuka", "Siku", "Pita",
]

# ---------------------------------------------------------------------------
# Scene types — everyday routines a young child would recognize
# ---------------------------------------------------------------------------

SCENE_TYPES: list[str] = [
    "morning routine",
    "breakfast time",
    "getting dressed",
    "going to school",
    "playground time",
    "lunchtime",
    "afternoon snack",
    "bath time",
    "bedtime routine",
    "grocery shopping",
    "walk in the park",
    "car ride",
    "cooking together",
    "cleaning up",
    "rainy day inside",
    "sunny day outside",
    "visit to the doctor",
    "feeding pets",
    "playing outside",
    "drawing and painting",
    "reading time",
    "garden time",
    "visiting grandparents",
    "birthday party",
    "picnic",
    "laundry day",
    "building with blocks",
    "baking cookies",
    "setting the table",
    "watching animals",
]

# ---------------------------------------------------------------------------
# Scene prompt templates
# ---------------------------------------------------------------------------

_SCENE_TEMPLATES: dict[str, str] = {
    "p2-scene-v1": """\
Write a short everyday scene for young children (ages 3-5).

Scene type: {scene_type}
The child in the scene is named {child_name}.
Include these facts naturally in the scene:
{fact_list}

Rules:
- Write 4-10 simple sentences describing this everyday moment
- Use simple words a young child would understand
- Describe what happens step by step using words like "first", "then", "next", "after"
- Do NOT include any dialogue or quoted speech
- Do NOT tell a story — just describe a normal everyday moment
- Do NOT include a moral, lesson, or character journey
- Do NOT start with "Once upon a time" or similar story openings
- Use present tense
- Do not use markdown formatting
- Write only the scene, nothing else\
""",
    "p2-scene-v2": """\
You describe small everyday moments for young children (age 3-5).

Rules:
- Write 4-10 short sentences in simple English
- Just describe what is happening — no story, no lesson, no dialogue
- Use present tense
- Vary how you connect sentences — do not always use "first, then, next, after"
- Sometimes just describe what is there, what someone notices, or what changes
- Start the scene in the middle of the moment, not at the beginning
- No markdown formatting
- Write only the scene, nothing else

The child's name is {child_name}.
Setting: {scene_type}
Weave in these facts:
{fact_list}\
""",
    "p2-scene-v3": """\
Describe a small everyday moment for young children (age 3-5).

The child's name is {child_name}.

These facts should appear naturally in the scene:
{fact_list}

If it fits, set the scene during: {scene_type}
If that setting does not match the facts above, pick a different everyday moment where they fit naturally.

Rules:
- Write 4-10 short sentences in simple English
- Just describe what is happening — no story, no lesson, no dialogue
- Use present tense
- Vary how you connect sentences
- Do not start with "It is" or "Today is"
- No markdown formatting
- Write only the scene, nothing else\
""",
}

DEFAULT_SCENE_VERSION = "p2-scene-v3"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_scene_template(version: str | None = None) -> tuple[str, str]:
    """Return (template_string, version) for a scene prompt.

    Raises:
        KeyError: If the version is unknown.
    """
    v = version or DEFAULT_SCENE_VERSION
    if v not in _SCENE_TEMPLATES:
        raise KeyError(
            f"Unknown scene prompt version: {v!r}. "
            f"Valid versions: {list(_SCENE_TEMPLATES)}"
        )
    return _SCENE_TEMPLATES[v], v


def build_scene_prompt(
    concept_pairs: list[tuple[str, str, str]],
    scene_type: str,
    child_name: str,
    version: str | None = None,
) -> str:
    """Build a scene prompt from a list of (noun, relation, value) triples.

    Args:
        concept_pairs: 2-4 (noun, relation, value) triples to exercise.
        scene_type: The everyday routine/scene type.
        child_name: Name to use for the child character in the scene.
        version: Prompt template version.

    Returns:
        The fully formatted prompt string.
    """
    template, _ = get_scene_template(version)
    fact_lines = []
    for noun, relation, value in concept_pairs:
        fact_sentence = relation_to_sentence(noun, relation, value)
        fact_lines.append(f"- {fact_sentence}")
    fact_list = "\n".join(fact_lines)
    return template.format(scene_type=scene_type, fact_list=fact_list, child_name=child_name)


def select_scene_type(rng: random.Random) -> str:
    """Pick a random scene type."""
    return rng.choice(SCENE_TYPES)


def select_child_name(rng: random.Random) -> str:
    """Pick a random child name from the diverse name pool."""
    return rng.choice(CHILD_NAMES)


def select_concept_pairs(
    candidates: list[tuple[str, str, str]],
    rng: random.Random,
    n_pairs: int = 3,
) -> list[tuple[str, str, str]]:
    """Select n_pairs concept triples for a scene, preferring diverse nouns.

    Picks from candidates, trying to use different nouns in each scene.
    """
    n_pairs = min(n_pairs, len(candidates))
    if n_pairs <= 0:
        return []

    # Group by noun for diversity.
    by_noun: dict[str, list[tuple[str, str, str]]] = {}
    for triple in candidates:
        by_noun.setdefault(triple[0], []).append(triple)

    selected: list[tuple[str, str, str]] = []
    used_nouns: set[str] = set()
    noun_list = list(by_noun.keys())
    rng.shuffle(noun_list)

    # First pass: one triple per noun.
    for noun in noun_list:
        if len(selected) >= n_pairs:
            break
        triple = rng.choice(by_noun[noun])
        selected.append(triple)
        used_nouns.add(noun)

    # If we still need more, allow repeats.
    if len(selected) < n_pairs:
        remaining = [t for t in candidates if t not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n_pairs - len(selected)])

    return selected
