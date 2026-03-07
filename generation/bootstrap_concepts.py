"""Bootstrap script: generate initial concepts.json via LLM.

One-shot helper that calls the LLM to produce ~50 nouns with relation
annotations, then writes the result to generation/concepts.json.

Usage::

    python -m generation.bootstrap_concepts \\
        --model nvidia/Llama-3.3-70B-Instruct-FP4 \\
        --api-base http://localhost:8355/v1 \\
        --output generation/concepts.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

_BOOTSTRAP_PROMPT = """\
Generate a JSON object listing 50 nouns that a 3-year-old child would know, \
organized by category. Each noun should have relation annotations.

Categories and approximate counts:
- animals: 15 nouns
- food: 8 nouns
- body parts: 5 nouns
- vehicles: 5 nouns
- household objects: 5 nouns
- people: 4 nouns
- places: 4 nouns
- weather: 4 nouns

For each noun, provide relations from this set (use all that apply):
- IsA: what category it belongs to (e.g. ["animal", "pet"])
- Does: what it does (verb phrases, e.g. ["bark", "run", "fetch"])
- Has: what physical features it has (e.g. ["four legs", "a tail"])
- Likes: what it likes (e.g. ["bones", "walks"])
- AtLocation: where you find it (e.g. ["house", "park"])
- Sound: what sound it makes (e.g. ["woof"])
- Size: how big it is (e.g. ["small", "big"])
- Color: what colors it can be (e.g. ["brown", "white"])
- Taste: how it tastes (only for food, e.g. ["sweet"])
- Shape: what shape it is (only if notable, e.g. ["round"])

Output format (valid JSON only, no markdown, no extra text):
{
  "metadata": { "version": "0.1", "noun_count": 50 },
  "nouns": {
    "dog": {
      "category": "animals",
      "relations": {
        "IsA": ["animal", "pet"],
        "Does": ["bark", "run"],
        "Has": ["four legs", "a tail"],
        "Likes": ["bones"],
        "AtLocation": ["house", "park"],
        "Sound": ["woof"],
        "Size": ["medium"],
        "Color": ["brown", "white", "black"]
      }
    }
  }
}

Use simple words a young child would know. Include common, everyday nouns.
Output ONLY the JSON object, nothing else.\
"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Try to find JSON in code fences first.
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # Try parsing directly.
    return json.loads(text)


def bootstrap(
    model: str,
    api_base: str,
    api_key: str,
    output: Path,
) -> None:
    """Call the LLM and write concepts.json."""
    client = OpenAI(api_key=api_key, base_url=api_base)

    print(f"Calling {model} to generate concepts...", file=sys.stderr)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _BOOTSTRAP_PROMPT}],
        temperature=0.3,
        max_tokens=8192,
    )

    content = response.choices[0].message.content
    if not content:
        print("ERROR: Empty response from model", file=sys.stderr)
        sys.exit(1)

    try:
        concepts = _extract_json(content)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(content, file=sys.stderr)
        sys.exit(1)

    # Validate basic structure.
    if "nouns" not in concepts:
        print("ERROR: Response missing 'nouns' key", file=sys.stderr)
        sys.exit(1)

    noun_count = len(concepts["nouns"])
    concepts.setdefault("metadata", {})["noun_count"] = noun_count

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(concepts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {noun_count} nouns to {output}", file=sys.stderr)

    # Print category breakdown.
    categories: dict[str, int] = {}
    for info in concepts["nouns"].values():
        cat = info.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print("Category breakdown:", file=sys.stderr)
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generation.bootstrap_concepts",
        description="Generate initial concepts.json via LLM",
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8355/v1",
        help="API base URL (default: http://localhost:8355/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key (default: EMPTY)",
    )
    parser.add_argument(
        "--output",
        default="generation/concepts.json",
        help="Output path (default: generation/concepts.json)",
    )
    args = parser.parse_args()

    bootstrap(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
