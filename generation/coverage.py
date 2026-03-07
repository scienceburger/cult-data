"""Coverage matrix for tracking descriptor generation across concept × relation cells.

Tracks how many descriptors have been generated for each (noun, relation, value)
triple, enabling targeted generation of under-covered cells.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONCEPTS_PATH = Path(__file__).parent / "concepts.json"


def load_concepts(path: Path | None = None) -> dict[str, Any]:
    """Load the concept structure from concepts.json.

    Returns:
        The full concepts dict with ``metadata`` and ``nouns`` keys.
    """
    p = path or _CONCEPTS_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def flatten_triples(concepts: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Expand concepts into a list of (noun, relation, value) triples.

    Returns:
        Sorted list of all triples from the concept structure.
    """
    triples: list[tuple[str, str, str]] = []
    for noun, info in concepts["nouns"].items():
        for relation, values in info.get("relations", {}).items():
            for value in values:
                triples.append((noun, relation, value))
    triples.sort()
    return triples


class CoverageMatrix:
    """Tracks per-cell descriptor counts for (noun, relation, value) triples."""

    def __init__(self, concepts: dict[str, Any]) -> None:
        self._concepts = concepts
        # {noun: {relation: {value: {"count": int, "models": [str]}}}}
        self._matrix: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        self._init_from_concepts()

    def _init_from_concepts(self) -> None:
        """Initialize empty cells from the concept structure."""
        for noun, info in self._concepts["nouns"].items():
            self._matrix[noun] = {}
            for relation, values in info.get("relations", {}).items():
                self._matrix[noun][relation] = {}
                for value in values:
                    self._matrix[noun][relation][value] = {
                        "count": 0,
                        "models": [],
                    }

    def record(self, noun: str, relation: str, value: str, model: str) -> None:
        """Record a successful descriptor generation for a cell."""
        cell = self._matrix.get(noun, {}).get(relation, {}).get(value)
        if cell is None:
            return
        cell["count"] += 1
        if model not in cell["models"]:
            cell["models"].append(model)

    def get_count(self, noun: str, relation: str, value: str) -> int:
        """Return the descriptor count for a specific cell."""
        cell = self._matrix.get(noun, {}).get(relation, {}).get(value)
        return cell["count"] if cell else 0

    def get_undercovered(self, min_count: int = 3) -> list[tuple[str, str, str]]:
        """Return (noun, relation, value) triples with fewer than min_count descriptors.

        Results are sorted by count ascending (least-covered first).
        """
        under: list[tuple[int, str, str, str]] = []
        for noun, relations in self._matrix.items():
            for relation, values in relations.items():
                for value, cell in values.items():
                    if cell["count"] < min_count:
                        under.append((cell["count"], noun, relation, value))
        under.sort()
        return [(noun, rel, val) for _, noun, rel, val in under]

    def select_next_batch(self, n: int, min_count: int = 3) -> list[tuple[str, str, str]]:
        """Pick n under-covered triples, prioritizing zero-coverage cells.

        Args:
            n: Number of triples to select.
            min_count: Minimum target count per cell.

        Returns:
            Up to n (noun, relation, value) triples needing more descriptors.
        """
        undercovered = self.get_undercovered(min_count=min_count)
        return undercovered[:n]

    def total_cells(self) -> int:
        """Total number of cells in the matrix."""
        count = 0
        for relations in self._matrix.values():
            for values in relations.values():
                count += len(values)
        return count

    def covered_cells(self, min_count: int = 1) -> int:
        """Number of cells with at least min_count descriptors."""
        count = 0
        for relations in self._matrix.values():
            for values in relations.values():
                for cell in values.values():
                    if cell["count"] >= min_count:
                        count += 1
        return count

    def save(self, path: Path) -> None:
        """Persist the coverage matrix to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_cells": self.total_cells(),
            "covered_cells": self.covered_cells(),
            "matrix": self._matrix,
        }
        path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        """Load a previously saved coverage matrix, merging counts."""
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        saved = data.get("matrix", {})
        for noun, relations in saved.items():
            if noun not in self._matrix:
                continue
            for relation, values in relations.items():
                if relation not in self._matrix[noun]:
                    continue
                for value, cell in values.items():
                    if value not in self._matrix[noun][relation]:
                        continue
                    self._matrix[noun][relation][value] = cell

    def summary(self) -> str:
        """Return a human-readable coverage summary."""
        total = self.total_cells()
        covered = self.covered_cells()
        pct = covered / total * 100 if total else 0
        return f"Coverage: {covered}/{total} cells ({pct:.0f}%)"
