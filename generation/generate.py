"""Phase 1 story generation script.

Generates simple stories via a vLLM-served model (OpenAI-compatible API),
applies rule-based filters, and saves results to an isolated experiment
directory with full configuration and summary metadata.

Usage::

    python -m generation.generate \\
        --phase 1 \\
        --model llama-70b \\
        --api-base http://localhost:8000/v1 \\
        --count 1000 \\
        --batch-size 50 \\
        --output-dir data/generation/phase1/ \\
        --seed 42 \\
        --name baseline

Each run creates::

    {output-dir}/{run-id}/
        config.json     # all arguments + timestamps
        accepted.jsonl  # stories that passed all filters
        rejected.jsonl  # stories that failed, with rejection_reasons
        summary.json    # final stats written at completion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .filters import FilterStats, add_to_minhash_index, create_minhash_index, run_all_filters
from .prompts.phase1 import PROMPT_TEMPLATE_VERSION, build_prompt, random_feature
from .triplets import load_vocab, sample_triplet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 512
_MAX_API_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


# ---------------------------------------------------------------------------
# Run ID and experiment directory
# ---------------------------------------------------------------------------


def _slugify(value: str) -> str:
    """Convert an arbitrary string to a filesystem-safe slug."""
    value = value.strip()
    value = re.sub(r"[/\\: ]+", "-", value)
    value = re.sub(r"[^\w\-]", "", value)
    return value.strip("-")[:40]


def _make_run_id(model: str, seed: int, name: str | None) -> str:
    """Generate a descriptive, unique run identifier.

    Format: ``{timestamp}_{model-slug}_s{seed}[_{name}]``
    Example: ``20250222T103045_llama-70b_s42_baseline``
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_slug = _slugify(model)
    run_id = f"{ts}_{model_slug}_s{seed}"
    if name:
        run_id = f"{run_id}_{_slugify(name)}"
    return run_id


def _experiment_dir(base: Path, run_id: str) -> Path:
    """Return (and create) the isolated directory for this run."""
    d = base / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Config / summary helpers
# ---------------------------------------------------------------------------


def _write_config(
    exp_dir: Path,
    run_id: str,
    started_at: str,
    phase: int,
    model: str,
    api_base: str,
    count: int,
    batch_size: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    name: str | None,
) -> None:
    config: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "phase": phase,
        "model": model,
        "api_base": api_base,
        "count": count,
        "batch_size": batch_size,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }
    if name:
        config["name"] = name
    (exp_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_summary(
    exp_dir: Path,
    run_id: str,
    stats: FilterStats,
    elapsed_s: float,
) -> None:
    summary: dict[str, Any] = {
        "run_id": run_id,
        "completed_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_s": round(elapsed_s, 2),
        "accepted": stats.passed,
        "rejected": stats.rejected,
        "total_generated": stats.total,
        "pass_rate": round(stats.pass_rate, 4),
        "filter_breakdown": dict(
            sorted(stats.failures.items(), key=lambda kv: -kv[1])
        ),
    }
    (exp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _story_record(
    content: str,
    phase: int,
    model: str,
    triplet: dict[str, str],
    feature: str,
    run_id: str,
    filters_passed: bool,
    rejection_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build the JSON record for one story."""
    record: dict[str, Any] = {
        "content": content,
        "phase": phase,
        "source_model": model,
        "word_triplet": triplet,
        "narrative_feature": feature,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "run_id": run_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "filters_passed": filters_passed,
    }
    if rejection_reasons is not None:
        record["rejection_reasons"] = rejection_reasons
    return record


async def _call_api_with_retry(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str | None:
    """Call the API with exponential-backoff retry on failure.

    Returns the generated text, or ``None`` if all retries failed.
    """
    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _MAX_API_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as exc:  # noqa: BLE001
            if attempt == _MAX_API_RETRIES:
                logger.warning("API call failed after %d retries: %s", _MAX_API_RETRIES, exc)
                return None
            logger.debug("API call attempt %d failed (%s); retrying in %.1fs", attempt, exc, delay)
            await asyncio.sleep(delay)
            delay *= 2
    return None


async def _generate_one(
    client: AsyncOpenAI,
    model: str,
    phase: int,
    vocab: dict[str, list[str]],
    run_id: str,
    temperature: float,
    max_tokens: int,
    minhash_index: Any | None,
    rng: random.Random,
    story_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Generate and filter one story.

    Returns:
        ``(accepted_record, rejected_record)`` — exactly one will be non-None.
    """
    triplet = sample_triplet(vocab=vocab, phase=phase, rng=rng)
    feature = random_feature(rng=rng)
    prompt = build_prompt(triplet, feature)

    content = await _call_api_with_retry(client, model, prompt, temperature, max_tokens)

    if content is None:
        rejected = _story_record(
            content="",
            phase=phase,
            model=model,
            triplet=triplet,
            feature=feature,
            run_id=run_id,
            filters_passed=False,
            rejection_reasons=["api_failure"],
        )
        return None, rejected

    passed, reasons = run_all_filters(content, triplet, minhash_index)

    if passed:
        if minhash_index is not None:
            add_to_minhash_index(content, f"{run_id}-{story_index}", minhash_index)
        accepted = _story_record(
            content=content,
            phase=phase,
            model=model,
            triplet=triplet,
            feature=feature,
            run_id=run_id,
            filters_passed=True,
        )
        return accepted, None
    else:
        rejected = _story_record(
            content=content,
            phase=phase,
            model=model,
            triplet=triplet,
            feature=feature,
            run_id=run_id,
            filters_passed=False,
            rejection_reasons=reasons,
        )
        return None, rejected


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


async def run_generation(
    phase: int,
    model: str,
    api_base: str,
    api_key: str,
    count: int,
    batch_size: int,
    output_dir: Path,
    seed: int,
    temperature: float,
    max_tokens: int,
    name: str | None = None,
) -> None:
    """Run the full generation pipeline.

    Each invocation creates an isolated experiment directory under
    ``output_dir`` containing ``config.json``, ``accepted.jsonl``,
    ``rejected.jsonl``, and ``summary.json``.

    Args:
        phase: Curriculum phase number.
        model: Model name as served by vLLM.
        api_base: Base URL of the OpenAI-compatible API endpoint.
        api_key: API key (use "EMPTY" for local vLLM servers).
        count: Target number of *accepted* stories to generate.
        batch_size: Number of concurrent API calls per batch.
        output_dir: Root directory; each run gets its own subdirectory.
        seed: Random seed for reproducibility.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per generation.
        name: Optional human-readable experiment label included in the run ID
            and config (e.g. "baseline", "temp-09", "llama-vs-mistral").
    """
    rng = random.Random(seed)
    started_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    run_id = _make_run_id(model, seed, name)

    vocab = load_vocab(phase)
    minhash_index = create_minhash_index()
    stats = FilterStats()

    # One subdirectory per run — fully isolated.
    exp_dir = _experiment_dir(Path(output_dir), run_id)
    accepted_path = exp_dir / "accepted.jsonl"
    rejected_path = exp_dir / "rejected.jsonl"

    _write_config(
        exp_dir=exp_dir,
        run_id=run_id,
        started_at=started_at,
        phase=phase,
        model=model,
        api_base=api_base,
        count=count,
        batch_size=batch_size,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        name=name,
    )
    logger.info("Experiment dir: %s", exp_dir)
    logger.info("Starting run %s | target=%d | model=%s | phase=%d", run_id, count, model, phase)

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    accepted_count = 0
    total_generated = 0
    story_index = 0
    start_time = time.monotonic()

    with accepted_path.open("w", encoding="utf-8") as acc_f, rejected_path.open(
        "w", encoding="utf-8"
    ) as rej_f:

        while accepted_count < count:
            remaining = count - accepted_count
            # Oversample by 3× to absorb expected rejections without extra round-trips.
            current_batch = min(batch_size, remaining * 3)

            tasks = [
                _generate_one(
                    client=client,
                    model=model,
                    phase=phase,
                    vocab=vocab,
                    run_id=run_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    minhash_index=minhash_index,
                    rng=rng,
                    story_index=story_index + i,
                )
                for i in range(current_batch)
            ]
            story_index += current_batch

            results = await asyncio.gather(*tasks)

            for accepted, rejected in results:
                total_generated += 1
                if accepted is not None:
                    acc_f.write(json.dumps(accepted, ensure_ascii=False) + "\n")
                    accepted_count += 1
                    stats.record(True, [])
                else:
                    assert rejected is not None
                    rej_f.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                    stats.record(False, rejected.get("rejection_reasons", []))

            elapsed = time.monotonic() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0.0
            print(
                f"\r[{run_id}] accepted={accepted_count}/{count} "
                f"rejected={stats.rejected} total={total_generated} "
                f"pass_rate={stats.pass_rate:.1%} rate={rate:.1f}/s",
                end="",
                flush=True,
            )

            if accepted_count >= count:
                break

    print()  # Newline after progress line.
    elapsed = time.monotonic() - start_time
    _write_summary(exp_dir, run_id, stats, elapsed)

    logger.info("Run %s complete in %.1fs", run_id, elapsed)
    print(stats.summary())
    print(f"\nExperiment: {exp_dir}")
    print(f"  accepted.jsonl  ({stats.passed} stories)")
    print(f"  rejected.jsonl  ({stats.rejected} stories)")
    print(f"  config.json")
    print(f"  summary.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generation.generate",
        description=(
            "Phase 1 story generation pipeline. "
            "Each run creates an isolated experiment directory."
        ),
    )
    parser.add_argument("--phase", type=int, default=1, help="Curriculum phase (default: 1)")
    parser.add_argument("--model", required=True, help="Model name as served by vLLM")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key (use 'EMPTY' for local vLLM; default: EMPTY)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Target number of accepted stories (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Concurrent API calls per batch (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/generation/phase1",
        help="Root output directory; each run gets its own subdirectory (default: data/generation/phase1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=_DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {_DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULT_MAX_TOKENS,
        help=f"Max tokens per generation (default: {_DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional experiment label appended to the run ID "
            "(e.g. 'baseline', 'temp-09'). Useful for comparing runs."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    asyncio.run(
        run_generation(
            phase=args.phase,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            count=args.count,
            batch_size=args.batch_size,
            output_dir=Path(args.output_dir),
            seed=args.seed,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            name=args.name,
        )
    )


if __name__ == "__main__":
    main()
