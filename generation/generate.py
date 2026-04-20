"""Generation pipeline for Phases 1 and 2.

Phase 1 (descriptors): factual sentences from concept × relation triples.
Phase 2 (everyday events): short scenes exercising 2-4 concept pairs.

Both phases use a vLLM-served model (OpenAI-compatible API), apply rule-based
filters, track coverage, and save results to isolated experiment directories.

Usage::

    # Phase 1
    python -m generation.generate --phase 1 \\
        --model llama-70b --count 200

    # Phase 2
    python -m generation.generate --phase 2 \\
        --model llama-70b --count 100

Each run creates::

    {output-dir}/{run-id}/
        config.json     # all arguments + timestamps
        accepted.jsonl  # samples that passed all filters
        rejected.jsonl  # samples that failed, with rejection_reasons
        coverage.json   # coverage matrix snapshot
        summary.json    # final stats written at completion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

from .coverage import CoverageMatrix, load_concepts
from .filters import FilterStats, add_to_minhash_index, create_minhash_index, run_descriptor_filters, run_scene_filters
from .prompts.phase1 import DEFAULT_DESCRIPTOR_VERSION, build_descriptor_prompt, get_descriptor_template
from .prompts.phase2 import DEFAULT_SCENE_VERSION, build_scene_prompt, get_scene_template, select_child_name, select_concept_pairs, select_scene_type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_MAX_TOKENS_PHASE2 = 512
_MAX_API_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_DEFAULT_PAIRS_PER_SCENE = 3
_MAX_RETRIES_PER_CELL = 4  # max repeats per undercovered cell per wave (avoids near-dup spam)

_SYSTEM_PROMPT_PHASE1 = (
    "You are a children's encyclopedia writer. "
    "Write only simple, factual sentences for young children ages 3-5. "
    "Never tell stories, use dialogue, or add markdown formatting."
)

_SYSTEM_PROMPT_PHASE2 = (
    "You are a children's encyclopedia writer. "
    "Describe simple everyday moments for young children ages 3-5. "
    "Never tell stories with characters, use dialogue, or add markdown formatting."
)


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
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_slug = _slugify(model)
    run_id = f"{ts}_{model_slug}_s{seed}"
    if name:
        run_id = f"{run_id}_{_slugify(name)}"
    return run_id


def _experiment_dir(base: Path, run_id: str) -> Path:
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
    model: str,
    api_base: str,
    count: int,
    batch_size: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    prompt_version: str,
    prompt_template: str,
    name: str | None,
    min_per_cell: int,
    phase: int = 1,
    pairs_per_scene: int | None = None,
) -> None:
    mode = "descriptors" if phase == 1 else "everyday_events"
    prompt_vars = ["noun", "fact_sentence"] if phase == 1 else ["scene_type", "fact_list"]
    config: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "phase": phase,
        "mode": mode,
        "model": model,
        "api_base": api_base,
        "count": count,
        "min_per_cell": min_per_cell,
        "batch_size": batch_size,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_template_version": prompt_version,
        "prompt_template": prompt_template,
        "prompt_variables": prompt_vars,
    }
    if phase == 2 and pairs_per_scene is not None:
        config["pairs_per_scene"] = pairs_per_scene
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
    total_tokens: int,
    coverage_summary: str,
) -> None:
    summary: dict[str, Any] = {
        "run_id": run_id,
        "completed_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_s": round(elapsed_s, 2),
        "throughput_descriptors_per_s": round(stats.total / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "throughput_tokens_per_s": round(total_tokens / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "accepted": stats.passed,
        "rejected": stats.rejected,
        "total_generated": stats.total,
        "total_tokens": total_tokens,
        "pass_rate": round(stats.pass_rate, 4),
        "coverage": coverage_summary,
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


def _descriptor_record(
    content: str,
    model: str,
    noun: str,
    relation: str,
    value: str,
    run_id: str,
    prompt_version: str,
    filters_passed: bool,
    rejection_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build the JSON record for one descriptor."""
    record: dict[str, Any] = {
        "content": content,
        "phase": 1,
        "source_model": model,
        "noun": noun,
        "relation": relation,
        "value": value,
        "prompt_template_version": prompt_version,
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
    extra_body: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> tuple[str | None, int]:
    """Call the API with exponential-backoff retry on failure.

    Returns ``(text, completion_tokens)``; text is ``None`` if all retries failed.
    """
    delay = _RETRY_BASE_DELAY
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(1, _MAX_API_RETRIES + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            text = content.strip() if content else None
            if response.usage and response.usage.completion_tokens:
                tokens = response.usage.completion_tokens
            else:
                tokens = len(text.split()) if text else 0
            return text, tokens
        except Exception as exc:  # noqa: BLE001
            if attempt == _MAX_API_RETRIES:
                logger.warning("API call failed after %d retries: %s", _MAX_API_RETRIES, exc)
                return None, 0
            logger.debug("API call attempt %d failed (%s); retrying in %.1fs", attempt, exc, delay)
            await asyncio.sleep(delay)
            delay *= 2
    return None, 0


async def _generate_one(
    client: AsyncOpenAI,
    model: str,
    noun: str,
    relation: str,
    value: str,
    run_id: str,
    temperature: float,
    max_tokens: int,
    minhash_index: Any | None,
    descriptor_index: int,
    prompt_version: str,
    extra_body: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """Generate and filter one descriptor.

    Returns:
        ``(accepted_record, rejected_record, completion_tokens)``
    """
    prompt = build_descriptor_prompt(noun, relation, value, version=prompt_version, rng=rng)

    content, tokens = await _call_api_with_retry(
        client, model, prompt, temperature, max_tokens,
        extra_body=extra_body, system_prompt=_SYSTEM_PROMPT_PHASE1,
    )

    if content is None:
        rejected = _descriptor_record(
            content="",
            model=model,
            noun=noun,
            relation=relation,
            value=value,
            run_id=run_id,
            prompt_version=prompt_version,
            filters_passed=False,
            rejection_reasons=["api_failure"],
        )
        return None, rejected, tokens

    passed, reasons, cleaned = run_descriptor_filters(content, noun, relation, value, minhash_index)

    if passed:
        if minhash_index is not None:
            add_to_minhash_index(cleaned, f"{run_id}-{descriptor_index}", minhash_index)
        accepted = _descriptor_record(
            content=cleaned,
            model=model,
            noun=noun,
            relation=relation,
            value=value,
            run_id=run_id,
            prompt_version=prompt_version,
            filters_passed=True,
        )
        return accepted, None, tokens
    else:
        rejected = _descriptor_record(
            content=content,
            model=model,
            noun=noun,
            relation=relation,
            value=value,
            run_id=run_id,
            prompt_version=prompt_version,
            filters_passed=False,
            rejection_reasons=reasons,
        )
        return None, rejected, tokens


def _scene_record(
    content: str,
    model: str,
    concept_pairs: list[tuple[str, str, str]],
    scene_type: str,
    child_name: str,
    run_id: str,
    prompt_version: str,
    filters_passed: bool,
    rejection_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build the JSON record for one everyday scene."""
    record: dict[str, Any] = {
        "content": content,
        "phase": 2,
        "source_model": model,
        "concept_pairs": [
            {"noun": n, "relation": r, "value": v} for n, r, v in concept_pairs
        ],
        "scene_type": scene_type,
        "child_name": child_name,
        "prompt_template_version": prompt_version,
        "run_id": run_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "filters_passed": filters_passed,
    }
    if rejection_reasons is not None:
        record["rejection_reasons"] = rejection_reasons
    return record


async def _generate_one_scene(
    client: AsyncOpenAI,
    model: str,
    concept_pairs: list[tuple[str, str, str]],
    scene_type: str,
    child_name: str,
    run_id: str,
    temperature: float,
    max_tokens: int,
    minhash_index: Any | None,
    scene_index: int,
    prompt_version: str,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """Generate and filter one everyday scene.

    Returns:
        ``(accepted_record, rejected_record, completion_tokens)``
    """
    prompt = build_scene_prompt(concept_pairs, scene_type, child_name=child_name, version=prompt_version)

    content, tokens = await _call_api_with_retry(
        client, model, prompt, temperature, max_tokens,
        extra_body=extra_body, system_prompt=_SYSTEM_PROMPT_PHASE2,
    )

    if content is None:
        rejected = _scene_record(
            content="",
            model=model,
            concept_pairs=concept_pairs,
            scene_type=scene_type,
            child_name=child_name,
            run_id=run_id,
            prompt_version=prompt_version,
            filters_passed=False,
            rejection_reasons=["api_failure"],
        )
        return None, rejected, tokens

    passed, reasons, cleaned = run_scene_filters(content, concept_pairs, minhash_index)

    if passed:
        if minhash_index is not None:
            add_to_minhash_index(cleaned, f"{run_id}-scene-{scene_index}", minhash_index)
        accepted = _scene_record(
            content=cleaned,
            model=model,
            concept_pairs=concept_pairs,
            scene_type=scene_type,
            child_name=child_name,
            run_id=run_id,
            prompt_version=prompt_version,
            filters_passed=True,
        )
        return accepted, None, tokens
    else:
        rejected = _scene_record(
            content=content,
            model=model,
            concept_pairs=concept_pairs,
            scene_type=scene_type,
            child_name=child_name,
            run_id=run_id,
            prompt_version=prompt_version,
            filters_passed=False,
            rejection_reasons=reasons,
        )
        return None, rejected, tokens


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


async def run_generation(
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
    prompt_version: str | None = None,
    min_per_cell: int = 3,
    concepts_path: Path | None = None,
    no_think: bool = False,
    phase: int = 1,
    pairs_per_scene: int = _DEFAULT_PAIRS_PER_SCENE,
) -> None:
    """Run the generation pipeline for Phase 1 or Phase 2.

    Args:
        model: Model name as served by vLLM.
        api_base: Base URL of the OpenAI-compatible API endpoint.
        api_key: API key (use "EMPTY" for local vLLM servers).
        count: Target number of *accepted* samples to generate.
        batch_size: Number of concurrent API calls per batch.
        output_dir: Root directory; each run gets its own subdirectory.
        seed: Random seed for reproducibility.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per generation.
        name: Optional experiment label.
        prompt_version: Prompt template version.
        min_per_cell: Minimum samples per coverage cell (used for batch selection).
        concepts_path: Path to concepts.json (defaults to generation/concepts.json).
        no_think: Disable thinking/reasoning mode (for Qwen3, DeepSeek, etc.).
        phase: Curriculum phase (1=descriptors, 2=everyday events).
        pairs_per_scene: Number of concept pairs per scene (phase 2 only).
    """
    rng = random.Random(seed)
    started_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    run_id = _make_run_id(model, seed, name)

    if phase == 1:
        prompt_template, prompt_ver = get_descriptor_template(prompt_version)
    else:
        prompt_template, prompt_ver = get_scene_template(prompt_version)

    # Load concepts and initialize coverage.
    concepts = load_concepts(concepts_path)
    coverage = CoverageMatrix(concepts)

    # Load prior coverage and seed the minhash dedup index from previous runs.
    minhash_index = create_minhash_index()
    prior_accepted = 0
    prior_dir = Path(output_dir)
    if prior_dir.is_dir():
        for prev_run in sorted(prior_dir.iterdir()):
            if not prev_run.is_dir():
                continue
            # Load coverage snapshots.
            prev_cov = prev_run / "coverage.json"
            if prev_cov.exists():
                coverage.load(prev_cov)
            # Seed minhash from previously accepted content for cross-run dedup.
            prev_accepted = prev_run / "accepted.jsonl"
            if prev_accepted.exists() and minhash_index is not None:
                for line in prev_accepted.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        add_to_minhash_index(
                            rec["content"],
                            f"prior-{prior_accepted}",
                            minhash_index,
                        )
                        prior_accepted += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        if prior_accepted > 0:
            logger.info(
                "Loaded prior state: %d accepted descriptors in minhash, coverage=%s",
                prior_accepted, coverage.summary(),
            )

    stats = FilterStats()

    exp_dir = _experiment_dir(Path(output_dir), run_id)
    accepted_path = exp_dir / "accepted.jsonl"
    rejected_path = exp_dir / "rejected.jsonl"
    coverage_path = exp_dir / "coverage.json"

    _write_config(
        exp_dir=exp_dir,
        run_id=run_id,
        started_at=started_at,
        model=model,
        api_base=api_base,
        count=count,
        batch_size=batch_size,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_version=prompt_ver,
        prompt_template=prompt_template,
        name=name,
        min_per_cell=min_per_cell,
        phase=phase,
        pairs_per_scene=pairs_per_scene if phase == 2 else None,
    )
    logger.info("Experiment dir: %s", exp_dir)
    unit = "descriptors" if phase == 1 else "scenes"
    logger.info("Starting run %s | phase=%d | target=%d %s | model=%s", run_id, phase, count, unit, model)

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    extra_body: dict[str, Any] | None = None
    if no_think:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        logger.info("Thinking mode disabled via extra_body")

    accepted_count = 0
    total_generated = 0
    total_tokens = 0
    start_time = time.monotonic()

    sem = asyncio.Semaphore(batch_size)

    # --- Phase 1: Descriptor generation ---
    if phase == 1:
        async def _bounded_generate(
            noun: str, relation: str, value: str, idx: int,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
            async with sem:
                return await _generate_one(
                    client=client,
                    model=model,
                    noun=noun,
                    relation=relation,
                    value=value,
                    run_id=run_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    minhash_index=minhash_index,
                    descriptor_index=idx,
                    prompt_version=prompt_ver,
                    extra_body=extra_body,
                    rng=rng,
                )

        def _process_result(
            result: tuple[dict[str, Any] | None, dict[str, Any] | None, int],
            acc_f: Any,
            rej_f: Any,
            bar: Any,
        ) -> None:
            nonlocal accepted_count, total_generated, total_tokens
            accepted, rejected, tokens = result
            total_generated += 1
            total_tokens += tokens
            if accepted is not None:
                acc_f.write(json.dumps(accepted, ensure_ascii=False) + "\n")
                accepted_count += 1
                stats.record(True, [])
                coverage.record(accepted["noun"], accepted["relation"], accepted["value"], model)
                bar.update(1)
            else:
                assert rejected is not None
                rej_f.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                stats.record(False, rejected.get("rejection_reasons", []))

            if total_generated % batch_size == 0:
                elapsed = time.monotonic() - start_time
                bar.set_postfix(
                    depth=min_per_cell,
                    desc_s=f"{total_generated / elapsed:.1f}" if elapsed > 0 else "0.0",
                    tok_s=f"{total_tokens / elapsed:.0f}" if elapsed > 0 else "0",
                    pass_rate=f"{stats.pass_rate:.1%}",
                    rejected=stats.rejected,
                )

        with (
            accepted_path.open("w", encoding="utf-8") as acc_f,
            rejected_path.open("w", encoding="utf-8") as rej_f,
            tqdm(total=count, unit="desc", desc=run_id, dynamic_ncols=True) as bar,
        ):
            global_idx = 0
            while accepted_count < count:
                remaining = count - accepted_count
                wave_size = min(remaining * 3, batch_size * 4)

                batch_triples = coverage.select_next_batch(wave_size, min_count=min_per_cell)
                while not batch_triples:
                    min_per_cell += 1
                    logger.info("All cells covered; escalating min_per_cell to %d", min_per_cell)
                    batch_triples = coverage.select_next_batch(wave_size, min_count=min_per_cell)

                # Pad if undercovered pool is smaller than the wave.
                if len(batch_triples) < wave_size:
                    # Cap repeats per cell to avoid near-duplicate spam for hard cells.
                    effective_wave = min(wave_size, len(batch_triples) * _MAX_RETRIES_PER_CELL)
                    extended: list[tuple[str, str, str]] = []
                    while len(extended) < effective_wave:
                        rng.shuffle(batch_triples)
                        extended.extend(batch_triples)
                    batch_triples = extended[:effective_wave]

                tasks = []
                for noun, relation, value in batch_triples:
                    tasks.append(_bounded_generate(noun, relation, value, global_idx))
                    global_idx += 1

                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    _process_result(result, acc_f, rej_f, bar)
                    if accepted_count >= count:
                        break

    # --- Phase 2: Scene generation ---
    else:
        async def _bounded_generate_scene(
            pairs: list[tuple[str, str, str]], scene_type: str, child_name: str, idx: int,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
            async with sem:
                return await _generate_one_scene(
                    client=client,
                    model=model,
                    concept_pairs=pairs,
                    scene_type=scene_type,
                    child_name=child_name,
                    run_id=run_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    minhash_index=minhash_index,
                    scene_index=idx,
                    prompt_version=prompt_ver,
                    extra_body=extra_body,
                )

        def _process_scene_result(
            result: tuple[dict[str, Any] | None, dict[str, Any] | None, int],
            acc_f: Any,
            rej_f: Any,
            bar: Any,
        ) -> None:
            nonlocal accepted_count, total_generated, total_tokens
            accepted, rejected, tokens = result
            total_generated += 1
            total_tokens += tokens
            if accepted is not None:
                acc_f.write(json.dumps(accepted, ensure_ascii=False) + "\n")
                accepted_count += 1
                stats.record(True, [])
                for pair in accepted["concept_pairs"]:
                    coverage.record(pair["noun"], pair["relation"], pair["value"], model)
                bar.update(1)
            else:
                assert rejected is not None
                rej_f.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                stats.record(False, rejected.get("rejection_reasons", []))

            if total_generated % batch_size == 0:
                elapsed = time.monotonic() - start_time
                bar.set_postfix(
                    depth=min_per_cell,
                    scene_s=f"{total_generated / elapsed:.1f}" if elapsed > 0 else "0.0",
                    tok_s=f"{total_tokens / elapsed:.0f}" if elapsed > 0 else "0",
                    pass_rate=f"{stats.pass_rate:.1%}",
                    rejected=stats.rejected,
                )

        with (
            accepted_path.open("w", encoding="utf-8") as acc_f,
            rejected_path.open("w", encoding="utf-8") as rej_f,
            tqdm(total=count, unit="scene", desc=run_id, dynamic_ncols=True) as bar,
        ):
            global_scene_idx = 0
            candidate_window = pairs_per_scene * 4

            while accepted_count < count:
                remaining = count - accepted_count
                wave_scenes = min(remaining * 3, batch_size * 4)

                all_undercovered = coverage.select_next_batch(
                    wave_scenes * pairs_per_scene, min_count=min_per_cell,
                )
                while not all_undercovered:
                    min_per_cell += 1
                    logger.info("All cells covered; escalating min_per_cell to %d", min_per_cell)
                    all_undercovered = coverage.select_next_batch(
                        wave_scenes * pairs_per_scene, min_count=min_per_cell,
                    )

                if len(all_undercovered) < wave_scenes * pairs_per_scene:
                    # Cap repeats per cell to avoid near-duplicate spam for hard cells.
                    effective_total = min(
                        wave_scenes * pairs_per_scene,
                        len(all_undercovered) * _MAX_RETRIES_PER_CELL,
                    )
                    # Align to pairs_per_scene so every scene gets a full set of pairs.
                    wave_scenes = max(1, effective_total // pairs_per_scene)
                    effective_total = wave_scenes * pairs_per_scene
                    extended: list[tuple[str, str, str]] = []
                    while len(extended) < effective_total:
                        rng.shuffle(all_undercovered)
                        extended.extend(all_undercovered)
                    all_undercovered = extended[:effective_total]

                tasks = []
                for i in range(wave_scenes):
                    start_idx = i * pairs_per_scene
                    candidates = all_undercovered[start_idx : start_idx + candidate_window]
                    if len(candidates) < pairs_per_scene:
                        candidates = all_undercovered[start_idx : start_idx + pairs_per_scene]
                    pairs = select_concept_pairs(candidates, rng, n_pairs=pairs_per_scene)
                    scene_type = select_scene_type(rng)
                    child_name = select_child_name(rng)
                    tasks.append(_bounded_generate_scene(pairs, scene_type, child_name, global_scene_idx))
                    global_scene_idx += 1

                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    _process_scene_result(result, acc_f, rej_f, bar)
                    if accepted_count >= count:
                        break

    elapsed = time.monotonic() - start_time

    # Save coverage snapshot.
    coverage.save(coverage_path)

    _write_summary(exp_dir, run_id, stats, elapsed, total_tokens, coverage.summary())

    sample_label = "descriptors" if phase == 1 else "scenes"
    logger.info("Run %s complete in %.1fs", run_id, elapsed)
    print(stats.summary())
    print(f"\n{coverage.summary()}")
    print(f"\nExperiment: {exp_dir}")
    print(f"  accepted.jsonl  ({stats.passed} {sample_label})")
    print(f"  rejected.jsonl  ({stats.rejected} {sample_label})")
    print(f"  coverage.json")
    print(f"  config.json")
    print(f"  summary.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generation.generate",
        description=(
            "Generation pipeline (Phase 1: descriptors, Phase 2: everyday events). "
            "Each run creates an isolated experiment directory."
        ),
    )
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Curriculum phase (1=descriptors, 2=everyday events; default: 1)")
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
        default=200,
        help="Target number of accepted samples (default: 200)",
    )
    parser.add_argument(
        "--min-per-cell",
        type=int,
        default=3,
        help="Minimum descriptors per coverage cell (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Concurrent API calls per batch (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory (default: ~/data/generation/phase{N})",
    )
    parser.add_argument(
        "--pairs-per-scene",
        type=int,
        default=_DEFAULT_PAIRS_PER_SCENE,
        help=f"Concept pairs per scene, phase 2 only (default: {_DEFAULT_PAIRS_PER_SCENE})",
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
        "--prompt-version",
        default=None,
        help="Prompt template version (default: per-phase, p1-desc-v1 / p2-scene-v3)",
    )
    parser.add_argument(
        "--concepts",
        default=None,
        help="Path to concepts.json (default: generation/concepts.json)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional experiment label appended to the run ID",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking/reasoning mode (for Qwen3, DeepSeek, etc.)",
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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        concepts_path = Path(args.concepts) if args.concepts else None
        _concepts_meta = load_concepts(concepts_path)
        _version = _concepts_meta.get("metadata", {}).get("version", "unknown")
        output_dir = os.path.expanduser(f"~/data/generation/v{_version}/phase{args.phase}")
    max_tokens = args.max_tokens
    # Default to higher token limit for phase 2 scenes.
    if args.phase == 2 and args.max_tokens == _DEFAULT_MAX_TOKENS:
        max_tokens = _DEFAULT_MAX_TOKENS_PHASE2

    asyncio.run(
        run_generation(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            count=args.count,
            batch_size=args.batch_size,
            output_dir=Path(output_dir),
            seed=args.seed,
            temperature=args.temperature,
            max_tokens=max_tokens,
            name=args.name,
            prompt_version=args.prompt_version,
            min_per_cell=args.min_per_cell,
            concepts_path=Path(args.concepts) if args.concepts else None,
            no_think=args.no_think,
            phase=args.phase,
            pairs_per_scene=args.pairs_per_scene,
        )
    )


if __name__ == "__main__":
    main()
