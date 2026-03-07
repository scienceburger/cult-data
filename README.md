# cult-data

Curriculum training data generation pipeline for a 100M parameter language model trained from scratch.

## Overview

Knowledge-first synthetic data generation driven by a curated concept list and coverage matrix. Six curriculum phases progress from concept grounding to conversation. See [docs/plan-v2.md](docs/plan-v2.md) for the full plan.

### Curriculum Phases

| Phase | Name | Format | Status |
|-------|------|--------|--------|
| 1 | Descriptors | Factual sentences from (noun, relation, value) triples | Implemented |
| 2 | Everyday Events | Short scenes exercising 2–4 concept pairs, no dialogue | Planned |
| 3 | Events with Dialogue | Phase 2 scenes extended with character dialogue | Planned |
| 4 | Advanced Descriptors | Compound concepts, cross-concept relations, comparatives | Planned |
| 5 | Dialogue-Heavy Events | Thin scene wrapper, mostly conversation | Planned |
| 6 | Pure Conversation | Dialogue only, no narrative frame | Planned |

## Concept Structure

~200 nouns (expanding to 300–500) organized by category with relation annotations (IsA, Does, Has, Likes, AtLocation, Sound, Size, Color). Stored in `generation/concepts.json`.

Generation is coverage-driven: the pipeline queries the coverage matrix for under-covered (noun, relation, value) cells and targets those first.

## Phase 1: Descriptors

Generates factual sentences about concept × relation triples for young children (ages 3–5).

### Running

```bash
pip install -r requirements.txt

python -m generation.generate \
  --model nvidia/Llama-3.3-70B-Instruct-FP4 \
  --api-base http://localhost:8355/v1 \
  --count 200 \
  --batch-size 50 \
  --output-dir data/generation/phase1/ \
  --seed 42
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Model name as served by vLLM |
| `--api-base` | `http://localhost:8000/v1` | OpenAI-compatible API URL |
| `--api-key` | `EMPTY` | API key (`EMPTY` for local vLLM) |
| `--count` | `200` | Target accepted descriptor count |
| `--min-per-cell` | `3` | Minimum descriptors per coverage cell |
| `--batch-size` | `50` | Concurrent API calls |
| `--output-dir` | `data/generation/phase1` | Output directory |
| `--seed` | `42` | Random seed |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `256` | Max tokens per generation |
| `--prompt-version` | `p1-desc-v1` | Prompt template version |
| `--concepts` | `generation/concepts.json` | Path to concept structure |
| `--name` | *(optional)* | Experiment label |
| `--no-think` | `false` | Disable thinking mode (Qwen3, DeepSeek) |

### Output

Each run creates an isolated experiment directory:

```
data/generation/phase1/{run-id}/
    config.json     # all arguments + timestamps
    accepted.jsonl  # descriptors that passed all filters
    rejected.jsonl  # descriptors that failed, with rejection_reasons
    coverage.json   # coverage matrix snapshot
    summary.json    # final stats
```

Each descriptor record:

```json
{
  "content": "Dogs have four legs and a tail. They are covered in soft fur.",
  "phase": 1,
  "source_model": "nvidia/Llama-3.3-70B-Instruct-FP4",
  "noun": "dog",
  "relation": "Has",
  "value": "four legs",
  "prompt_template_version": "p1-desc-v1",
  "run_id": "20250306T...",
  "timestamp": "2025-03-06T10:30:00Z",
  "filters_passed": true
}
```

### Filters

All filters are rule-based — no LLM calls.

| Filter | Rejection condition |
|--------|---------------------|
| Word count | Outside 5–60 words |
| Concept mention | Target noun absent (stem-matched) |
| Factual grounding | Relation value absent (stem-matched) |
| No dialogue | Quotation marks or speech attribution present |
| No narrative | Story indicators (once upon, one day, there was, etc.) |
| No markers | Structural placeholders (`[`, `]`, `*`, `#`, etc.) |
| Near-duplicate | MinHash similarity > 0.8 against accepted descriptors |

## Repository Structure

```
cult-data/
├── docs/
│   └── plan-v2.md              # Full generation & validation plan
├── generation/
│   ├── __init__.py
│   ├── generate.py             # Main generation pipeline
│   ├── filters.py              # Rule-based quality filters
│   ├── coverage.py             # Coverage matrix tracking
│   ├── concepts.json           # Curated noun × relation structure
│   ├── bootstrap_concepts.py   # One-shot concept list generator
│   └── prompts/
│       ├── __init__.py
│       └── phase1.py           # Descriptor prompt templates
├── pyproject.toml
├── requirements.txt
└── README.md
```
