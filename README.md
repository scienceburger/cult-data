# cult-data

Curriculum training data generation pipeline for a 100M parameter language model trained from scratch.

## Overview

This repo generates synthetic story data across curriculum phases. Each phase targets a specific linguistic capability. Phase 1 covers grammar and syntactic composition using age 3-5 vocabulary with no dialogue.

## Phase 1: Grammar & Syntax

Phase 1 generates ~200-word simple third-person stories with no dialogue. Stories are produced by a vLLM-served model and filtered using rule-based checks before saving.

### Vocabulary

Three word lists (~500 words each) live under `generation/vocab/`:

| File | Content |
|---|---|
| `phase1_nouns.txt` | Concrete nouns (animals, food, objects, places, roles, nature) |
| `phase1_verbs.txt` | Action and state verbs, including common inflected forms |
| `phase1_adjs.txt` | Descriptive adjectives (size, color, emotion, texture, etc.) |

Each story is generated with a randomly sampled noun + verb + adjective triplet and one of five narrative features (plot twist, moral, foreshadowing, bad ending, conflict).

### Running Phase 1 Generation

Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

Start a vLLM server (example):

```bash
vllm serve meta-llama/Llama-3-70b-Instruct --port 8000
```

Run the generator:

```bash
python -m generation.generate \
  --phase 1 \
  --model llama-70b \
  --api-base http://localhost:8000/v1 \
  --count 1000 \
  --batch-size 50 \
  --output-dir data/generation/phase1/ \
  --seed 42
```

All CLI options:

| Flag | Default | Description |
|---|---|---|
| `--phase` | `1` | Curriculum phase |
| `--model` | *(required)* | Model name as served by vLLM |
| `--api-base` | `http://localhost:8000/v1` | OpenAI-compatible API URL |
| `--api-key` | `EMPTY` | API key (`EMPTY` for local vLLM) |
| `--count` | `1000` | Target accepted story count |
| `--batch-size` | `50` | Concurrent API calls per batch |
| `--output-dir` | `data/generation/phase1` | Output directory |
| `--seed` | `42` | Random seed |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `512` | Max tokens per generation |
| `--log-level` | `INFO` | Logging verbosity |

## Output Format

Accepted stories are written to `{output-dir}/{run-id}.jsonl`, one JSON object per line:

```json
{
  "content": "Once there was a tiny rabbit who loved to climb...",
  "phase": 1,
  "source_model": "llama-70b",
  "word_triplet": {"noun": "rabbit", "verb": "climb", "adj": "tiny"},
  "narrative_feature": "plot_twist",
  "prompt_template_version": "p1-v1",
  "run_id": "gen-20250222-001",
  "timestamp": "2025-02-22T10:30:00Z",
  "filters_passed": true
}
```

Rejected stories go to `{output-dir}/rejected/{run-id}.jsonl` with the same schema plus a `rejection_reasons` list.

## Filters

All filters are rule-based — no LLM calls.

| Filter | Rejection condition |
|---|---|
| Word count | Outside 150–250 words |
| Triplet words | Any seed word absent (stem-matched) |
| No dialogue | Quotation marks or speech attribution patterns present |
| No first person | "I", "me", "my", "mine", "myself" appear as standalone words |
| No markers | Structural placeholders (`[`, `]`, `*`, `#`, `---`, `Title:`, etc.) |
| Near-duplicate | MinHash similarity > 0.8 against accepted stories |

## Repository Structure

```
cult-data/
├── generation/
│   ├── __init__.py
│   ├── generate.py           # Main generation script
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── phase1.py         # Prompt template + narrative features
│   ├── triplets.py           # Word triplet sampling
│   ├── filters.py            # Rule-based quality filters
│   └── vocab/
│       ├── __init__.py
│       ├── phase1_nouns.txt
│       ├── phase1_verbs.txt
│       └── phase1_adjs.txt
├── pyproject.toml
├── requirements.txt
└── README.md
```
