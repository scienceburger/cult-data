# Data Generation & Validation Plan v2

Replaces v1 (TinyStories-based triplet generation). Core changes: knowledge-first generation driven by a curated concept list and coverage matrix (not random word triplets), and everyday events instead of fictional stories to avoid repetitive LLM narrative tropes.

-----

## 1. Concept Structure

### Noun List

A curated list of ~300–500 nouns that a 3-year-old would reasonably know. Organized by category (animals, food, body parts, vehicles, household objects, people, places, weather, etc.).

This is the ground truth for what the model should learn. It is maintained as a flat file (CSV or JSON) and versioned alongside the data pipeline.

### Relation Types

Each noun is annotated with facts across a fixed set of relation types:

| Relation    | Example (dog)              | Description                          |
|-------------|----------------------------|--------------------------------------|
| IsA         | dog → animal               | Category membership                  |
| Does        | dog → bark, run, fetch     | Actions / behaviors                  |
| Has         | dog → tail, fur, paws      | Parts / physical attributes          |
| Likes       | dog → bones, walks, belly rubs | Preferences                      |
| AtLocation  | dog → park, house, yard    | Where it's typically found           |
| Sound       | dog → woof                 | Characteristic sound                 |
| Size        | dog → medium               | Relative size (small/medium/large)   |
| Color       | dog → brown, white, black  | Common colors (where applicable)     |

Not every noun needs every relation. Some relations are N/A for certain categories (weather nouns don't have Sound in the same way animals do). The schema is intentionally simple — a spreadsheet, not a graph.

**TBD:** Final noun list — start with 30–50 for pipeline validation, expand to 300–500
**TBD:** Whether to seed from ConceptNet (filtered) or curate from scratch
**TBD:** Whether relation types need expansion for non-object nouns (e.g. emotions, time)

### Coverage Matrix

A noun × relation matrix that tracks which (noun, relation, value) triples have been exercised in generated data. Each cell records:

- Whether a descriptor has been generated for it
- How many events/scenes exercise it
- Which models generated those samples

This is the primary tool for identifying gaps. Generation targets are set per-cell, not per-phase.

-----

## 2. Training Curriculum

Six phases progressing from concept grounding to conversation. Each phase feeds vocabulary and structure into the next. Phase transitions overlap formats to address cross-mode knowledge retrieval (CASCADE, 2025).

### Phase 1 — Descriptors

Dictionary/Wikipedia-style definitions and fact statements grounded in the noun list.

- One or more descriptors per (noun, relation) pair
- Mix of formats: "A dog is an animal." / "Dogs have four legs and a tail." / "You can find dogs at the park."
- Simple English, short sentences, no narrative structure
- Vocabulary restricted to the noun list + common function words + relation-specific verbs
- Purpose: anchor each concept with explicit factual statements

### Phase 2 — Everyday Events

Short descriptions of everyday scenes and routines that exercise specific concept-relation pairs from the coverage matrix. Not stories — no plot arc, no moral, no character journey. Just life happening.

- Each scene targets 2–4 concept-relation pairs (not random — selected to fill coverage gaps)
- Grounded in recognizable routines: morning, mealtime, bath, playground, bedtime, grocery store, walk, car ride, etc.
- Exercises temporal structure (first/then/after) and casual causality (hungry → eat → not hungry) without requiring narrative
- Simple grammar, ~100–200 words, no dialogue
- Avoids LLM story-writing tropes (moral lessons, conflict-resolution arcs, "once upon a time") by design — the prompt targets scenes, not stories
- Purpose: teach the model to use concepts in sequential, real-world context without triggering repetitive narrative patterns

### Phase 3 — Events with Dialogue

Everyday scenes from Phase 2, extended with character dialogue. Bridge phase between descriptive scenes and conversation.

- Scenes from Phase 2 patterns, with characters talking during the event ("Mom, can I have more cereal?" / "Look, the dog is barking!")
- Dialogue should reference concept-relation knowledge naturally within the scene
- Purpose: introduce conversational structure while maintaining concept grounding in everyday context

### Phase 4 — Advanced Descriptors

Expanded vocabulary, abstraction, more complex relational statements.

- Introduce compound concepts (e.g. "A veterinarian is a doctor for animals")
- Cross-concept relations (e.g. "Cats and dogs are both pets, but cats are smaller")
- Comparative and conditional structures
- Purpose: push beyond prototype-level knowledge toward compositional understanding

### Phase 5 — Dialogue-Heavy Events

Thin scene wrapper, mostly conversation.

- Characters discuss, explain, ask questions about concepts during everyday situations
- Multi-turn exchanges that require tracking context
- Purpose: bridge to pure conversation while retaining situational scaffolding

### Phase 6 — Pure Conversation

Dialogue only, no narrative frame.

- Question-answer, explanation, discussion formats
- Covers concept-relation knowledge in conversational form
- Purpose: final format target

### Phase Budget

- Phase 1 is sized by the concept structure: (number of nouns) × (average relations per noun) × (descriptors per relation). For 400 nouns × 6 relations × 3 descriptors = ~7,200 descriptors.
- Phase 2 is sized by coverage matrix targets: enough everyday scenes to exercise all concept-relation pairs multiple times in varied contexts and routines. Exact count determined empirically after Phase 1.
- Later phases sized based on earlier phase performance — no projections until we have empirical data.
- Total target remains ~2B tokens (Chinchilla-optimal for 100M parameters). The distribution across phases is TBD.

**TBD:** Exact token budget per phase
**TBD:** Overlap strategy between phases — how much blending at transitions

-----

## 3. Synthesis Methodology

### Approach

- **Descriptors:** Generated from (noun, relation, value) triples. Prompt includes the triple and target format (definition, fact statement, simple explanation). Diversity comes from varying sentence structure and phrasing, not from randomizing content.
- **Everyday events:** Generated from sets of 2–4 concept-relation pairs selected from the coverage matrix, anchored to a routine or scene type (morning, mealtime, playground, etc.). Priority given to under-covered pairs. Prompt includes the target pairs, scene type, and phase-appropriate constraints (length, vocabulary, dialogue presence). Prompts target scenes and sequences, not narratives — this avoids triggering LLM story-writing defaults.
- **Multi-model synthesis:** Llama 70B, Gemma 70B, Qwen 2.5 72B to reduce stylistic monoculture.
- **Per-sample metadata:** Each sample records its source concept-relation pairs, source model, generation timestamp, phase, and coverage matrix coordinates.

### Coverage-Driven Generation

Generation is not random. The pipeline:

1. Query the coverage matrix for under-covered (noun, relation) pairs
2. Select 2–4 pairs for the next scene (or 1 pair for a descriptor)
3. Generate using a phase-appropriate prompt template
4. Filter (rule-based, same as before)
5. On pass, update the coverage matrix

This ensures generation effort is directed at gaps, not duplicating already-covered territory.

### Curriculum Learning Notes

- Stanford CS224N finding: abrupt easy→full transition outperformed gradual ramp in early training
- CASCADE (2025): cross-mode knowledge fails without format overlap at phase transitions — addressed by phases 3 and 5 as bridge phases

**TBD:** Exact prompt templates per phase
**TBD:** How to select concept-relation pair combinations for scenes (random from under-covered? thematic clustering by routine? adjacent in category?)
**TBD:** Filtering and quality thresholds post-synthesis

-----

## 4. Phase Transition Diagnostics

Carried forward from v1. Cross-mode failures in bridge phases (3, 5) may originate from data quality or coverage issues in earlier phases.

### Per-Phase Regression Probes

Before advancing to phase N+1, test phase N concepts in the formats of future phases. For example, after Phase 1, probe descriptor concepts in everyday event format. If retrieval degrades, the problem is upstream.

### Diagnostic Backtracking Protocol

When a cross-mode failure appears in Phase 3+, re-probe the failing concepts in their original phase format first. If they fail there too, it's a data quality or coverage issue in the source phase.

-----

## 5. Dataset Validation

### Pre-Tokenization

- **Coverage matrix completeness** — all target (noun, relation) pairs have minimum descriptor and scene counts
- **Coverage distribution** — no noun or relation type is massively over/under-represented
- **Model diversity per cell** — each covered pair has samples from multiple source models
- Zipf distribution check — word frequency should follow power law
- POS distribution — verify coverage of nouns, verbs, adjectives, adverbs, prepositions, pronouns
- Sentence length distribution per phase — should increase progressively
- Vocabulary boundary enforcement — phase N should not contain phase N+2 vocabulary
- Duplicate and near-duplicate detection
- Dialogue structure validation (phases 3–6) — well-formed turns, balanced, non-degenerate
- Coherence scoring — scene/definition endings consistent with beginnings
- Cross-phase concept probe — sample 50–100 concepts from Phase 1, verify presence in Phase 2+ in new formats

### Post-Tokenization

- **Lexical saturation** — for each noun in the concept list, measure fertility (tokens per word) after tokenizer training. Fully saturated concepts should be 1 token. Concepts that fragment indicate under-coverage in the corpus.
- Fertility rate per phase — should increase progressively
- OOV byte rate
- Sequence length distribution — context window utilization
- Token frequency distribution — should remain roughly Zipfian
- Subword fragmentation check — core vocabulary words should not fragment unexpectedly

### Conceptual Dimensionality (Deferred)

Not implemented in v1 of the pipeline. The concept is: for each noun, how many already-known sub-concepts does the model need to compose to understand it? This requires a dependency structure between concepts (not just a flat list) and a way to probe the model's internal representations. Noted here for future work — it informs teaching order but is not required for initial generation.

**TBD:** Human review guidelines document — per-phase checklist for spot-checking
**TBD:** Automated scoring pipeline tooling
**TBD:** Thresholds for pass/fail per check

-----

## 6. Graduation Criteria — Phase 1

Phase 1 (Descriptors) graduates when all three gates pass. Gates are evaluated in order.

### Gate 1 — Data Validation

All pre-tokenization and post-tokenization checks from Section 5 must pass. Coverage matrix must meet minimum thresholds.

### Gate 2 — Tokenizer Validation

All five tokenizer validation metrics (fidelity, fertility, fragmentation, OOV byte rate, sequence length distribution) from the Tokenizer Specification must pass on test split. Lexical saturation check must pass for all nouns in the concept list. An interim tokenizer trained on Phase 1–2 data is acceptable.

### Gate 3 — Concept Retrieval Probe

- For each noun in the concept list, prompt the model with a partial descriptor and evaluate whether it can complete it correctly
- Example: "A dog is a ___" → expects "animal" or equivalent
- 300 test cases: 100 IsA prompts, 100 Does/Has prompts, 100 AtLocation/Likes prompts
- Prompts are synthetically constructed using only Phase 1 vocabulary, not sampled from training data
- LLM-as-judge evaluates each completion: PASS if factually correct and grammatically coherent, FAIL otherwise
- Judge model and version pinned and recorded per run

### Scoring

- Overall threshold: ≥ 90% (270/300 PASS)
- Per-category threshold: ≥ 90% (90/100 per category)
- Per-category thresholds prevent a strong relation type from masking a weak one

**TBD:** Gate 1 & 2 numeric pass/fail thresholds
**TBD:** Lexical saturation pass/fail threshold (e.g. all nouns in concept list must be ≤ 2 tokens)
**TBD:** Judge model selection (instruction-tuned, version-pinned)
**TBD:** Prompt construction templates per relation type
**TBD:** Graduation criteria for phases 2–6 (deferred until Phase 1 is complete)
