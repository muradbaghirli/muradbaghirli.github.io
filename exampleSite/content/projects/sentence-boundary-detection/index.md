---
title: "Sentence Boundary Detection for Azerbaijani: 76ms → 3ms with Triton FIL"
date: 2026-01-31
description: "Character-level sentence boundary detector for Azerbaijani legal and administrative text. Random forest over 28 engineered features (0.958 F1), served on NVIDIA Triton Inference Server's FIL backend: 76ms → 3ms, a 25× speedup."
tags:
  - Python
  - scikit-learn
  - Random Forest
  - Treelite
  - NVIDIA Triton
  - FastAPI
  - Docker
  - Azerbaijani
  - NLP
showDate: true
showReadingTime: false
showAuthor: false
showPagination: true
showTableOfContents: true
---

*Status: **Complete** · Nov 2025 – Jan 2026 · [GitHub](https://github.com/muradbaghirli/SentenceBoundaryDetection)*

## TL;DR

I built a sentence boundary detector for Azerbaijani: a random forest classifier that decides, for every candidate terminator character (`.`, `!`, `?`, `;`, quotes, newlines…), whether a sentence actually ends there. Each candidate is described by a 19-character context window plus 9 engineered linguistic and punctuation features. The model reaches **0.958 F1** on boundary detection.

For serving, I compiled the forest with Treelite and deployed it on **NVIDIA Triton Inference Server (FIL backend)** behind a FastAPI service. Inference latency dropped from **76ms to 3ms, a 25× speedup**.

---

## Problem: Why Splitting on Periods Fails

Sentence splitting is the first step of almost every NLP pipeline (chunking for retrieval, summarization, translation), and errors here propagate everywhere downstream. Splitting on `.` breaks immediately on real Azerbaijani text, especially legal and administrative documents:

- **Abbreviations**: `prof.`, `məs.`, `mln.`, `kv.m.`, `Ph.D.`, initials like `Ə.M.`, Roman numerals like `XIV.`
- **Enumerations**: `1.`, `a)`, `(iii)`, bullet symbols, and lists introduced by phrases like *"aşağıdakılar müəyyən edilir:"*
- **Quotes**: a closing quote may or may not end the sentence, depending on balance and what follows
- **Semicolons and colons**: they end clauses inside lists, but sometimes end sentences

General-purpose splitters are tuned for English and miss Azerbaijani-specific abbreviations and morphology. Neural models would handle this, but they are overkill in cost and latency for a step that runs on every document.

---

## Approach: Character-Level Classification

The design follows the structure of the [CharBoundary](https://github.com/alea-institute/charboundary) library, adapted for Azerbaijani.

**1. Candidate detection.** Only characters that *could* end a sentence are classified. Everything else is skipped, so the model sees a small fraction of the text.

**2. Character encoding.** Each character is mapped to an integer by type: Azerbaijani letters (all 33, including `ə`, `ğ`, `ı`, `ş`, `ç`, `ö`, `ü`) get their own IDs; digits, terminators, whitespace, punctuation and other characters get category codes.

**3. Feature vector (28 dims)** for each candidate position:

| Features | Dims | Description |
|---|---|---|
| Context window | 19 | Encoded characters, 9 to the left + the candidate + 9 to the right |
| Abbreviation flag | 1 | Token ending at this period is in an 830+ entry Azerbaijani abbreviation list |
| Terminator type | 1 | +1 primary (`. ! ?` newline), −1 secondary (quotes, `;`, `:`), 0 otherwise |
| Quote balance | 1 | Are straight and curly quotes balanced up to this point? |
| Word completion | 1 | Does a closing quote look like it ends a word? |
| Followed by lowercase | 1 | Next character is lowercase, a strong "not a boundary" signal |
| In list item | 1 | Near a list marker (`1.`, `a)`, `(ii)`, `•`…) or list conjunction (`və`, `və ya`, `həmçinin`…) |
| Semicolon in list | 1 | `;` separating list items rather than sentences |
| Near colon | 1 | Proximity to a list-introducing colon |
| Next word uppercase | 1 | Next word starts with a capital letter |

**4. Classifier.** A random forest (128 trees, max depth 32) trained on labeled Azerbaijani text, with a 0.75 probability threshold that favors precision: a missed split is cheaper downstream than a wrong one.

**5. Preprocessing.** Training text is normalized first: Cyrillic and Greek look-alike characters are mapped to Latin (`а → a`, `ә → ə`), and chunks with characters outside the Azerbaijani alphabet are dropped.

---

## Serving: Triton FIL

The random forest in scikit-learn was the latency bottleneck at **76ms**. The fix was to stop running it in Python:

1. **Treelite compilation.** The trained forest is converted to a Treelite checkpoint.
2. **Triton FIL backend.** Triton's Forest Inference Library runs tree ensembles natively on NVIDIA GPUs (or CPU) and returns class probabilities.
3. **Dynamic batching.** Triton merges concurrent requests into batches of up to 1,024 candidates, with a 5ms max queue delay.
4. **FastAPI front end.** The API extracts features, calls Triton over gRPC, and splits the text at positive predictions.

Result: **76ms → 3ms (25×)**.

```
text ─► FastAPI ─► find candidates ─► 28-dim features ─► Triton (FIL, gRPC) ─► p(boundary) ─► split
```

The whole stack runs under Docker Compose: Triton plus the API, with health checks that separate liveness (`/health`) from readiness (`/ready`, meaning Triton is reachable and the model is loaded).

```bash
curl -X POST http://localhost:8000/segment \
  -H 'Content-Type: application/json' \
  -d '{"text": "Bu birinci cümlədir. Bu isə ikinci cümlədir."}'
```

---

## Results

| Metric | Value |
|---|---|
| Boundary F1 | **0.958** |
| Latency (scikit-learn) | 76ms |
| Latency (Triton FIL) | **3ms** |
| Speedup | **25×** |
| Feature vector | 28 dims |
| Model | Random forest, 128 trees, depth 32 |

---

## Stack

`Python` · `scikit-learn` · `NumPy` · `Treelite` · `NVIDIA Triton Inference Server (FIL)` · `gRPC` · `FastAPI` · `Docker Compose`

---

*Code on [GitHub](https://github.com/muradbaghirli/SentenceBoundaryDetection).*
