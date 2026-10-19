---
title: "About Me"
date: 2025-01-01
showDate: false
showReadingTime: false
showAuthor: false
showPagination: false
showTableOfContents: true
showBreadcrumbs: false
---

I'm an ML Engineer and an MSE student in Computer Science at Johns Hopkins University, where I'm also a Graduate Course Assistant for Artificial Intelligence (EN.601.464).

Before Hopkins, I spent nearly three years at Azerbaijan's National Artificial Intelligence Center (AZBA). There I built [E-QANUN.AI](https://www.e-qanun.ai/), the national legal search platform: 50M+ vectors across Elasticsearch and Vespa, serving 7K+ monthly active users and 50K+ queries per month. I worked across the whole stack, from training retrievers and re-rankers to designing the RAG pipeline and fine-tuning LLMs on H200 clusters.

I also pretrained **AzNeoBERT**, the first modern Azerbaijani-only encoder model, on a 12.2B-token corpus.

Connect on [LinkedIn](https://www.linkedin.com/in/muradbaghirli/).

---

## What I Build

- Hybrid search combining SBERT-LaBSE embeddings with lexical methods (BM25, fuzzy matching, n-gram indexing), with fusion weights tuned via Optuna: **+24% precision**.
- Multistage RAG pipelines: query normalization → spell correction → intent detection → hybrid retrieval → mGTE re-ranking → generation, at **250ms** query-to-retrieval latency.
- Dense retrievers (SBERT, DPR, mGTE) trained with hard negatives, MNRL, and LambdaLoss, reaching **nDCG@10 of 0.878** and **MRR@10 of 0.810**.
- LLM fine-tuning with LoRA, including Qwen3.6-35B-A3B (MoE) for Azerbaijani document summarization on 6×H200.
- Encoder pretraining at scale with DeepSpeed ZeRO-2, Accelerate, and SLURM on 8×H200, reaching **1.24M tokens/sec**.
- Low-latency inference: Triton Inference Server deployments with up to **25× speedups**, and services sustaining 500 RPS at 180ms p95.

---

## Core Stack

| Area | Tools |
|---|---|
| Languages | Python · C · C++ · Java · SQL |
| ML / AI | PyTorch · scikit-learn · Transformers · LLMs & VLMs · RAG · Fine-tuning · Pre-training · Recommender Systems |
| Training | LoRA/QLoRA · PEFT · Axolotl · Unsloth · DeepSpeed · HuggingFace Accelerate · FlashAttention · Optuna |
| Inference & Optimization | vLLM · NVIDIA Triton Inference Server · HuggingFace TEI · ONNX · Quantization |
| Search / IR | Vespa · Elasticsearch · HNSW · BM25 · Learning-to-Rank · Hybrid Search |
| MLOps & Infra | MLflow · Weights & Biases · Docker · Git · FastAPI · SLURM · PostgreSQL · MongoDB |

---

## Education

**M.S.E. in Computer Science**: The Johns Hopkins University, Baltimore, MD
*Aug 2025 – May 2027* · GPA: 3.76/4.00

**B.Sc. in Computer Science**: French-Azerbaijani University (UFAZ), Baku, Azerbaijan
*Sep 2021 – Jul 2025* · GPA: 3.92/4.00
