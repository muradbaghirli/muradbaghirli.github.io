---
title: "Experience"
date: 2025-01-01
showDate: false
showReadingTime: false
showAuthor: false
showPagination: false
showTableOfContents: false
showBreadcrumbs: false
---

## Graduate Course Assistant
**Johns Hopkins University** · Baltimore, MD, USA
*Sep 2026 – Present*

- Support 136 students in Artificial Intelligence (EN.601.464) through weekly office hours and grading.
- Contribute to course design, including homework and project development.

---

## Machine Learning Engineer
**National Artificial Intelligence Center (AZBA)** · Baku, Azerbaijan
*Jul 2025 – Aug 2026*

- Built [E-QANUN.AI](https://www.e-qanun.ai/), the national legal search platform: **50M+ vectors** across Elasticsearch and Vespa, **85% latency reduction** via HNSW indexing, serving **7K+ monthly active users** and **50K+ queries per month**.
- Architected a multistage RAG pipeline (query normalization → spell correction → intent detection → hybrid retrieval → mGTE re-ranking → decoder generation), achieving **250ms** query-to-retrieval latency.
- Designed hybrid search combining SBERT-LaBSE embeddings with lexical methods (BM25, fuzzy matching, n-gram indexing); tuned fusion weights with Optuna on synthetic queries, improving precision by **24%**.
- Fine-tuned Qwen3.6-35B-A3B (MoE) with LoRA for Azerbaijani document summarization on 6×H200, adapting attention and shared-expert FFN (42M params) with frozen vision; cut eval loss from **0.561 to 0.232**.

---

## Junior Machine Learning Engineer
**National Artificial Intelligence Center (AZBA)** · Baku, Azerbaijan
*Dec 2023 – Jun 2025*

- Trained and evaluated SBERT/DPR/mGTE retrievers on A100/H100/H200 clusters using hard negatives, MNRL, and LambdaLoss, reaching **nDCG@10 of 0.878** and **MRR@10 of 0.810**.
- Developed and stress-tested an auto-suggestion service sustaining **500 RPS at 180ms p95** latency with zero failures.
- Designed synthetic data generation pipelines producing **1B+ tokens** for retriever and reranker training.
