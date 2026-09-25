# 🏆 Amazon ML Challenge 2026: Business Entity Resolution at Scale

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![RapidFuzz](https://img.shields.io/badge/SIMD-RapidFuzz-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle%20CPU-20BEFF.svg)](https://www.kaggle.com/)

---

## 📌 Problem Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources—each contributing partial, noisy fragments of information about the same real-world entities. Determining which records refer to the same business without shared unique identifiers is known as **Entity Resolution (ER)**.

Given records across **3 independent data sources** with high noise, missing values, legal abbreviations, OCR errors, and address reorderings across **US**, **India**, and **France** (zero-shot unseen test country):
- **Source 1 ($S_1$):** Deduplicated reference entity list.
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Noisy candidate records.
- **Goal:** Determine all matching records from $S_2$ and $S_3$ for every $S_1$ reference entity.
- **Evaluation Metric:** **Macro $F_{0.5}$** (precision weighted 2× over recall, evaluated per entity including singletons).

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          RAW INPUT DATA                                │
│         S1 (1.7M)      │      S2 (4.9M)       │      S3 (5.1M)         │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │  COUNTRY PARTITION  │ (100% intra-country matches)
                  │ US | India | France │
                  └─────────┬──────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│ STAGE 1: C-ACCELERATED MULTI-RESOLUTION BLOCKING                       │
│ • Character 3/4-Gram TF-IDF Sparse Cosine Retrieval                    │
│ • Direct C/NumPy CSR Pointer Slicing via np.argpartition (50x speedup) │
│ • Reduction Ratio: > 99.999% | Recall Ceiling: > 98%                   │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│ STAGE 2: DECISIVE PAIRWISE FEATURE ENGINEERING                         │
│ • TF-IDF Cosine Similarity (from blocking, 0 extra compute cost)       │
│ • SIMD Jaro-Winkler & Token Sort Ratios (Name & Address)               │
│ • Postal Code Multi-Granularity Match (Exact, 3-digit prefix)          │
│ • Length Ratios & Missingness Flags                                    │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│ STAGE 3: REGULARIZED LIGHTGBM & THRESHOLD OPTIMIZATION                 │
│ • Leak-Free 5-Fold Entity-Level GroupKFold CV on S1                    │
│ • Anti-overfitting L1/L2 penalties + Subsampling                       │
│ • Grid Search Optimization for Macro F_0.5 Threshold tau*              │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│ STAGE 4: GRAPH CONSISTENCY & CARDINALITY POST-PROCESSING               │
│ • Greedy Maximum Weighted Bipartite Matching (1-to-1 cardinality)      │
│ • Singleton Protection (P_max < 0.30 -> empty match set)               │
│ • Domain Constraint Capping (matches <= 11)                            │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
   matching_results.tsv            candidate_pairs.tsv
    (Leaderboard TSV)               (Auditing TSV)
```

---

## 📁 Repository Structure

```
.
├── code/
│   └── business_entity_resolution/      # Standalone submission package
│       ├── src/
│       │   ├── __init__.py
│       │   ├── preprocessing.py         # Vectorized lookups, diacritics, regex
│       │   ├── blocking.py              # TF-IDF sparse cosine & CSR pointer slicing
│       │   ├── features.py              # SIMD string distance features
│       │   ├── model.py                 # LightGBM training & Macro F_0.5 tuning
│       │   └── postprocessing.py        # Maximum weighted bipartite matching
│       ├── run_pipeline.py              # Production CLI runner with checkpointing
│       ├── requirements.txt             # Pinned package versions
│       └── README.md                    # Module documentation
├── docs/                                # Technical documentation suite
│   ├── 01_DATASET_AND_EDA.md
│   ├── 06_NOVEL_BLOCKING_STRATEGY.md
│   ├── 07_ADVANCED_FEATURE_ENGINEERING.md
│   ├── 08_MODEL_TRAINING_AND_CASCADE.md
│   ├── 09_FRANCE_DOMAIN_ADAPTATION.md
│   ├── 10_GRAPH_POST_PROCESSING.md
│   ├── 11_KAGGLE_IMPLEMENTATION_GUIDE.md
│   └── README.md
├── notebooks/                           # Jupyter notebooks
│   ├── kaggle_runner.ipynb              # Production Kaggle runner
│   └── master_entity_resolution_pipeline.ipynb # All-in-one standalone notebook
├── .gitignore
├── PROBLEM_STATEMENT.md
└── README.md
```

---

## ⚡ Performance Highlights & Speedups

- **Zero `iterrows()` Overhead:** Lookup dictionaries for 10.3M records are constructed in **~2 seconds** via `dict(zip(...))`, eliminating the 40+ minute pandas iteration bottleneck.
- **Direct CSR Pointer Indexing:** Directly accesses matrix pointers (`indptr`, `data`, `indices`) with `np.argpartition`, achieving a **50× speedup** over `scipy.getrow()`.
- **Low Memory Footprint:** Country-isolated execution ensures peak RAM usage remains strictly within **8–12 GB** on standard 30 GB Kaggle CPU instances.
- **End-to-End Runtime:** Full training, test candidate blocking, inference, and post-processing finish in **under 20 minutes**.

---

## 🚀 Quickstart & Kaggle Execution

### 1. Local CLI Execution
```bash
# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Run the complete pipeline
python code/business_entity_resolution/run_pipeline.py --output-dir ./output
```

### 2. Execution on Kaggle
In a Kaggle Notebook (Settings: **CPU**, **30 GB RAM**, **Internet: ON**):

```python
!git clone https://github.com/purvanshjoshi/Amazon-ML-Challenge-2026.git
%cd Amazon-ML-Challenge-2026/code/business_entity_resolution
!pip install -q -r requirements.txt
!python run_pipeline.py --output-dir /kaggle/working/output --sample-train 100000 --top-k 12
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
