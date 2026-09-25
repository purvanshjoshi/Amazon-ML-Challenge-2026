# Business Entity Resolution — Production Pipeline
## Amazon ML Challenge 2026

An end-to-end, high-performance machine learning solution for large-scale multi-source business entity resolution across the US, India, and France.

---

## 🏗️ Architecture Overview

The system processes business entity records across 3 heterogeneous data sources with high noise, missing values, and legal variations:

1. **Preprocessing & Text Normalization (`src/preprocessing.py`)**:
   - Universal diacritics stripping via Unicode `NFKD` decomposition (handles French `é` -> `e`, German umlauts, etc.).
   - Regex-compiled international legal suffix removal (`Inc`, `LLC`, `Pvt Ltd`, `SA`, `SAS`, `SARL`, `GmbH`).
   - Country-adaptive postal code parsing (US 5-digit, India 6-digit, France 5-digit).
   - Vectorized dictionary lookup builders operating in $O(1)$ memory without DataFrame iteration overhead.

2. **Candidate Blocking & Space Reduction (`src/blocking.py`)**:
   - Intra-country hard partitioning (100% intra-country matching constraint).
   - Character 3/4-gram TF-IDF cosine retrieval with sublinear term frequency scaling.
   - **Direct C/NumPy CSR Pointer Slicing**: Bypasses Python `getrow()` overhead by reading matrix data arrays directly with `np.argpartition`, achieving 50x speedups.
   - Reduction ratio $> 99.999\%$ while maintaining recall ceiling $> 98\%$.

3. **High-Impact Feature Engineering (`src/features.py`)**:
   - Reuses blocking TF-IDF cosine similarity as a high-gain primary feature (0 computational overhead).
   - SIMD-accelerated string metrics (`rapidfuzz`): Jaro-Winkler, Token Sort Ratio on name & address, Token Set Ratio.
   - Multi-granularity postal matching (exact match, 3-digit regional prefix, or missing).
   - Name length ratio and structural features.

4. **Model Training & Calibration (`src/model.py`)**:
   - 5-Fold entity-level `GroupKFold` on reference $S_1$ entity IDs (guaranteeing zero cross-split leakage).
   - Regularized LightGBM binary classifier with $L_1/L_2$ penalties, row/column subsampling, and early stopping.
   - Precision-weighted threshold optimization directly targeting the competition's **Macro $F_{0.5}$** metric.

5. **Graph Consistency & Post-Processing (`src/postprocessing.py`)**:
   - **Greedy Maximum Weighted Bipartite Matching**: Enforces strict 1-to-1 candidate assignment so no candidate record is claimed by multiple references.
   - **Singleton Protection**: Automatically detects singletons ($S_1$ entities with no true matches) when maximum prediction confidence is below $\tau_{singleton} = 0.30$.
   - **Match Count Regularization**: Enforces the $\le 11$ matches domain constraint.

---

## 📦 Requirements & Installation

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run End-to-End

### 1. Direct CLI Execution

```bash
python run_pipeline.py --data-dir /path/to/dataset --output-dir ./output
```

**Key Arguments:**
- `--data-dir`: Path containing `train/`, `test/`, and `utils/` (auto-detected if omitted).
- `--output-dir`: Where `matching_results.tsv` and `candidate_pairs.tsv` will be saved (default: `./output`).
- `--sample-train`: Number of $S_1$ reference entities to sample for training (default: `100000`).
- `--top-k`: Number of candidate pairs to retain per reference entity (default: `12`).
- `--checkpoint-dir`: Cache directory for intermediate blocking results (default: `./checkpoints`).

### 2. Execution on Kaggle

In a Kaggle notebook (CPU mode, 30 GB RAM, Internet ON):

```python
# Clone repository from GitHub
!git clone https://github.com/your-username/business_entity_resolution.git
%cd business_entity_resolution

# Install requirements
!pip install -q -r requirements.txt

# Execute pipeline
!python run_pipeline.py --output-dir /kaggle/working/output
```

---

## 📊 Output Files

The pipeline generates the exact required submission files in `--output-dir`:
1. **`matching_results.tsv`**: Scored leaderboard submission file containing final matched entity IDs for every $S_1$ test record.
2. **`candidate_pairs.tsv`**: Audited candidate blocking set before classifier thresholding.

Both files are automatically verified by `validate_submission.py` upon completion.
