# Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![RapidFuzz](https://img.shields.io/badge/SIMD-RapidFuzz-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end production module for large-scale multi-source business entity resolution.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### CLI Execution
```bash
python run_pipeline.py --output-dir ./output --sample-train 100000 --top-k 12
```

### Kaggle Execution
```python
!git clone https://github.com/purvanshjoshi/business-entity-resolution.git
%cd business-entity-resolution/code/business_entity_resolution
!pip install -q -r requirements.txt
!python run_pipeline.py --output-dir /kaggle/working/output --sample-train 100000 --top-k 12
```

---

## Pipeline Components

- **`src/preprocessing.py`**: Unicode NFKD diacritics stripping, regex legal suffix removal, and vectorized lookups.
- **`src/blocking.py`**: Character 3/4-gram TF-IDF cosine retrieval with direct C/NumPy CSR pointer slicing.
- **`src/features.py`**: Pairwise string similarities and blocking cosine scores via RapidFuzz SIMD.
- **`src/model.py`**: Regularized LightGBM with GroupKFold cross-validation and Macro F_0.5 threshold optimization.
- **`src/postprocessing.py`**: Greedy Maximum Weighted Bipartite Matching for 1-to-1 cardinality resolution.
- **`run_pipeline.py`**: CLI orchestrator with checkpointing and automated submission validation.
