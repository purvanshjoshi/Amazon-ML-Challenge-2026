# Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![RapidFuzz](https://img.shields.io/badge/SIMD-RapidFuzz-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle%20CPU-20BEFF.svg)](https://www.kaggle.com/)

A high-performance machine learning pipeline for multi-source business entity resolution at scale across heterogeneous datasets.

The system performs C-accelerated candidate blocking via character 3/4-gram TF-IDF sparse matrix cosine retrieval, extracts orthogonal pairwise similarity features using SIMD-accelerated string metrics, trains a regularized LightGBM binary classifier with GroupKFold cross-validation, and enforces 1-to-1 candidate cardinality using greedy maximum weighted bipartite matching.

---

## Quickstart

### Local Execution
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/run_pipeline.py --output-dir ./output
```

### Kaggle Execution
```python
!git clone https://github.com/purvanshjoshi/business-entity-resolution.git
%cd business-entity-resolution/code/business_entity_resolution
!pip install -q -r requirements.txt
!python run_pipeline.py --output-dir /kaggle/working/output --sample-train 100000 --top-k 12
```

---

## License
Distributed under the [MIT License](LICENSE).
