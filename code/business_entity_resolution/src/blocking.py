"""
Blocking & Candidate Generation Module
High-speed C-accelerated TF-IDF character n-gram cosine retrieval using direct CSR array pointers.
"""

import time
import gc
import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def fast_tfidf_blocking(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    top_k: int = 12,
    batch_size: int = 2500,
    max_features: int = 150000,
    min_df: int = 3,
    max_df: float = 0.35,
    checkpoint_file: str = None
) -> dict:
    """
    Perform multi-resolution candidate blocking using character 3/4-gram TF-IDF
    sparse matrix multiplication with direct C/NumPy CSR pointer slicing.
    
    Args:
        s1_df: DataFrame of reference S1 entities
        s2s3_df: DataFrame of candidate S2/S3 entities
        top_k: Maximum number of top candidates to retrieve per S1 entity
        batch_size: S1 batch size for matrix multiplication (2,500 prevents memory bloat)
        max_features: Maximum TF-IDF feature vocabulary size
        min_df: Minimum document frequency to prune noise
        max_df: Maximum document frequency to prune overly common n-grams (e.g. 'inc', 'llc')
        checkpoint_file: Optional path to save/load cached blocking results
        
    Returns:
        dict: {s1_id: [(candidate_id, tfidf_cosine_score), ...]}
    """
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading cached blocking results from {checkpoint_file}...", flush=True)
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["clean_name"].values
    s1_addrs = s1_df["business_address"].fillna("").values
    
    s2s3_ids = s2s3_df["entity_id"].values
    s2s3_names = s2s3_df["clean_name"].values
    s2s3_addrs = s2s3_df["business_address"].fillna("").values
    
    # Combined representation: normalized name + first 3 address tokens
    s2s3_comb = [n + " " + " ".join(a.split()[:3]).lower() for n, a in zip(s2s3_names, s2s3_addrs)]
    s1_comb = [n + " " + " ".join(a.split()[:3]).lower() for n, a in zip(s1_names, s1_addrs)]
    
    # Character 3-gram TF-IDF with min_df and max_df to eliminate ubiquitous n-grams
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
        sublinear_tf=True
    )
    
    print(f"  Fitting TF-IDF on {len(s2s3_comb):,} candidate records...", flush=True)
    m2 = vectorizer.fit_transform(s2s3_comb)
    print(f"  Transforming {len(s1_comb):,} reference records...", flush=True)
    m1 = vectorizer.transform(s1_comb)
    
    del s2s3_comb, s1_comb, vectorizer
    gc.collect()
    
    candidates = defaultdict(list)
    total_s1 = len(s1_ids)
    
    # Direct pointer slicing on CSR matrix in fast batches of 2,500
    print(f"  Performing sparse matrix cosine retrieval in batches of {batch_size:,}...", flush=True)
    t_last_log = time.time()
    
    for bs in range(0, total_s1, batch_size):
        be = min(bs + batch_size, total_s1)
        sc = m1[bs:be] @ m2.T  # Sparse cosine similarity
        
        data = sc.data
        indices = sc.indices
        indptr = sc.indptr
        
        for r in range(be - bs):
            s, e = indptr[r], indptr[r + 1]
            if s < e:
                row_data = data[s:e]
                row_indices = indices[s:e]
                
                if len(row_data) <= top_k:
                    top_idx = range(len(row_data))
                else:
                    top_idx = np.argpartition(row_data, -top_k)[-top_k:]
                    
                sid = s1_ids[bs + r]
                for idx in top_idx:
                    candidates[sid].append((s2s3_ids[row_indices[idx]], float(row_data[idx])))
                    
        del sc
        
        # Live progress output every 15,000 entities or at the end
        if be % 15000 < batch_size or be == total_s1:
            pct = (be / total_s1) * 100
            rate = be / max(time.time() - t0, 0.1)
            print(f"    Progress: {be:,}/{total_s1:,} entities ({pct:.1f}%) | Speed: {rate:.0f} entities/s", flush=True)
            gc.collect()
        
    del m1, m2
    gc.collect()
    
    elapsed = time.time() - t0
    total_c = sum(len(v) for v in candidates.values())
    print(f"  Blocking complete: {len(s1_ids):,} entities -> {total_c:,} candidate pairs in {elapsed:.1f}s", flush=True)
    
    if checkpoint_file:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(dict(candidates), f)
            
    return candidates
