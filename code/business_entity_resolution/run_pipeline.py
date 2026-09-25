#!/usr/bin/env python3
"""
Production Pipeline Runner
Amazon ML Challenge 2026: Business Entity Resolution

Usage:
    python run_pipeline.py --data-dir /path/to/dataset --output-dir /path/to/output
"""

import os
import gc
import sys
import time
import argparse
import pickle
import psutil
import numpy as np
import pandas as pd
from collections import Counter

# Package imports
from src.preprocessing import clean_text, build_entity_lookup
from src.blocking import fast_tfidf_blocking
from src.features import extract_pair_features, extract_features_batch, FEATURE_NAMES
from src.model import train_lightgbm_model
from src.postprocessing import apply_graph_postprocessing, write_submission_tsv, write_candidates_tsv


def get_ram_usage() -> str:
    """Return current process RAM usage in GB."""
    return f"{psutil.Process().memory_info().rss / 1e9:.2f} GB"


def auto_detect_paths():
    """Locate dataset directories across Kaggle and local environments."""
    search_roots = [
        "/kaggle/input",
        ".",
        "..",
        "D:/ML_Challenge",
        "D:/ML_Challenge/DATA/UNZIPPED"
    ]
    train_dir, test_dir, utils_dir = None, None, None
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dp, dn, fn in os.walk(root):
            if "train_source1.tsv" in fn and not train_dir:
                train_dir = dp
            if "test_source1.tsv" in fn and not test_dir:
                test_dir = dp
            if "validate_submission.py" in fn and not utils_dir:
                utils_dir = dp
    return train_dir, test_dir, utils_dir


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Root dataset directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for TSV submissions")
    parser.add_argument("--sample-train", type=int, default=100000, help="S1 sample size for training")
    parser.add_argument("--top-k", type=int, default=12, help="Top candidates per S1 entity")
    parser.add_argument("--batch-size", type=int, default=2500, help="S1 batch size for sparse matrix multiplication")
    parser.add_argument("--max-neg", type=int, default=4, help="Max hard negatives per S1 entity")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory for caching checkpoints")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE", flush=True)
    print("=" * 70, flush=True)
    print(f"Initial RAM: {get_ram_usage()}", flush=True)

    # 1. Resolve Directories
    if args.data_dir:
        train_dir = os.path.join(args.data_dir, "train") if os.path.exists(os.path.join(args.data_dir, "train")) else args.data_dir
        test_dir = os.path.join(args.data_dir, "test") if os.path.exists(os.path.join(args.data_dir, "test")) else args.data_dir
        utils_dir = os.path.join(args.data_dir, "utils") if os.path.exists(os.path.join(args.data_dir, "utils")) else None
    else:
        train_dir, test_dir, utils_dir = auto_detect_paths()

    assert train_dir and os.path.exists(train_dir), f"Could not find train directory: {train_dir}"
    assert test_dir and os.path.exists(test_dir), f"Could not find test directory: {test_dir}"
    print(f"Train Directory:  {train_dir}")
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {args.output_dir}")

    # =========================================================================
    # STAGE 1: TRAINING DATA & MODEL
    # =========================================================================
    model_checkpoint = os.path.join(args.checkpoint_dir, "lgbm_model.pkl")
    
    if os.path.exists(model_checkpoint):
        print(f"\n[Stage 1] Loading cached trained model from {model_checkpoint}...")
        with open(model_checkpoint, "rb") as f:
            checkpoint_data = pickle.load(f)
            model = checkpoint_data["model"]
            best_tau = checkpoint_data["best_tau"]
            best_f05 = checkpoint_data["best_f05"]
            val_auc = checkpoint_data["val_auc"]
        print(f"  Loaded model: optimal tau* = {best_tau:.4f}, Val F_0.5 = {best_f05:.5f}")
    else:
        print(f"\n[Stage 1] Building Training Pairs & Training LightGBM...")
        t_stage1 = time.time()
        
        # Load Ground Truth
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        gt_df = pd.read_csv(gt_path, sep="\t")
        gt_map = {}
        for sid, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")):
            gt_map[sid] = set(m.strip() for m in mids.split(",") if m.strip())
        del gt_df
        gc.collect()
        
        # Load & Sample S1 Train
        s1_train = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
        if len(s1_train) > args.sample_train:
            s1_train = s1_train.sample(n=args.sample_train, random_state=42).reset_index(drop=True)
        s1_train["clean_name"] = s1_train["business_name"].apply(clean_text)
        
        # Load S2 and S3 Train
        s2_train = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t")
        s3_train = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t")
        s2_train["clean_name"] = s2_train["business_name"].apply(clean_text)
        s3_train["clean_name"] = s3_train["business_name"].apply(clean_text)
        s2s3_train = pd.concat([s2_train, s3_train], ignore_index=True)
        del s2_train, s3_train
        gc.collect()
        
        print(f"  Sampled S1 reference: {len(s1_train):,} | Candidate S2/S3 pool: {len(s2s3_train):,}")
        
        # Build Vectorized Lookups (instantaneous)
        s1_lookup = build_entity_lookup(s1_train)
        cand_lookup = build_entity_lookup(s2s3_train)
        
        # Block US and India
        train_candidates = {}
        for ctry in ["US", "India"]:
            s1_c = s1_train[s1_train["country"] == ctry]
            s2s3_c = s2s3_train[s2s3_train["country"] == ctry]
            print(f"  Running blocking for {ctry} training partition...", flush=True)
            train_candidates.update(fast_tfidf_blocking(s1_c, s2s3_c, top_k=10, batch_size=args.batch_size))
            
        del s1_train, s2s3_train
        gc.collect()
        
        # Construct Labeled Pairs Matrix
        print("  Extracting training features with hard negatives...")
        X_list, y_list, group_list, pair_meta = [], [], [], []
        
        for sid, cand_tuples in train_candidates.items():
            if sid not in s1_lookup:
                continue
            sn, sa, sp = s1_lookup[sid]
            true_set = gt_map.get(sid, set())
            cand_dict = dict(cand_tuples)
            
            # Positives
            for mid in true_set:
                if mid in cand_lookup:
                    cn, ca, cp = cand_lookup[mid]
                    score = cand_dict.get(mid, 0.5)
                    X_list.append(extract_pair_features(sn, sa, sp, cn, ca, cp, score))
                    y_list.append(1)
                    group_list.append(sid)
                    pair_meta.append((sid, mid))
                    
            # Hard Negatives
            negs = [c for c in cand_tuples if c[0] not in true_set and c[0] in cand_lookup]
            if len(negs) > args.max_neg:
                negs = negs[:args.max_neg]
                
            for nid, score in negs:
                cn, ca, cp = cand_lookup[nid]
                X_list.append(extract_pair_features(sn, sa, sp, cn, ca, cp, score))
                y_list.append(0)
                group_list.append(sid)
                pair_meta.append((sid, nid))
                
        del s1_lookup, cand_lookup, train_candidates
        gc.collect()
        
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        group_arr = np.array(group_list)
        print(f"  Constructed {len(X_arr):,} labeled pairs in {(time.time() - t_stage1):.1f}s | RAM: {get_ram_usage()}")
        
        # Train Model
        model, best_tau, best_f05, val_auc = train_lightgbm_model(
            X_arr, y_arr, group_arr, pair_meta, gt_map, random_state=42
        )
        
        # Save Model Checkpoint
        with open(model_checkpoint, "wb") as f:
            pickle.dump({
                "model": model,
                "best_tau": best_tau,
                "best_f05": best_f05,
                "val_auc": val_auc
            }, f)
            
        del X_arr, y_arr, group_arr, pair_meta, gt_map
        gc.collect()
        print(f"  Stage 1 complete! Saved model to {model_checkpoint} | RAM: {get_ram_usage()}")

    # =========================================================================
    # STAGE 2: TEST BLOCKING & INFERENCE (COUNTRY-BY-COUNTRY ISOLATION)
    # =========================================================================
    print(f"\n[Stage 2] Running Test Inference (France -> US -> India)...")
    t_stage2 = time.time()
    
    # Load S1 Test
    s1_test_all = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    all_test_s1_ids = s1_test_all["entity_id"].tolist()
    print(f"  Total S1 Reference Test Entities: {len(all_test_s1_ids):,}")
    
    # Load S2 and S3 Test
    s2_test_all = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
    s3_test_all = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")
    s2s3_test_all = pd.concat([s2_test_all, s3_test_all], ignore_index=True)
    del s2_test_all, s3_test_all
    gc.collect()
    
    all_candidates_dict = {}
    all_test_pairs = []
    all_test_probs = []
    
    for ctry in ["France", "US", "India"]:
        print(f"\n  --- Processing {ctry} ---")
        s1_c = s1_test_all[s1_test_all["country"] == ctry].copy()
        s2s3_c = s2s3_test_all[s2s3_test_all["country"] == ctry].copy()
        
        s1_c["clean_name"] = s1_c["business_name"].apply(clean_text)
        s2s3_c["clean_name"] = s2s3_c["business_name"].apply(clean_text)
        
        # 1. Blocking
        ctry_checkpoint = os.path.join(args.checkpoint_dir, f"blocking_{ctry}.pkl")
        cand_results = fast_tfidf_blocking(s1_c, s2s3_c, top_k=args.top_k, batch_size=args.batch_size, checkpoint_file=ctry_checkpoint)
        
        for sid, ctuples in cand_results.items():
            all_candidates_dict[sid] = [ct[0] for ct in ctuples]
            
        # 2. Lookups & Feature Extraction
        s1_c_lookup = build_entity_lookup(s1_c, country=ctry)
        s2s3_c_lookup = build_entity_lookup(s2s3_c, country=ctry)
        del s1_c, s2s3_c
        gc.collect()
        
        ctry_X = []
        ctry_pairs = []
        for sid, ctuples in cand_results.items():
            sn, sa, sp = s1_c_lookup[sid]
            for cid, score in ctuples:
                if cid in s2s3_c_lookup:
                    cn, ca, cp = s2s3_c_lookup[cid]
                    ctry_X.append(extract_pair_features(sn, sa, sp, cn, ca, cp, score))
                    ctry_pairs.append((sid, cid))
                    
        del s1_c_lookup, s2s3_c_lookup, cand_results
        gc.collect()
        
        # 3. Model Scoring
        if ctry_X:
            ctry_X_arr = np.array(ctry_X, dtype=np.float32)
            ctry_probs = model.predict(ctry_X_arr, num_iteration=model.best_iteration)
            all_test_pairs.extend(ctry_pairs)
            all_test_probs.extend(ctry_probs)
            del ctry_X_arr, ctry_probs
            
        del ctry_X, ctry_pairs
        gc.collect()
        print(f"  Finished {ctry} | Total scored pairs: {len(all_test_pairs):,} | RAM: {get_ram_usage()}")
        
    del s1_test_all, s2s3_test_all
    gc.collect()
    print(f"  Stage 2 complete in {(time.time() - t_stage2)/60:.1f} min!")

    # =========================================================================
    # STAGE 3: POST-PROCESSING & TSV SUBMISSIONS
    # =========================================================================
    print(f"\n[Stage 3] Writing TSV Submissions & Enforcing Graph Consistency...")
    
    # 1. Write candidate_pairs.tsv
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    print(f"  Writing {cand_path} ({len(all_test_s1_ids):,} rows)...")
    write_candidates_tsv(cand_path, all_test_s1_ids, all_candidates_dict)
    
    # 2. Maximum Weighted Bipartite Matching Post-Processing
    print("  Running greedy maximum weighted bipartite matching...")
    pair_records = sorted(zip(all_test_pairs, all_test_probs), key=lambda x: x[1], reverse=True)
    del all_test_pairs, all_test_probs, all_candidates_dict
    gc.collect()
    
    final_matches = apply_graph_postprocessing(
        pair_records,
        tau_match=best_tau,
        tau_singleton=0.30,
        max_matches=11
    )
    
    # 3. Write matching_results.tsv
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    print(f"  Writing {match_path} ({len(all_test_s1_ids):,} rows)...")
    write_submission_tsv(match_path, all_test_s1_ids, final_matches)
    
    # 4. Optional Validation Check
    if utils_dir:
        val_script = os.path.join(utils_dir, "validate_submission.py")
        if os.path.exists(val_script):
            print("\n" + "=" * 55)
            print("RUNNING OFFICIAL SUBMISSION VALIDATOR:")
            print("=" * 55)
            os.system(f"python \"{val_script}\" --matching \"{match_path}\" --candidate \"{cand_path}\" --test-dir \"{test_dir}\"")
            print("=" * 55)

    # =========================================================================
    # SUMMARY DASHBOARD
    # =========================================================================
    match_counts = [len(final_matches.get(sid, set())) for sid in all_test_s1_ids]
    singletons = sum(1 for c in match_counts if c == 0)
    total_time = (time.time() - t_start) / 60.0
    
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Total Pipeline Runtime:      {total_time:.1f} minutes")
    print(f"  Validation AUC-ROC:          {val_auc:.5f}")
    print(f"  Validation Macro F_0.5:      {best_f05:.5f}")
    print(f"  Optimal Decision Threshold:  {best_tau:.4f}")
    print(f"  Total Test Entities (S1):    {len(all_test_s1_ids):,}")
    print(f"  Predicted Singletons:        {singletons:,} ({singletons/len(all_test_s1_ids)*100:.2f}%)")
    print(f"  Average Matches per Entity:  {np.mean(match_counts):.2f}")
    print(f"  Maximum Matches per Entity:  {max(match_counts)}")
    print(f"\n  Match Distribution:")
    dist = Counter(match_counts)
    for k in range(min(12, max(dist.keys()) + 1)):
        count = dist.get(k, 0)
        bar = "#" * min(40, int(count / max(dist.values()) * 40))
        print(f"    {k:2d} matches: {count:>8,} ({count/len(all_test_s1_ids)*100:5.2f}%) {bar}")
    print("=" * 70)
    print(f"Submissions ready:")
    print(f"  -> {match_path}")
    print(f"  -> {cand_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
