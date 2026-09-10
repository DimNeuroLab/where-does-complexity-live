#!/usr/bin/env python3
"""
Compute baseline difficulty ranking by averaging N (number of fixations) per image.
Produces a CSV compatible with the visualization scripts.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

# -------------------------
# CV Helper Functions
# -------------------------

@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray

def make_image_stratified_cell_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    image_col: str = "img_idx",
    subj_col: str = "subj_idx",
    min_subjects_per_image: int = 2,
) -> List[Fold]:
    rng = np.random.default_rng(seed)
    
    # Group by image and list available subjects
    img_to_subs = (
        df[[image_col, subj_col]]
        .drop_duplicates()
        .groupby(image_col)[subj_col]
        .apply(lambda x: x.to_numpy())
        .to_dict()
    )

    # Assign each (image, subject) pair to a fold
    img_sub_to_fold = {}
    for img, subs in img_to_subs.items():
        subs = subs.copy()
        rng.shuffle(subs)
        if len(subs) < min_subjects_per_image:
            continue
        for j, s in enumerate(subs):
            img_sub_to_fold[(int(img), int(s))] = j % n_splits

    all_idx = np.arange(len(df))
    img_vals = df[image_col].to_numpy().astype(int)
    subj_vals = df[subj_col].to_numpy().astype(int)

    folds = []
    for f in range(n_splits):
        # Identify test indices for this fold
        is_test = np.zeros(len(df), dtype=bool)
        # Vectorized check is harder with dict lookups, doing loop for simplicity 
        # (speed should be fine for moderate N)
        # Optimization: Map (img, sub) -> fold in a series first
        
        # Optimized fold assignment
        # Create a lookup series
        assign_keys = pd.Series(img_sub_to_fold)
        current_keys = pd.MultiIndex.from_arrays([img_vals, subj_vals])
        # This is strictly faster assuming pandas align
        
        # Fallback to loop if complex:
        for t in range(len(df)):
            key = (img_vals[t], subj_vals[t])
            if key in img_sub_to_fold and img_sub_to_fold[key] == f:
                is_test[t] = True
                
        folds.append(Fold(f, all_idx[~is_test], all_idx[is_test]))
    return folds

def ranking_stability_point_estimates(fold_rankings: List[pd.Series], topk: int = 20) -> Dict[str, float]:
    # Align to common images
    if not fold_rankings:
        return {}
        
    common_images = set(fold_rankings[0].index)
    for r in fold_rankings[1:]:
        common_images &= set(r.index)
    
    common_images = sorted(list(common_images))
    print(f"Computing stability on {len(common_images)} common images across {len(fold_rankings)} folds.")
    
    means = [r.loc[common_images].values for r in fold_rankings]
    # Ranks (descending score = rank 0)
    # argsort(argsort(-x)) gives rank (0 = highest value)
    ranks = [np.argsort(np.argsort(-m)) for m in means]
    
    F = len(means)
    spears, ktaus, top_over = [], [], []

    from scipy.stats import kendalltau

    for a in range(F):
        for b in range(a + 1, F):
            ra, rb = ranks[a], ranks[b]
            
            # Spearman
            if len(ra) > 1:
                spears.append(float(np.corrcoef(ra, rb)[0, 1]))
            
            # Kendall Tau
            kt, _ = kendalltau(ra, rb)
            ktaus.append(kt)

            # TopK Overlap
            I = len(ra)
            k = min(topk, I)
            if k > 0:
                # Indices in the 'common_images' array corresponding to top k scores
                ka = set(np.argsort(-means[a])[:k])
                kb = set(np.argsort(-means[b])[:k])
                top_over.append(len(ka & kb) / k)

    return dict(
        n_folds=F,
        spearman_rank_mean=float(np.nanmean(spears)) if spears else np.nan,
        kendall_tau_mean=float(np.nanmean(ktaus)) if ktaus else np.nan,
        topk_overlap_mean=float(np.nanmean(top_over)) if top_over else np.nan,
    )

def compute_ranking(csv_path, output_dir, model_name="mean_N", do_cv=True, cv_splits=5, seed=42):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Required columns
    if 'image' not in df.columns:
        raise ValueError("Missing 'image' column in CSV")
    if 'N' not in df.columns:
        raise ValueError("Missing 'N' column in CSV")

    # Clean data (similar to estimate_complexity_fixations.py)
    # Ensure N is numeric and filter valid entries
    df['N'] = pd.to_numeric(df['N'], errors='coerce')
    df = df.dropna(subset=['N'])
    
    # Filter N > 0 (assuming consistent with model scripts which take logs)
    # If N represents fixation count, 0 usually implies no search or invalid trial for this metric
    initial_len = len(df)
    df = df[df['N'] > 0]
    if len(df) < initial_len:
        print(f"Filtered {initial_len - len(df)} records with N <= 0")

    print(f"Computing average N for {df['image'].nunique()} images...")
    
    # Compute mean N (Full Data)
    ranking = df.groupby('image')['N'].mean()
    ranking = ranking.sort_values(ascending=False)
    
    # -------------------------
    # CV Stability
    # -------------------------
    if do_cv:
        print(f"\nRunning {cv_splits}-fold CV for stability analysis...")
        # Prepare indices for CV
        d_cv = df.copy()
        
        # Factorize subject and image to get integer indices for stratification
        # Note: df['subject'] and df['image'] might be existing strings
        subj_idx, _ = pd.factorize(d_cv['subject'], sort=True)
        img_idx, _ = pd.factorize(d_cv['image'], sort=True)
        d_cv['subj_idx'] = subj_idx
        d_cv['img_idx'] = img_idx
        
        folds = make_image_stratified_cell_folds(
            d_cv, 
            n_splits=cv_splits,
            seed=seed,
            image_col='img_idx', 
            subj_col='subj_idx'
        )
        
        fold_rankings = []
        for i, fold in enumerate(folds):
            # Train data for this fold
            d_tr = d_cv.iloc[fold.train_idx]
            
            # Compute ranking on training data
            r = d_tr.groupby('image')['N'].mean()
            fold_rankings.append(r)
            print(f"  Fold {i+1}: {len(d_tr)} trials, {len(r)} images scored")
            
        stability_metrics = ranking_stability_point_estimates(fold_rankings)
        print("\nStability Metrics:")
        for k, v in stability_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        # Save stability metrics
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stab_path = output_dir / f"stability_{model_name}.csv"
        pd.DataFrame([stability_metrics]).to_csv(stab_path, index=False)
        print(f"Saved stability metrics to {stab_path}")

    # -------------------------
    # Final Output
    # -------------------------

    # Create output DataFrame
    out_df = ranking.to_frame(name='score')
    out_df.index.name = 'image'
    out_df = out_df.reset_index()
    
    # Extract task mapping if available
    if 'task' in df.columns:
        # Create map from image to task
        task_map = df[['image', 'task']].dropna().drop_duplicates(subset=['image'])
        # Handle case where an image might have multiple tasks (unlikely but possible) - take first
        task_map = task_map.set_index('image')['task']
        out_df['task'] = out_df['image'].map(task_map).fillna('unknown')
    else:
        out_df['task'] = 'unknown'

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_filename = f"full_ranking_{model_name}.csv"
    out_path = output_dir / out_filename
    
    out_df.to_csv(out_path, index=False)
    print(f"Saved ranking to {out_path}")
    print("\nTop 5 hardest images:")
    print(out_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute baseline difficulty ranking using average N.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the results.")
    parser.add_argument("--name", default="mean_N", help="Model name suffix for the output filename.")
    parser.add_argument("--no_cv", action="store_true", help="Skip CV stability analysis.")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of CV splits.")
    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    compute_ranking(args.csv, args.output_dir, args.name, do_cv=not args.no_cv, cv_splits=args.cv_splits, seed=args.seed)
