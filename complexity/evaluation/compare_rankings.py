#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr


def compare_rankings(gt_dir: Path, gen_dir: Path, output_dir: Path) -> None:
  """
  Compares ranking CSV files found in both gt_dir and gen_dir.
  Calculates Spearman correlation and generates scatter plots.
  """
  if not gt_dir.exists():
    print(f"Error: GT directory '{gt_dir}' does not exist.")
    return
  if not gen_dir.exists():
    print(f"Error: GEN directory '{gen_dir}' does not exist.")
    return

  output_dir.mkdir(parents=True, exist_ok=True)

  # Get all csv files in GT dir
  gt_files = list(gt_dir.glob('*.csv'))

  results = []

  print(f"\n{'Model File':<40} | {'Rho':<8} | {'Tau':<8} | {'Top20':<8} | {'RMSE':<8} | {'N Images':<8}")
  print('-' * 100)

  for gt_file in gt_files:
    filename = gt_file.name
    gen_file = gen_dir / filename

    if not gen_file.exists():
      # Try to be flexible if headers are slightly different or just notify
      # print(f"Skipping {filename}: Not found in gen directory.")
      continue

    try:
      df_gt = pd.read_csv(gt_file)
      df_gen = pd.read_csv(gen_file)
    except Exception as e:
      print(f"Error reading {filename}: {e}")
      continue

    # Check required columns. Based on previous scripts, we expect 'image' and 'score'
    required_cols = {'image', 'score'}
    if not required_cols.issubset(df_gt.columns) or not required_cols.issubset(df_gen.columns):
      print(f"Skipping {filename}: Missing 'image' or 'score' columns.")
      continue

    # Merge on image ID (inner join to keep only images present in both sets)
    merged = pd.merge(df_gt, df_gen, on='image', suffixes=('_gt', '_gen'), how='inner')

    if len(merged) < 2:
      print(f"Skipping {filename}: Not enough overlapping images ({len(merged)}).")
      continue

    # 1. Spearman Rank Correlation
    rho, p_rho = spearmanr(merged['score_gt'], merged['score_gen'])

    # 2. Kendall's Tau
    tau, p_tau = kendalltau(merged['score_gt'], merged['score_gen'])

    # 3. Top-K Overlap (K=20)
    K = 20
    # Determine actual K if N is small
    actual_k = min(K, len(merged))

    # 'score' is difficulty (higher = harder)
    top_k_gt = set(merged.nlargest(actual_k, 'score_gt')['image'])
    top_k_gen = set(merged.nlargest(actual_k, 'score_gen')['image'])

    overlap_count = len(top_k_gt.intersection(top_k_gen))
    overlap_frac = overlap_count / actual_k if actual_k > 0 else 0.0

    # 4. RMSE (Root Mean Squared Error of scores)
    rmse = np.sqrt(((merged['score_gt'] - merged['score_gen']) ** 2).mean())

    results.append({
      'model_file': filename,
      'spearman_rho': rho,
      'kendall_tau': tau,
      'p_value_rho': p_rho,
      'top20_overlap': overlap_frac,
      'rmse': rmse,
      'n_images': len(merged)
    })

    print(f"{filename:<40} | {rho:.3f}    | {tau:.3f}    | {overlap_frac:.2f}     | {rmse:.3f}    | {len(merged)}")

    # --- Plotting ---
    plt.figure(figsize=(8, 8))

    # Scatter plot
    sns.scatterplot(
      data=merged,
      x='score_gt',
      y='score_gen',
      alpha=0.6,
      edgecolor=None
    )

    # Regression line for visual trend
    sns.regplot(
      data=merged,
      x='score_gt',
      y='score_gen',
      scatter=False,
      color='red',
      line_kws={'linestyle': '--', 'linewidth': 1, 'label': 'Trend'}
    )

    plt.title(f"Comparison: {filename}\nrho={rho:.3f}, tau={tau:.3f}, top20={overlap_frac:.2f}, N={len(merged)}")
    plt.xlabel('Ground Truth Difficulty Estimate')
    plt.ylabel('Generated Scanpaths Difficulty Estimate')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plot_path = output_dir / f"comparison_{filename.replace('.csv', '.png')}"
    plt.savefig(plot_path)
    plt.close()

  # Save summary CSV
  if results:
    summary_path = output_dir / 'comparison_summary.csv'
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print('-' * 85)
    print(f"Summary saved to {summary_path}")
    print(f"Comparison plots saved in {output_dir}")
  else:
    print('\nNo shared model files found or no overlapping images to compare.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(allow_abbrev=False, description='Compare ranking CSVs between Ground Truth and Generated models.')

  parser.add_argument(
    '--gt-dir',
    type=Path,
    default=Path('difficulty_rankings_gt_fixations'),
    help='Directory containing GT ranking CSVs'
  )
  parser.add_argument(
    '--gen-dir',
    type=Path,
    default=Path('difficulty_rankings_gazeformer'),
    help='Directory containing Generated ranking CSVs'
  )
  parser.add_argument(
    '--output-dir', dest='out_dir',
    type=Path,
    default=Path('ranking_comparisons_gazeformer'),
    help='Directory to save comparison results and plots'
  )

  args = parser.parse_args()

  print(f"Comparing rankings from:\n  GT:  {args.gt_dir}\n  GEN: {args.gen_dir}")
  compare_rankings(args.gt_dir, args.gen_dir, args.out_dir)
