#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr


def compare_two_rankings(file1: Path, file2: Path, output_dir: Path, label1: str = 'File 1', label2: str = 'File 2') -> None:
  """
  Compares two specific ranking CSV files.
  Calculates Spearman correlation and generates a scatter plot.
  """
  if not file1.exists():
    print(f"Error: File 1 '{file1}' does not exist.")
    raise ValueError('Ranking comparison could not be completed.')
  if not file2.exists():
    print(f"Error: File 2 '{file2}' does not exist.")
    raise ValueError('Ranking comparison could not be completed.')

  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"\n{'Analysis':<40} | {'Rho':<8} | {'Tau':<8} | {'Top20':<8} | {'RMSE':<8} | {'N Images':<8}")
  print('-' * 100)

  try:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
  except Exception as e:
    print(f'Error reading files: {e}')
    raise ValueError('Ranking comparison could not be completed.')

  # Check required columns. Based on previous scripts, we expect 'image' and 'score'
  required_cols = {'image', 'score'}
  if not required_cols.issubset(df1.columns):
    print(f"Error: File 1 missing 'image' or 'score' columns.")
    raise ValueError('Ranking comparison could not be completed.')
  if not required_cols.issubset(df2.columns):
    print(f"Error: File 2 missing 'image' or 'score' columns.")
    raise ValueError('Ranking comparison could not be completed.')

  # Merge on image ID (inner join to keep only images present in both sets)
  merged = pd.merge(df1, df2, on='image', suffixes=('_f1', '_f2'), how='inner')

  if len(merged) < 2:
    print(f'Error: Not enough overlapping images ({len(merged)}).')
    raise ValueError('Ranking comparison could not be completed.')

  # 1. Spearman Rank Correlation
  rho, p_rho = spearmanr(merged['score_f1'], merged['score_f2'])

  # 2. Kendall's Tau
  tau, p_tau = kendalltau(merged['score_f1'], merged['score_f2'])

  # 3. Top-K Overlap (K=20)
  K = 20
  # Determine actual K if N is small
  actual_k = min(K, len(merged))

  # 'score' is difficulty (higher = harder)
  top_k_f1 = set(merged.nlargest(actual_k, 'score_f1')['image'])
  top_k_f2 = set(merged.nlargest(actual_k, 'score_f2')['image'])

  overlap_count = len(top_k_f1.intersection(top_k_f2))
  overlap_frac = overlap_count / actual_k if actual_k > 0 else 0.0

  # 4. RMSE (Root Mean Squared Error of scores)
  rmse = np.sqrt(((merged['score_f1'] - merged['score_f2']) ** 2).mean())

  comparison_name = f'{file1.stem}_vs_{file2.stem}'

  print(f'{comparison_name:<40} | {rho:.3f}    | {tau:.3f}    | {overlap_frac:.2f}     | {rmse:.3f}    | {len(merged)}')

  # --- Plotting ---
  plt.figure(figsize=(8, 8))

  # Scatter plot
  sns.scatterplot(
    data=merged,
    x='score_f1',
    y='score_f2',
    alpha=0.6,
    edgecolor=None
  )

  # Regression line for visual trend
  sns.regplot(
    data=merged,
    x='score_f1',
    y='score_f2',
    scatter=False,
    color='red',
    line_kws={'linestyle': '--', 'linewidth': 1, 'label': 'Trend'}
  )

  plt.title(f'Comparison: {comparison_name}\nrho={rho:.3f}, tau={tau:.3f}, top20={overlap_frac:.2f}, N={len(merged)}')
  plt.xlabel(f'{label1}\n({file1.name})')
  plt.ylabel(f'{label2}\n({file2.name})')
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.legend()

  plot_path = output_dir / f'comparison_{comparison_name}.png'
  plt.savefig(plot_path)
  plt.close()
  print(f'Plot saved to {plot_path}')

  # Save summary CSV
  results = [{
    'file1': str(file1),
    'file2': str(file2),
    'spearman_rho': rho,
    'kendall_tau': tau,
    'p_value_rho': p_rho,
    'top20_overlap': overlap_frac,
    'rmse': rmse,
    'n_images': len(merged)
  }]
  summary_path = output_dir / f'summary_{comparison_name}.csv'
  pd.DataFrame(results).to_csv(summary_path, index=False)
  print(f'Summary saved to {summary_path}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(allow_abbrev=False, description='Compare two specific ranking CSV files.')

  parser.add_argument(
    '--left-file', dest='file1',
    type=Path,
    required=True,
    help='Path to first ranking CSV'
  )
  parser.add_argument(
    '--right-file', dest='file2',
    type=Path,
    required=True,
    help='Path to second ranking CSV'
  )
  parser.add_argument(
    '--output-dir', dest='out_dir',
    type=Path,
    default=Path('comparison_results'),
    help='Directory to save comparison results and plots'
  )
  parser.add_argument('--left-label', dest='label1', type=str, default='File 1', help='Label for x-axis')
  parser.add_argument('--right-label', dest='label2', type=str, default='File 2', help='Label for y-axis')

  args = parser.parse_args()

  print(f'Comparing files:\n  1: {args.file1}\n  2: {args.file2}')
  compare_two_rankings(args.file1, args.file2, args.out_dir, args.label1, args.label2)
