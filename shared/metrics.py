"""Correlation statistics shared by route evaluation code: Pearson/Spearman
with bootstrap confidence intervals, a permutation test, and
Benjamini-Hochberg FDR correction across categories.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
  """Pearson correlation and its p-value."""
  r, p = stats.pearsonr(x, y)
  return float(r), float(p)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
  """Spearman correlation and its p-value."""
  r, p = stats.spearmanr(x, y)
  return float(r), float(p)


def bootstrap_pearson_ci(x: np.ndarray, y: np.ndarray, n: int = 5000, seed: int = 42) -> tuple[float, float]:
  """Bootstrap 95% confidence interval for Pearson r, resampling pairs with replacement."""
  rng = np.random.RandomState(seed)
  values = []
  for _ in range(n):
    idx = rng.randint(0, len(x), len(x))
    if x[idx].std() > 0 and y[idx].std() > 0:
      values.append(stats.pearsonr(x[idx], y[idx])[0])
  values_arr = np.array(values)
  return float(np.percentile(values_arr, 2.5)), float(np.percentile(values_arr, 97.5))


def permutation_p(x: np.ndarray, y: np.ndarray, n: int = 5000, seed: int = 42) -> float:
  """Two-sided permutation test p-value for Pearson r, shuffling ``x``."""
  rng = np.random.RandomState(seed)
  observed_r = stats.pearsonr(x, y)[0]
  count = sum(abs(stats.pearsonr(rng.permutation(x), y)[0]) >= abs(observed_r) for _ in range(n))
  return float(count / n)


def fdr_bh(p_values: list[float]) -> list[float]:
  """Benjamini-Hochberg FDR-corrected q-values for a list of p-values."""
  n = len(p_values)
  order = np.argsort(p_values)
  q_values = np.array(p_values, dtype=float)
  for rank, idx in enumerate(order):
    q_values[idx] = p_values[idx] * n / (rank + 1)
  for i in range(n - 2, -1, -1):
    q_values[order[i]] = min(q_values[order[i]], q_values[order[i + 1]])
  return q_values.clip(0, 1).tolist()


def significance_stars(p_value: float) -> str:
  """Conventional significance stars for a p-value: ``***``, ``**``, ``*``, or ``ns``."""
  if p_value < 0.001:
    return '***'
  if p_value < 0.01:
    return '**'
  if p_value < 0.05:
    return '*'
  return 'ns'
