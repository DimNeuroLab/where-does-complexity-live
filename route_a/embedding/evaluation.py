"""Global evaluation of the embedding-conditioned complexity model.

Reproduces the metrics behind paper Table 6 (``--features all``) and the
corresponding rows of Appendix Table/Fig. B1 (``--features clip_only`` /
``--features dino_only``): pooled out-of-fold Pearson/Spearman correlation
(residualised and full-score), MAE, R-squared, and per-category breakdowns
with Benjamini-Hochberg FDR correction.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Subset

from route_a.embedding.model import ImageComplexityDataset, make_model
from route_a.embedding.training import add_common_arguments, build_dataset
from shared.constants import DEFAULT_SEED
from shared.metrics import bootstrap_pearson_ci, fdr_bh, pearson, permutation_p, significance_stars, spearman

CATEGORY_COLORS: list[str] = [
  '#4472C4', '#ED7D31', '#A9D18E', '#FF0000', '#FFC000',
  '#9B59B6', '#2ECC71', '#E74C3C', '#1ABC9C', '#F39C12',
  '#7F8C8D', '#C0392B', '#2980B9', '#27AE60', '#8E44AD', '#D35400',
]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
  """Load a fold checkpoint written by :mod:`route_a.embedding.training`.

  :raises ValueError: If the checkpoint is a deployment-only checkpoint
    (retrained on the full dataset, not safe to evaluate on held-out data).
  """
  checkpoint = torch.load(checkpoint_path, map_location=device)
  state_dict = checkpoint.get('model', checkpoint)
  if 'warning' in checkpoint:
    raise ValueError(f'{checkpoint["warning"]} Use a per-fold checkpoint for evaluation.')

  model = make_model(checkpoint.get('arch', 'small'), checkpoint.get('features', 'all'))
  model.load_state_dict(state_dict)
  model.to(device).eval()
  return model, checkpoint


@torch.no_grad()
def run_inference(
  model: torch.nn.Module,
  dataset: Subset | ImageComplexityDataset,
  device: torch.device,
  batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
  """Run inference over a dataset or a :class:`~torch.utils.data.Subset` of one.

  :returns: ``(preds, targets, tasks, log_vars)``.
  """
  if isinstance(dataset, Subset):
    base_samples = dataset.dataset.samples
    index_map = dataset.indices
  else:
    base_samples = dataset.samples
    index_map = list(range(len(dataset)))

  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
  model.eval()
  all_mu: list[float] = []
  all_log_var: list[float] = []
  all_target: list[float] = []
  all_task: list[str] = []

  sample_idx = 0
  for batch in loader:
    mu, log_var = model(
      batch['dino_early'].to(device), batch['dino_mid'].to(device), batch['dino_late'].to(device),
      batch['clip_img'].to(device), batch['category_idx'].to(device), batch['clip_text'].to(device),
    )
    batch_n = mu.shape[0]
    all_mu.extend(mu.cpu().tolist())
    all_log_var.extend(log_var.cpu().tolist())
    all_target.extend(batch['score'].tolist())
    all_task.extend(base_samples[index_map[sample_idx + i]][3] for i in range(batch_n))
    sample_idx += batch_n

  return (
    np.array(all_mu, dtype=np.float32),
    np.array(all_target, dtype=np.float32),
    all_task,
    np.array(all_log_var, dtype=np.float32),
  )


def compute_metrics(
  preds: np.ndarray,
  targets: np.ndarray,
  tasks: list[str],
  log_vars: np.ndarray,
  category_means: dict[str, float],
  n_boot: int = 5000,
  n_perm: int = 5000,
  seed: int = DEFAULT_SEED,
) -> dict:
  """Compute residual and full-score global and per-category metrics."""
  r, r_p = pearson(preds, targets)
  rho, rho_p = spearman(preds, targets)
  ci_lo, ci_hi = bootstrap_pearson_ci(preds, targets, n=n_boot, seed=seed)
  rho_ci_lo, rho_ci_hi = bootstrap_pearson_ci(preds, targets, n=n_boot, seed=seed + 1)
  perm_p = permutation_p(preds, targets, n=n_perm, seed=seed)

  metrics: dict = {
    'n_val': len(preds),
    'pearson_residual': r, 'pearson_residual_p': r_p, 'pearson_residual_ci': [ci_lo, ci_hi],
    'pearson_perm_p': perm_p, 'n_permutations': n_perm,
    'spearman_residual': rho, 'spearman_residual_p': rho_p, 'spearman_residual_ci': [rho_ci_lo, rho_ci_hi],
    'mae': float(mean_absolute_error(targets, preds)),
    'r2': float(r2_score(targets, preds)),
    'mean_sigma': float(np.mean(np.exp(0.5 * log_vars))),
    # Aliases kept for readability elsewhere in this module.
    'pearson': r, 'pearson_p': r_p, 'pearson_ci': [ci_lo, ci_hi],
    'spearman': rho, 'spearman_p': rho_p, 'spearman_ci': [rho_ci_lo, rho_ci_hi],
  }

  category_mean_arr = np.array([category_means.get(task, 0.0) for task in tasks], dtype=np.float32)
  preds_full = preds + category_mean_arr
  targets_full = targets + category_mean_arr
  rf, rf_p = pearson(preds_full, targets_full)
  rhof, rhof_p = spearman(preds_full, targets_full)
  ci_f_lo, ci_f_hi = bootstrap_pearson_ci(preds_full, targets_full, n=n_boot, seed=seed + 2)
  rho_f_ci_lo, rho_f_ci_hi = bootstrap_pearson_ci(preds_full, targets_full, n=n_boot, seed=seed + 3)
  metrics.update({
    'pearson_full': rf, 'pearson_full_p': rf_p, 'pearson_full_ci': [ci_f_lo, ci_f_hi],
    'spearman_full': rhof, 'spearman_full_p': rhof_p, 'spearman_full_ci': [rho_f_ci_lo, rho_f_ci_hi],
    'mae_full': float(mean_absolute_error(targets_full, preds_full)),
    'r2_full': float(r2_score(targets_full, preds_full)),
  })

  category_preds: dict[str, list[float]] = defaultdict(list)
  category_targets: dict[str, list[float]] = defaultdict(list)
  for pred, target, task in zip(preds, targets, tasks):
    category_preds[task].append(pred)
    category_targets[task].append(target)

  category_metrics: dict[str, dict] = {}
  category_pvals: list[float] = []
  category_names: list[str] = []
  for category in sorted(category_preds):
    cat_p = np.array(category_preds[category])
    cat_t = np.array(category_targets[category])
    if len(cat_p) < 4:
      continue
    cr, cr_p = pearson(cat_p, cat_t)
    crho, crho_p = spearman(cat_p, cat_t)
    cat_ci = bootstrap_pearson_ci(cat_p, cat_t, n=min(n_boot, 2000), seed=seed)
    category_metrics[category] = {
      'pearson': cr, 'pearson_p': cr_p, 'pearson_ci': list(cat_ci),
      'spearman': crho, 'spearman_p': crho_p,
      'mae': float(mean_absolute_error(cat_t, cat_p)), 'n': int(len(cat_p)),
    }
    category_pvals.append(cr_p)
    category_names.append(category)

  if category_pvals:
    for category, q_value in zip(category_names, fdr_bh(category_pvals)):
      category_metrics[category]['pearson_p_fdr'] = float(q_value)

  metrics['category_wise'] = category_metrics
  return metrics


def plot_calibration(preds: np.ndarray, targets: np.ndarray, tasks: list[str], save_dir: Path, tag: str) -> None:
  unique_categories = sorted(set(tasks))
  category_color = {category: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, category in enumerate(unique_categories)}

  fig, ax = plt.subplots(figsize=(7, 6))
  for category in unique_categories:
    mask = [task == category for task in tasks]
    ax.scatter(targets[mask], preds[mask], c=category_color[category], label=category, alpha=0.55, s=18, linewidths=0)

  lims = [min(targets.min(), preds.min()) - 0.1, max(targets.max(), preds.max()) + 0.1]
  ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
  ax.set_xlim(lims)
  ax.set_ylim(lims)
  ax.set_xlabel('Ground truth (residual)')
  ax.set_ylabel('Predicted (residual)')
  ax.set_title('Predicted vs ground truth')
  ax.legend(fontsize=6, ncol=2, loc='upper left')
  ax.grid(True, alpha=0.25)

  fig.tight_layout()
  for ext in ('png', 'pdf'):
    fig.savefig(save_dir / f'{tag}_calibration_scatter.{ext}', dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'  Saved -> {save_dir}/{tag}_calibration_scatter.png')


def plot_category_bars(category_metrics: dict[str, dict], save_dir: Path, tag: str) -> None:
  categories = sorted(category_metrics, key=lambda c: category_metrics[c]['pearson'], reverse=True)
  rs = [category_metrics[c]['pearson'] for c in categories]
  cis = [category_metrics[c].get('pearson_ci', [0, 0]) for c in categories]
  significant = [category_metrics[c].get('pearson_p_fdr', 1.0) < 0.05 for c in categories]
  ns = [category_metrics[c]['n'] for c in categories]

  lo = [r - ci[0] for r, ci in zip(rs, cis)]
  hi = [ci[1] - r for r, ci in zip(rs, cis)]

  fig, ax = plt.subplots(figsize=(10, 6))
  y = np.arange(len(categories))
  colors = ['#2a9d8f' if sig else '#e76f51' for sig in significant]

  ax.barh(y, rs, color=colors, edgecolor='black', linewidth=0.5, height=0.6, alpha=0.85)
  ax.errorbar(rs, y, xerr=[lo, hi], fmt='none', ecolor='black', elinewidth=1, capsize=3, capthick=1)
  for i, (r, n, sig, ci) in enumerate(zip(rs, ns, significant, cis)):
    star = ' *' if sig else ''
    x_end = max(r, ci[1]) + 0.03 if r >= 0 else min(r, ci[0]) - 0.03
    ax.text(x_end, i, f'r={r:.3f}{star} (n={n})', va='center', ha='left' if r >= 0 else 'right', fontsize=7.5)

  ax.set_yticks(y)
  ax.set_yticklabels(categories)
  ax.invert_yaxis()
  ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
  ax.set_xlabel('Pearson r')
  ax.set_title('Per-category Pearson r with 95% bootstrap CI\n(green = FDR q<0.05, red = not significant)')
  ax.margins(x=0.3)
  ax.grid(axis='x', alpha=0.3)

  fig.tight_layout()
  for ext in ('png', 'pdf'):
    fig.savefig(save_dir / f'{tag}_category_bars.{ext}', dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'  Saved -> {save_dir}/{tag}_category_bars.png')


def _print_summary(metrics: dict) -> None:
  r, r_p = metrics['pearson'], metrics['pearson_p']
  rho, rho_p = metrics['spearman'], metrics['spearman_p']
  rf, rf_p = metrics['pearson_full'], metrics['pearson_full_p']
  rhof, rhof_p = metrics['spearman_full'], metrics['spearman_full_p']

  print(f'\n  Global metrics ({metrics["n_val"]} val samples):')
  print('    --- Residual (within-category) ---')
  print(f'    Pearson r  : {r:.6f}  p={r_p:.6e} {significance_stars(r_p)}')
  print(f'    Spearman r : {rho:.6f}  p={rho_p:.6e} {significance_stars(rho_p)}')
  print('    --- Full score (residual + category mean) ---')
  print(f'    Pearson r  : {rf:.6f}  p={rf_p:.6e} {significance_stars(rf_p)}')
  print(f'    Spearman r : {rhof:.6f}  p={rhof_p:.6e} {significance_stars(rhof_p)}')
  print(f'    MAE        : {metrics["mae_full"]:.6f}')
  print(f'    R2         : {metrics["r2_full"]:.6f}')
  n_sig = sum(1 for m in metrics['category_wise'].values() if m.get('pearson_p_fdr', 1.0) < 0.05)
  print(f'    {n_sig}/{len(metrics["category_wise"])} categories significant (BH-FDR q<0.05)')


def evaluate_kfold(
  dataset: ImageComplexityDataset,
  checkpoint_glob: str,
  device: torch.device,
  fig_dir: Path,
  out_dir: Path,
  n_boot: int,
  n_perm: int,
) -> None:
  """Pool out-of-fold predictions across every checkpoint matching ``checkpoint_glob`` and report metrics."""
  checkpoint_paths = sorted(glob.glob(checkpoint_glob))
  if not checkpoint_paths:
    print(f'  No checkpoints matched: {checkpoint_glob}')
    return

  fold_rs: list[float] = []
  all_preds: list[float] = []
  all_targets: list[float] = []
  all_tasks: list[str] = []
  all_log_vars: list[float] = []
  arch = 'model'
  features = 'model'

  for checkpoint_path in checkpoint_paths:
    print(f'\n  Fold checkpoint: {Path(checkpoint_path).name}')
    try:
      model, checkpoint = load_model(Path(checkpoint_path), device)
    except ValueError as error:
      print(f'  Skipping: {error}')
      continue

    val_idx = checkpoint.get('val_idx')
    if val_idx is None:
      print('  No val_idx in checkpoint, skipping (re-run training.py)')
      continue

    arch = checkpoint.get('arch', arch)
    features = checkpoint.get('features', features)
    preds, targets, tasks, log_vars = run_inference(model, Subset(dataset, val_idx), device)
    fold_r, _ = pearson(preds, targets)
    fold_rs.append(fold_r)
    all_preds.extend(preds.tolist())
    all_targets.extend(targets.tolist())
    all_tasks.extend(tasks)
    all_log_vars.extend(log_vars.tolist())
    print(f'    n={len(preds)}  fold r={fold_r:.4f}')

  print(f'\n  Per-fold Pearson r: {[f"{r:.4f}" for r in fold_rs]}')
  print(f'  Mean r: {np.mean(fold_rs):.4f} +/- {np.std(fold_rs):.4f}')

  metrics = compute_metrics(
    np.array(all_preds, dtype=np.float32), np.array(all_targets, dtype=np.float32), all_tasks,
    np.array(all_log_vars, dtype=np.float32), dataset.category_means, n_boot=n_boot, n_perm=n_perm,
  )
  metrics['fold_pearson_r'] = fold_rs
  metrics['fold_pearson_mean'] = float(np.mean(fold_rs))
  metrics['fold_pearson_std'] = float(np.std(fold_rs))
  _print_summary(metrics)

  tag = f'route_a_embedding_{features}_{arch}'
  plot_calibration(np.array(all_preds), np.array(all_targets), all_tasks, fig_dir, tag)
  plot_category_bars(metrics['category_wise'], fig_dir, tag)

  out_path = out_dir / f'metrics_{tag}.json'
  with open(out_path, 'w') as handle:
    json.dump(metrics, handle, indent=2, default=str)
  print(f'  Metrics -> {out_path}')


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Evaluate the embedding-conditioned complexity model.')
  add_common_arguments(parser)
  parser.add_argument('--out', type=Path, required=True, help='Output directory for metrics JSON and figures.')
  parser.add_argument(
    '--checkpoint-glob', required=True,
    help="Glob matching one condition's fold checkpoints, for example "
         "'<out>/checkpoints/route_a_embedding_all_small_noise0.00_fold*of5.pt'.",
  )
  parser.add_argument('--n-boot', type=int, default=5000)
  parser.add_argument('--n-perm', type=int, default=5000)
  parser.add_argument('--device', default='cuda')
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  fig_dir = args.out / 'figures'
  fig_dir.mkdir(parents=True, exist_ok=True)

  dataset = build_dataset(args)
  evaluate_kfold(dataset, args.checkpoint_glob, device, fig_dir, args.out, args.n_boot, args.n_perm)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
