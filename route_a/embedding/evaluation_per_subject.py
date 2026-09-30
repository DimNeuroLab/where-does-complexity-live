"""Per-subject evaluation of the embedding-conditioned complexity model.

Reproduces the per-subject metrics and bar chart behind paper Fig. 6
(``--features all``) and the per-subject part of Appendix Fig. B1
(``--features clip_only`` / ``--features dino_only``).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from route_a.embedding.evaluation import load_model, run_inference
from route_a.embedding.model import ImageComplexityDataset
from route_a.embedding.training import add_common_arguments, build_dataset
from shared.constants import NSD_SUBJECTS
from shared.metrics import bootstrap_pearson_ci, pearson, significance_stars, spearman
from shared.nsd_utils import list_training_images, parse_image_filename


def per_subject_metrics(
  nsd_root: Path,
  preds_by_sample: dict[int, float],
  targets_by_sample: dict[int, float],
  nsd_id_by_sample: dict[int, int],
  task_by_sample: dict[int, str],
  category_means: dict[str, float],
  n_boot: int = 2000,
) -> dict[str, dict]:
  """Compute Pearson/Spearman/MAE/R-squared per subject, over every
  ``(image, category)`` pair that subject's training set contains.
  """
  results: dict[str, dict] = {}
  for subject in NSD_SUBJECTS:
    subject_nsd_ids = {
      parse_image_filename(path.name)[2]
      for path in list_training_images(nsd_root, subject)
    }
    sample_ids = [sample_id for sample_id, nsd_id in nsd_id_by_sample.items() if nsd_id in subject_nsd_ids]
    if len(sample_ids) < 10:
      print(f'  {subject}: only {len(sample_ids)} val samples, skipping')
      continue

    preds = np.array([preds_by_sample[i] for i in sample_ids], dtype=np.float32)
    targets = np.array([targets_by_sample[i] for i in sample_ids], dtype=np.float32)

    r, r_p = pearson(preds, targets)
    rho, rho_p = spearman(preds, targets)
    ci_lo, ci_hi = bootstrap_pearson_ci(preds, targets, n=n_boot)
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))

    entry: dict = {
      'pearson': r, 'pearson_p': r_p, 'pearson_ci': [ci_lo, ci_hi],
      'spearman': rho, 'spearman_p': rho_p,
      'mae': float(np.mean(np.abs(preds - targets))),
      'r2': 1 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
      'n': len(sample_ids),
    }

    category_mean_arr = np.array([category_means.get(task_by_sample[i], 0.0) for i in sample_ids], dtype=np.float32)
    preds_full = preds + category_mean_arr
    targets_full = targets + category_mean_arr
    rf, rf_p = pearson(preds_full, targets_full)
    rhof, rhof_p = spearman(preds_full, targets_full)
    ci_f_lo, ci_f_hi = bootstrap_pearson_ci(preds_full, targets_full, n=n_boot)
    entry.update({
      'pearson_full': rf, 'pearson_full_p': rf_p, 'pearson_full_ci': [ci_f_lo, ci_f_hi],
      'spearman_full': rhof, 'spearman_full_p': rhof_p,
    })

    results[subject] = entry
  return results


def plot_per_subject_bar(subject_metrics: dict[str, dict], save_dir: Path, tag: str) -> None:
  """Blue/orange Pearson r + Spearman rho bar chart per subject, using the
  full-score correlation (residual plus category mean restored), matching
  paper Fig. 6 / Appendix Fig. B1.
  """
  subjects = sorted(subject_metrics)
  rs = [subject_metrics[s].get('pearson_full', subject_metrics[s]['pearson']) for s in subjects]
  rhos = [subject_metrics[s].get('spearman_full', subject_metrics[s]['spearman']) for s in subjects]
  ns = [subject_metrics[s]['n'] for s in subjects]

  x = np.arange(len(subjects))
  width = 0.35
  fig, ax = plt.subplots(figsize=(11, 5.5))

  bars1 = ax.bar(x - width / 2, rs, width, label='Pearson r', color='#4472C4', edgecolor='black', linewidth=0.5)
  bars2 = ax.bar(x + width / 2, rhos, width, label='Spearman rho', color='#ED7D31', edgecolor='black', linewidth=0.5)
  for bar, value in zip(list(bars1) + list(bars2), rs + rhos):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
  for i, n in enumerate(ns):
    ax.text(x[i], -0.05, f'n={n}', ha='center', va='top', fontsize=7, color='#666666')

  mean_r = float(np.nanmean(rs))
  mean_rho = float(np.nanmean(rhos))
  ax.axhline(mean_r, color='#4472C4', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Mean Pearson r = {mean_r:.3f}')
  ax.axhline(mean_rho, color='#ED7D31', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Mean Spearman rho = {mean_rho:.3f}')

  ax.set_ylabel('Correlation')
  ax.set_title('Per-subject complexity prediction performance (all subjects p < 0.001)', fontsize=10)
  ax.set_xticks(x)
  ax.set_xticklabels(subjects)
  ax.set_ylim(0, max(rs + rhos) * 1.25)
  ax.grid(axis='y', alpha=0.3)
  ax.legend(loc='upper right', fontsize=8)

  fig.tight_layout()
  for ext in ('png', 'pdf'):
    fig.savefig(save_dir / f'{tag}_per_subject_bar.{ext}', dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'  Saved -> {save_dir}/{tag}_per_subject_bar.png')


def evaluate_kfold_per_subject(
  nsd_root: Path,
  dataset: ImageComplexityDataset,
  checkpoint_glob: str,
  device: torch.device,
  fig_dir: Path,
  out_dir: Path,
  n_boot: int,
) -> None:
  checkpoint_paths = sorted(glob.glob(checkpoint_glob))
  if not checkpoint_paths:
    print(f'  No checkpoints matched: {checkpoint_glob}')
    return

  feat_to_nsd = dataset.feat_idx_to_nsd
  preds_by_sample: dict[int, float] = {}
  targets_by_sample: dict[int, float] = {}
  nsd_id_by_sample: dict[int, int] = {}
  task_by_sample: dict[int, str] = {}
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
    preds, targets, tasks, _ = run_inference(model, Subset(dataset, val_idx), device)

    for i, sample_id in enumerate(val_idx):
      feat_idx = dataset.samples[sample_id][0]
      nsd_id = feat_to_nsd.get(feat_idx)
      if nsd_id is not None:
        preds_by_sample[sample_id] = float(preds[i])
        targets_by_sample[sample_id] = float(targets[i])
        nsd_id_by_sample[sample_id] = nsd_id
        task_by_sample[sample_id] = tasks[i]

  print(f'\n  Total val pairs (image x category): {len(preds_by_sample)}')

  subject_metrics = per_subject_metrics(
    nsd_root, preds_by_sample, targets_by_sample, nsd_id_by_sample, task_by_sample,
    dataset.category_means, n_boot=n_boot,
  )

  print(f'\n  {"Subject":10s} {"r_full":>8s} {"rho_full":>9s} {"MAE":>8s} {"R2":>8s} {"n":>6s} {"sig":>5s}')
  for subject in sorted(subject_metrics):
    m = subject_metrics[subject]
    print(
      f'  {subject:10s} {m["pearson_full"]:8.4f} {m["spearman_full"]:9.4f} {m["mae"]:8.4f} '
      f'{m["r2"]:8.4f} {m["n"]:6d} {significance_stars(m["pearson_p"]):>5s}'
    )

  tag = f'route_a_embedding_{features}_{arch}'
  plot_per_subject_bar(subject_metrics, fig_dir, tag)

  out_path = out_dir / f'per_subject_metrics_{tag}.json'
  mean_r = float(np.mean([m['pearson'] for m in subject_metrics.values()]))
  with open(out_path, 'w') as handle:
    json.dump({
      'per_subject': subject_metrics,
      'mean_pearson': mean_r,
      'std_pearson': float(np.std([m['pearson'] for m in subject_metrics.values()])),
    }, handle, indent=2, default=str)
  print(f'\n  Metrics -> {out_path}')


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Per-subject evaluation of the embedding-conditioned complexity model.')
  add_common_arguments(parser)
  parser.add_argument('--out', type=Path, required=True, help='Output directory for metrics JSON and figures.')
  parser.add_argument(
    '--checkpoint-glob', required=True,
    help="Glob matching one condition's fold checkpoints, for example "
         "'<out>/checkpoints/route_a_embedding_all_small_noise0.00_fold*of5.pt'.",
  )
  parser.add_argument('--n-boot', type=int, default=2000)
  parser.add_argument('--device', default='cuda')
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  fig_dir = args.out / 'figures'
  fig_dir.mkdir(parents=True, exist_ok=True)

  dataset = build_dataset(args)
  evaluate_kfold_per_subject(args.nsd_root, dataset, args.checkpoint_glob, device, fig_dir, args.out, args.n_boot)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
