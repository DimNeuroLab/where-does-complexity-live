"""Subject-averaged image-target prediction variance."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from route_b.config import configure, paths as get_paths
from route_b.constants import (
  CATEGORY_EMBED_DIM,
  CLIP_EMBED_DIM,
  DINO_EMBED_DIM,
  FMRI_ENCODER_DROPOUT,
  FMRI_ENCODER_HIDDEN,
  FMRI_ENCODER_LATENT,
  NSD_SUBJECTS,
  NUM_CATEGORIES,
  NUM_SUBJECTS,
  SEED,
  SUBJECT_EMBED_DIM,
)
from route_b.data.datasets import (
  CATEGORY_TO_IDX,
  complexity_stratified_group_kfold,
  get_clip_text_embeddings,
  get_nsd_id_to_index,
  load_complexity_records,
)
from route_b.models.complexity import BrainComplexityModel
from route_b.types import Record


def sample_variance(values: list[float], mean: float) -> float:
  if len(values) < 2:
    raise ValueError('At least two target scores are required for variance')
  return sum(((value - mean) ** 2 for value in values)) / (len(values) - 1)


def decompose_variance(scores_by_image: dict[str, dict[str, float]], min_pairs: int) -> tuple[float, float, float, int]:
  """Return total, within, between variance and retained image count."""
  groups = [list(scores.values()) for scores in scores_by_image.values() if len(scores) >= min_pairs]
  if not groups:
    raise ValueError(f'No images have K_i >= {min_pairs} target-score pairs')
  all_scores = [score for group in groups for score in group]
  overall_mean = sum(all_scores) / len(all_scores)
  group_means = [sum(group) / len(group) for group in groups]
  denominator = len(all_scores) - 1
  total = sample_variance(all_scores, overall_mean)
  within = sum(
    sum((score - group_mean) ** 2 for score in group) for group, group_mean in zip(groups, group_means)
  ) / denominator
  between = sum(
    len(group) * (group_mean - overall_mean) ** 2 for group, group_mean in zip(groups, group_means)
  ) / denominator
  return (total, within, between, len(groups))


def build_model(fmri_dim: int, device: torch.device) -> BrainComplexityModel:
  return BrainComplexityModel(
    fmri_dim=fmri_dim,
    dino_dim=DINO_EMBED_DIM,
    clip_dim=CLIP_EMBED_DIM,
    clip_text_dim=CLIP_EMBED_DIM,
    n_categories=NUM_CATEGORIES,
    cat_embed_dim=CATEGORY_EMBED_DIM,
    n_subjects=NUM_SUBJECTS,
    subject_embed_dim=SUBJECT_EMBED_DIM,
    encoder_hidden=FMRI_ENCODER_HIDDEN,
    encoder_latent=FMRI_ENCODER_LATENT,
    encoder_dropout=FMRI_ENCODER_DROPOUT
  ).to(device)


def load_normalized_fmri() -> dict[str, np.ndarray]:
  """Load PCA fMRI using the normalization used by ComplexityDataset."""
  normalized = {}
  for subject in NSD_SUBJECTS:
    fmri = np.load(get_paths().pca / f'{subject}_pca_fmri.npy')
    normalized[subject] = (fmri - fmri.mean(axis=0)) / (fmri.std(axis=0) + 1e-06)
  return normalized


def unique_scene_target_records(records: list[Record]) -> list[Record]:
  """Keep one record per physical NSD scene and target task."""
  unique_records: dict[tuple[int, str], Record] = {}
  for record in records:
    unique_records.setdefault((record['nsd_id'], record['task']), record)
  return list(unique_records.values())


def predict_oof_subject_pair_scores(
  checkpoint_path: Path,
  device: torch.device,
  batch_size: int
) -> dict[str, dict[tuple[str, str], float]]:
  """Return full-scale OOF predictions by subject, physical scene, and task."""
  checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  fold_states = checkpoint.get('fold_state_dicts')
  cv_results = checkpoint.get('cv_results')
  if not fold_states or not cv_results:
    raise ValueError((
      'The checkpoint has no saved cross-validation fold states; out-of-fold predi'
      'ctions cannot be reproduced.'
    ))
  n_folds = cv_results['n_folds']
  if len(fold_states) != n_folds:
    raise ValueError(f'Checkpoint has {len(fold_states)} fold states but declares {n_folds} folds')
  category_means = checkpoint.get('category_means', {})
  missing_means = set(CATEGORY_TO_IDX) - set(category_means)
  if missing_means:
    raise ValueError(f"Checkpoint is missing category means for: {', '.join(sorted(missing_means))}")
  records = load_complexity_records(train_only=True)
  folds = complexity_stratified_group_kfold(records, n_folds, SEED)
  fmri_by_subject = load_normalized_fmri()
  row_by_subject = {subject: get_nsd_id_to_index(subject) for subject in NSD_SUBJECTS}
  clip_text_by_task = get_clip_text_embeddings()
  subject_pair_predictions: dict[str, dict[tuple[str, str], list[float]]] = {
    subject: defaultdict(list) for subject in NSD_SUBJECTS
  }
  for (fold_index, (_, val_records)) in enumerate(folds):
    val_records = unique_scene_target_records(val_records)
    model = build_model(fmri_by_subject[NSD_SUBJECTS[0]].shape[1], device)
    model.load_state_dict(fold_states[fold_index])
    model.eval()
    for (subject_index, subject) in enumerate(NSD_SUBJECTS):
      rows_and_records = [
        (row_by_subject[subject][record['nsd_id']], record) for record in val_records
        if record['nsd_id'] in row_by_subject[subject] and record['task'] in CATEGORY_TO_IDX
      ]
      if not rows_and_records:
        continue
      for start in range(0, len(rows_and_records), batch_size):
        batch = rows_and_records[start:start + batch_size]
        rows = [row for (row, _) in batch]
        tasks = [record['task'] for (_, record) in batch]
        fmri = torch.from_numpy(fmri_by_subject[subject][rows]).to(device)
        subject_ids = torch.full((len(batch),), subject_index, dtype=torch.long, device=device)
        category_ids = torch.tensor([CATEGORY_TO_IDX[task] for task in tasks], dtype=torch.long, device=device)
        clip_text = torch.from_numpy(np.stack([clip_text_by_task[task] for task in tasks]).astype(np.float32)).to(device)
        with torch.no_grad():
          (predicted_residual, _, _) = model(fmri, subject_ids, category_ids, clip_text)
        predicted_scores = predicted_residual.cpu().numpy() + np.array([category_means[task] for task in tasks])
        for ((_, record), score) in zip(batch, predicted_scores):
          subject_pair_predictions[subject][str(record['nsd_id']), record['task']].append(float(score))
  if not any(subject_pair_predictions.values()):
    raise ValueError('No predictions were produced from the checkpoint')
  return {
    subject: {pair: float(np.mean(predictions)) for pair, predictions in pair_predictions.items()}
    for subject, pair_predictions in subject_pair_predictions.items()
  }


def aggregate_subject_predictions(subject_pair_scores: dict[str, dict[tuple[str, str], float]]) -> dict[str, dict[str, float]]:
  """Average available subject predictions for every physical scene and target."""
  pair_predictions: dict[tuple[str, str], list[float]] = defaultdict(list)
  for pair_scores in subject_pair_scores.values():
    for (pair, score) in pair_scores.items():
      pair_predictions[pair].append(score)
  scores_by_image: dict[str, dict[str, float]] = defaultdict(dict)
  for ((image, task), predictions) in pair_predictions.items():
    scores_by_image[image][task] = float(np.mean(predictions))
  return scores_by_image


def predict_oof_pair_scores(checkpoint_path: Path, device: torch.device, batch_size: int) -> dict[str, dict[str, float]]:
  """Return subject-averaged, full-scale OOF predictions by scene and task."""
  return aggregate_subject_predictions(predict_oof_subject_pair_scores(checkpoint_path, device, batch_size))


def mean_ground_truth_by_scene_target() -> dict[tuple[str, str], float]:
  """Return mean train-set ground truth for each physical scene and target."""
  scores: dict[tuple[str, str], list[float]] = defaultdict(list)
  for record in load_complexity_records(train_only=True):
    scores[str(record['nsd_id']), record['task']].append(record['score'])
  return {pair: float(np.mean(values)) for (pair, values) in scores.items()}


def multitarget_subject_correlations(
  subject_pair_scores: dict[str, dict[tuple[str, str], float]],
  scores_by_image: dict[str, dict[str, float]],
  min_pairs: int
) -> tuple[dict[str, dict[str, float]], float, float]:
  """Calculate correlations on the multi-target scenes used in the variance."""
  from scipy.stats import pearsonr, spearmanr
  eligible_scenes = {scene for (scene, target_scores) in scores_by_image.items() if len(target_scores) >= min_pairs}
  ground_truth = mean_ground_truth_by_scene_target()
  metrics: dict[str, dict[str, float]] = {}
  (pooled_predictions, pooled_targets) = ([], [])
  for (subject, pair_scores) in subject_pair_scores.items():
    (predictions, targets) = ([], [])
    for (pair, prediction) in pair_scores.items():
      if pair[0] in eligible_scenes and pair in ground_truth:
        predictions.append(prediction)
        targets.append(ground_truth[pair])
    if len(predictions) < 3:
      continue
    metrics[subject] = {
      'pearson': float(pearsonr(predictions, targets).statistic),
      'spearman': float(spearmanr(predictions, targets).statistic),
      'n': len(predictions)
    }
    pooled_predictions.extend(predictions)
    pooled_targets.extend(targets)
  if len(pooled_predictions) < 3:
    raise ValueError('Fewer than three predictions are available for correlation')
  return (
    metrics,
    float(pearsonr(pooled_predictions, pooled_targets).statistic),
    float(spearmanr(pooled_predictions, pooled_targets).statistic)
  )


def plot_multitarget_subject_correlations(
  metrics: dict[str, dict[str, float]],
  pooled_pearson: float,
  pooled_spearman: float
) -> Path:
  """Create a per-subject-comparison-style chart for the multi-target subset."""
  import os
  os.environ.setdefault('MPLCONFIGDIR', str(get_paths().output / '.matplotlib'))
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  save_dir = get_paths().output / 'figures'
  save_dir.mkdir(parents=True, exist_ok=True)
  subjects = sorted(metrics)
  pearsons = [metrics[subject]['pearson'] for subject in subjects]
  spearmans = [metrics[subject]['spearman'] for subject in subjects]
  sample_sizes = [metrics[subject]['n'] for subject in subjects]
  x = np.arange(len(subjects))
  width = 0.35
  (fig, axis) = plt.subplots(figsize=(10, 5.5))
  pearson_bars = axis.bar(x - width / 2, pearsons, width, label='Pearson r', color='#4472C4', edgecolor='black', linewidth=0.5)
  spearman_bars = axis.bar(
    x + width / 2,
    spearmans,
    width,
    label='Spearman ρ',
    color='#ED7D31',
    edgecolor='black',
    linewidth=0.5
  )
  for (bars, values) in ((pearson_bars, pearsons), (spearman_bars, spearmans)):
    for (bar, value) in zip(bars, values):
      axis.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f'{value:.3f}',
        ha='center',
        va='bottom',
        fontsize=8
      )
  for (index, sample_size) in enumerate(sample_sizes):
    axis.text(x[index], -0.05, f'n={sample_size}', ha='center', va='top', fontsize=7, color='#666666')
  axis.set_ylabel('Correlation')
  axis.set_title('Per-Subject Prediction Performance\n(Multi-Target Physical NSD Scenes)')
  axis.set_xticks(x)
  axis.set_xticklabels(subjects)
  axis.set_ylim(0, max(pearsons + spearmans) * 1.2)
  axis.grid(axis='y', alpha=0.3)
  axis.axhline(
    pooled_pearson,
    color='#4472C4',
    linestyle='--',
    alpha=0.6,
    linewidth=1.2,
    label=f'Pooled Pearson r = {pooled_pearson:.3f}'
  )
  axis.axhline(
    pooled_spearman,
    color='#ED7D31',
    linestyle='--',
    alpha=0.6,
    linewidth=1.2,
    label=f'Pooled Spearman ρ = {pooled_spearman:.3f}'
  )
  axis.legend(loc='upper right', fontsize=8)
  fig.tight_layout()
  output_path = save_dir / 'multitarget_per_subject_comparison.pdf'
  fig.savefig(output_path, dpi=300, bbox_inches='tight')
  fig.savefig(output_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
  plt.close(fig)
  return output_path


def main() -> None:
  parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--checkpoint-file', dest='checkpoint', type=Path)
  parser.add_argument('--min-pairs', type=int, default=2, metavar='T')
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument(
    '--plot-subject-correlations',
    action='store_true',
    help='write the per-subject correlation plot for this multi-target subset'
  )
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=('cpu', 'cuda'))
  args = parser.parse_args()
  configure(args.config, ('variance',))
  if args.checkpoint is None:
    args.checkpoint = get_paths().checkpoints / 'complexity/visual_1.pt'
  if args.min_pairs < 2:
    parser.error('--min-pairs must be at least 2')
  if args.batch_size < 1:
    parser.error('--batch-size must be positive')
  if not args.checkpoint.is_file():
    parser.error(f'Checkpoint does not exist: {args.checkpoint}')
  if args.device == 'cuda' and (not torch.cuda.is_available()):
    parser.error('--device cuda was requested but CUDA is unavailable')
  if args.plot_subject_correlations:
    subject_pair_scores = predict_oof_subject_pair_scores(args.checkpoint, torch.device(args.device), args.batch_size)
    scores_by_image = aggregate_subject_predictions(subject_pair_scores)
  else:
    subject_pair_scores = None
    scores_by_image = predict_oof_pair_scores(args.checkpoint, torch.device(args.device), args.batch_size)
  (total, within, between, image_count) = decompose_variance(scores_by_image, args.min_pairs)
  within_percent = 0.0 if total == 0 else 100 * within / total
  between_percent = 0.0 if total == 0 else 100 * between / total
  print('VARIANCE DECOMPOSITION')
  print(f'Total variance:          {total:.6f}')
  print(f'Within-image variance:   {within:.6f}  ({within_percent:.2f}%)')
  print(f'Between-image variance:  {between:.6f}  ({between_percent:.2f}%)')
  print()
  print(f'...computed on {image_count:,} images with K_i >= {args.min_pairs} pairs')
  if subject_pair_scores is not None:
    metrics, pooled_pearson, pooled_spearman = multitarget_subject_correlations(
      subject_pair_scores, scores_by_image, args.min_pairs,
    )
    output_path = plot_multitarget_subject_correlations(metrics, pooled_pearson, pooled_spearman)
    print()
    print('MULTI-TARGET SCENE CORRELATIONS (OUT-OF-FOLD)')
    print(f'Pooled Pearson r:        {pooled_pearson:.4f}')
    print(f'Pooled Spearman ρ:       {pooled_spearman:.4f}')
    print(f"{'Subject':<10} {'Pearson r':>10} {'Spearman ρ':>11} {'n':>6}")
    for subject in sorted(metrics):
      metric = metrics[subject]
      print(f"{subject:<10} {metric['pearson']:10.4f} {metric['spearman']:11.4f} {metric['n']:6d}")
    print(f'Saved → {output_path} (+png)')
if __name__ == '__main__':
  main()
