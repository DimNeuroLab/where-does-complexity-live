"""Original checkpoint evaluation and statistical calculations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import ConcatDataset, DataLoader

from route_b.config import paths as get_paths
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
  ComplexityDataset,
  complexity_split_by_image,
  complexity_stratified_group_kfold,
  load_complexity_records,
)
from route_b.models.complexity import BrainComplexityModel
from route_b.types import Array, JSON, Metrics, Record, Sample, TensorMap

matplotlib.use('Agg')
IDX_TO_CATEGORY = {v: k for (k, v) in CATEGORY_TO_IDX.items()}
MIN_N_FOR_CATEGORY = 10
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 5000


def _collate_fn(batch: list[Sample]) -> TensorMap:
  out = {}
  for key in batch[0]:
    vals = [b[key] for b in batch]
    if isinstance(vals[0], torch.Tensor):
      out[key] = torch.stack(vals)
    else:
      out[key] = torch.tensor(vals, dtype=torch.long)
  return out


def _build_eval_model(fmri_dim: int, device: str) -> BrainComplexityModel:
  """Instantiate a BrainComplexityModel for evaluation."""
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


def _run_inference(
  model: torch.nn.Module,
  loader: DataLoader,
  device: torch.device | str
) -> tuple[Array, Array, Array, Array, Array]:
  """Run inference and return (preds, log_vars, gts, cat_idx, subj_ids)."""
  (all_preds, all_log_vars, all_gts) = ([], [], [])
  (all_cat_idx, all_subjs) = ([], [])
  model.eval()
  with torch.no_grad():
    for batch in loader:
      fmri = batch['fmri'].to(device)
      subj_id = batch['subj_id'].to(device)
      category_idx = batch['category_idx'].to(device)
      clip_text = batch['clip_text'].to(device)
      (mu, log_var, _) = model(fmri, subj_id, category_idx, clip_text)
      all_preds.append(mu.cpu().numpy())
      all_log_vars.append(log_var.cpu().numpy())
      all_gts.append(batch['score'].numpy())
      all_cat_idx.append(batch['category_idx'].numpy())
      all_subjs.append(batch['subj_id'].numpy())
  return (
    np.concatenate(all_preds),
    np.concatenate(all_log_vars),
    np.concatenate(all_gts),
    np.concatenate(all_cat_idx),
    np.concatenate(all_subjs)
  )


def _build_val_loader(
  train_recs: list[Record],
  val_recs: list[Record],
  category_means: dict[str, float]
) -> tuple[DataLoader, int]:
  """Build val DataLoader for one fold, return (loader, fmri_dim)."""
  (train_datasets, val_datasets) = ([], [])
  for subj in NSD_SUBJECTS:
    tds = ComplexityDataset(subj, records=train_recs, category_means=category_means, visual_targets=False)
    vds = ComplexityDataset(
      subj,
      records=val_recs,
      fmri_mean=tds._fmri_mean,
      fmri_std=tds._fmri_std,
      category_means=category_means,
      visual_targets=False,
    )
    train_datasets.append(tds)
    val_datasets.append(vds)
  val_ds = ConcatDataset(val_datasets)
  loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=_collate_fn)
  fmri_dim = val_datasets[0].fmri_dim
  return (loader, fmri_dim)


def evaluate_complexity_model(device: str = 'cuda') -> tuple[Metrics, Array, Array, list[str], Array] | dict[str, JSON]:
  """Evaluate each saved fold on its validation records using the checkpoint category means."""
  ckpt_path = get_paths().checkpoints / 'complexity' / 'multisubj_best.pt'
  if not ckpt_path.exists():
    print(f'  No checkpoint found at {ckpt_path}')
    return {}
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  category_means = ckpt.get('category_means', {})
  cv_results = ckpt.get('cv_results', None)
  fold_state_dicts = ckpt.get('fold_state_dicts', None)
  if not category_means:
    print('  WARNING: No category_means in checkpoint, residual un-centering disabled')
  records = load_complexity_records()
  if cv_results is not None and fold_state_dicts is not None:
    n_folds = cv_results['n_folds']
    assert len(fold_state_dicts) == n_folds
    folds = complexity_stratified_group_kfold(records, n_folds, SEED)
    print(f'  Ensemble mode: out-of-fold predictions across {n_folds} folds')
    (all_preds, all_log_vars, all_gts) = ([], [], [])
    (all_cat_idx, all_subjs) = ([], [])
    fmri_dim = None
    for (fi, (train_recs, val_recs)) in enumerate(folds):
      (loader, fmri_dim) = _build_val_loader(train_recs, val_recs, category_means)
      model = _build_eval_model(fmri_dim, device)
      model.load_state_dict(fold_state_dicts[fi])
      (preds_f, lv_f, gts_f, ci_f, si_f) = _run_inference(model, loader, device)
      all_preds.append(preds_f)
      all_log_vars.append(lv_f)
      all_gts.append(gts_f)
      all_cat_idx.append(ci_f)
      all_subjs.append(si_f)
      r_f = pearsonr(preds_f, gts_f)[0] if len(preds_f) > 2 else 0.0
      print(f'    Fold {fi + 1}: val={len(val_recs)} → r_resid={r_f:.4f}')
    preds_resid = np.concatenate(all_preds)
    gts_resid = np.concatenate(all_gts)
    cat_indices = np.concatenate(all_cat_idx)
    subj_ids = np.concatenate(all_subjs)
    log_vars = np.concatenate(all_log_vars)
    print(f'  Total out-of-fold samples: {len(preds_resid)}')
  elif cv_results is not None:
    n_folds = cv_results['n_folds']
    if 'best_fold_idx' in cv_results:
      best_fold_idx = cv_results['best_fold_idx']
    else:
      fold_losses = cv_results['fold_val_losses']
      best_fold_idx = int(np.argmin(fold_losses))
      print(f'  WARNING: best_fold_idx missing, recomputed → fold {best_fold_idx + 1}')
    folds = complexity_stratified_group_kfold(records, n_folds, SEED)
    (train_recs, val_recs) = folds[best_fold_idx]
    print(f'  Best-fold mode: fold {best_fold_idx + 1}/{n_folds} (val={len(val_recs)} records)')
    (loader, fmri_dim) = _build_val_loader(train_recs, val_recs, category_means)
    model = _build_eval_model(fmri_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    (preds_resid, log_vars, gts_resid, cat_indices, subj_ids) = _run_inference(model, loader, device)
  else:
    (train_recs, val_recs) = complexity_split_by_image(records, 0.1, SEED)
    (loader, fmri_dim) = _build_val_loader(train_recs, val_recs, category_means)
    model = _build_eval_model(fmri_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    (preds_resid, log_vars, gts_resid, cat_indices, subj_ids) = _run_inference(model, loader, device)
  mean_sigma = float(np.mean(np.exp(0.5 * log_vars)))
  all_tasks = [IDX_TO_CATEGORY[int(ci)] for ci in cat_indices]
  cat_mean_vec = np.array([category_means.get(t, 0.0) for t in all_tasks])
  preds = preds_resid + cat_mean_vec
  gts = gts_resid + cat_mean_vec
  (r, r_p) = pearsonr(preds, gts)
  (rho, rho_p) = spearmanr(preds, gts)
  mae = mean_absolute_error(gts, preds)
  r2 = r2_score(gts, preds)
  r_ci = _bootstrap_ci(preds, gts, _pearson_fn)
  rho_ci = _bootstrap_ci(preds, gts, _spearman_fn)
  r_perm_p = _permutation_pvalue(preds, gts, _pearson_fn)
  (r_resid, r_resid_p) = pearsonr(preds_resid, gts_resid)
  (rho_resid, rho_resid_p) = spearmanr(preds_resid, gts_resid)
  (steiger_z, steiger_p) = _steiger_test(float(r), float(r_resid), float(np.corrcoef(preds, preds_resid)[0, 1]), len(preds))
  metrics = {
    'pearson': float(r),
    'pearson_p': float(r_p),
    'pearson_ci': r_ci,
    'pearson_perm_p': float(r_perm_p),
    'spearman': float(rho),
    'spearman_p': float(rho_p),
    'spearman_ci': rho_ci,
    'mae': float(mae),
    'r2': float(r2),
    'pearson_residual': float(r_resid),
    'pearson_residual_p': float(r_resid_p),
    'spearman_residual': float(rho_resid),
    'spearman_residual_p': float(rho_resid_p),
    'steiger_z_full_vs_resid': float(steiger_z),
    'steiger_p_full_vs_resid': float(steiger_p),
    'n_val': int(len(gts)),
    'mean_sigma': mean_sigma
  }
  subj_metrics = {}
  for (si, subj) in enumerate(NSD_SUBJECTS):
    mask = subj_ids == si
    if mask.sum() < 3:
      continue
    (sr, sr_p) = pearsonr(preds[mask], gts[mask])
    (srho, srho_p) = spearmanr(preds[mask], gts[mask])
    smae = mean_absolute_error(gts[mask], preds[mask])
    sr2 = r2_score(gts[mask], preds[mask])
    subj_metrics[subj] = {
      'pearson': float(sr),
      'pearson_p': float(sr_p),
      'spearman': float(srho),
      'spearman_p': float(srho_p),
      'mae': float(smae),
      'r2': float(sr2),
      'n': int(mask.sum())
    }
  metrics['per_subject'] = subj_metrics
  metrics['n_permutations'] = N_PERMUTATIONS
  if cv_results is not None:
    fold_losses = cv_results['fold_val_losses']
    _best_fold_idx = int(np.argmin(fold_losses))
    metrics['cv'] = {
      'n_folds': cv_results['n_folds'],
      'best_fold': _best_fold_idx + 1,
      'ensemble': fold_state_dicts is not None,
      'fold_val_losses': cv_results['fold_val_losses'],
      'fold_metrics': cv_results['fold_metrics'],
      'mean_val_loss': float(np.mean(cv_results['fold_val_losses'])),
      'std_val_loss': float(np.std(cv_results['fold_val_losses'])),
      'mean_pearson_resid': float(np.mean([m.get('pearson_resid', 0) for m in cv_results['fold_metrics']])),
      'std_pearson_resid': float(np.std([m.get('pearson_resid', 0) for m in cv_results['fold_metrics']])),
      'mean_spearman_resid': float(np.mean([m.get('spearman_resid', 0) for m in cv_results['fold_metrics']])),
      'std_spearman_resid': float(np.std([m.get('spearman_resid', 0) for m in cv_results['fold_metrics']])),
      'mean_mae': float(np.mean([m.get('mae', 0) for m in cv_results['fold_metrics']])),
      'std_mae': float(np.std([m.get('mae', 0) for m in cv_results['fold_metrics']]))
    }
  cat_metrics = {}
  raw_p_pearson = {}
  raw_p_spearman = {}
  for task in sorted(set(all_tasks)):
    mask = np.array([t == task for t in all_tasks])
    n = mask.sum()
    if n < MIN_N_FOR_CATEGORY:
      continue
    (cat_r, cat_r_p) = pearsonr(preds[mask], gts[mask])
    (cat_rho, cat_rho_p) = spearmanr(preds[mask], gts[mask])
    (cat_r_resid, cat_r_resid_p) = pearsonr(preds_resid[mask], gts_resid[mask])
    (cat_rho_resid, cat_rho_resid_p) = spearmanr(preds_resid[mask], gts_resid[mask])
    cat_mae = mean_absolute_error(gts[mask], preds[mask])
    cat_r_ci = _bootstrap_ci(preds[mask], gts[mask], _pearson_fn)
    raw_p_pearson[task] = cat_r_p
    raw_p_spearman[task] = cat_rho_p
    cat_metrics[task] = {
      'pearson': float(cat_r),
      'pearson_p': float(cat_r_p),
      'pearson_ci': cat_r_ci,
      'spearman': float(cat_rho),
      'spearman_p': float(cat_rho_p),
      'pearson_resid': float(cat_r_resid),
      'pearson_resid_p': float(cat_r_resid_p),
      'spearman_resid': float(cat_rho_resid),
      'spearman_resid_p': float(cat_rho_resid_p),
      'mae': float(cat_mae),
      'n': int(n)
    }
  if raw_p_pearson:
    fdr_r = _benjamini_hochberg(raw_p_pearson)
    fdr_rho = _benjamini_hochberg(raw_p_spearman)
    for task in cat_metrics:
      cat_metrics[task]['pearson_p_fdr'] = fdr_r.get(task, 1.0)
      cat_metrics[task]['spearman_p_fdr'] = fdr_rho.get(task, 1.0)
  metrics['category_wise'] = cat_metrics
  return (metrics, preds, gts, all_tasks, subj_ids)


def sig_stars(p: float) -> str:
  """Return significance stars: *** < .001, ** < .01, * < .05, n.s. otherwise."""
  if p < 0.001:
    return '***'
  if p < 0.01:
    return '**'
  if p < 0.05:
    return '*'
  return 'n.s.'


def _fmt_p(p: float) -> str:
  """Format p-value for display: scientific notation if very small."""
  if p < 1e-10:
    return f'{p:.2e}'
  if p < 0.001:
    return f'{p:.2e}'
  return f'{p:.4f}'


def _pearson_fn(p: Array, g: Array) -> float:
  return pearsonr(p, g)[0]


def _spearman_fn(p: Array, g: Array) -> float:
  return spearmanr(p, g)[0]


def _bootstrap_ci(
  preds: Array,
  gts: Array,
  stat_fn: Callable[[Array, Array], float],
  n_boot: int = N_BOOTSTRAP,
  alpha: float = 0.05
) -> tuple[float, float]:
  rng = np.random.RandomState(42)
  n = len(preds)
  stats = []
  for _ in range(n_boot):
    idx = rng.randint(0, n, n)
    try:
      stats.append(stat_fn(preds[idx], gts[idx]))
    except Exception:
      continue
  lo = float(np.percentile(stats, 100 * alpha / 2))
  hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
  return [lo, hi]


def _permutation_pvalue(
  preds: Array,
  gts: Array,
  stat_fn: Callable[[Array, Array], float],
  n_perm: int = N_PERMUTATIONS
) -> float:
  """Two-sided permutation p-value: fraction of permuted |stat| ≥ observed |stat|."""
  rng = np.random.RandomState(42)
  observed = abs(stat_fn(preds, gts))
  count = 0
  for _ in range(n_perm):
    gts_perm = rng.permutation(gts)
    try:
      if abs(stat_fn(preds, gts_perm)) >= observed:
        count += 1
    except Exception:
      continue
  return (count + 1) / (n_perm + 1)


def _steiger_test(r_xy: float, r_xz: float, r_yz: float, n: int) -> tuple[float, float]:
  """Test H0: r(X,Y) == r(X,Z) where Y,Z are correlated, using Steiger (1980)."""
  if n < 4:
    return (0.0, 1.0)

  def fisher(r: float) -> float:
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))
  r_det = 1 - r_xy ** 2 - r_xz ** 2 - r_yz ** 2 + 2 * r_xy * r_xz * r_yz
  r_mean_sq = 0.5 * (r_xy ** 2 + r_xz ** 2)
  denom = (1 - r_yz) ** 3 if 1 - r_yz > 1e-10 else 1e-10
  f_factor = (1 - r_yz) / (2 * (1 - r_mean_sq) + 1e-10)
  z = (fisher(r_xy) - fisher(r_xz)) * np.sqrt((n - 3) / (2 * (1 - r_yz) + 1e-10)) * np.sqrt(f_factor + 1e-10)
  p = 2 * (1 - norm.cdf(abs(z)))
  return (float(z), float(p))


def _benjamini_hochberg(p_dict: dict) -> dict[str, float]:
  """Apply BH FDR correction to a {label: p_value} dict. Returns {label: q_value}."""
  if not p_dict:
    return {}
  labels = sorted(p_dict.keys())
  pvals = np.array([p_dict[k] for k in labels])
  m = len(pvals)
  sorted_idx = np.argsort(pvals)
  sorted_pvals = pvals[sorted_idx]
  adjusted = np.empty(m)
  adjusted[sorted_idx[-1]] = sorted_pvals[-1]
  for i in range(m - 2, -1, -1):
    si = sorted_idx[i]
    adjusted[si] = min(adjusted[sorted_idx[i + 1]], sorted_pvals[i] * m / (i + 1))
  adjusted = np.clip(adjusted, 0, 1)
  return {labels[i]: float(adjusted[i]) for i in range(m)}


def plot_calibration(preds: Array, gts: Array, tasks: list[str], save_dir: Path | None = None) -> None:
  """Predicted vs GT scatter, colour by category, bubble size ~ n."""
  if save_dir is None:
    save_dir = get_paths().output / 'figures'
  save_dir.mkdir(parents=True, exist_ok=True)
  unique_tasks = sorted(set(tasks))
  cmap = plt.cm.get_cmap('tab20', len(unique_tasks))
  task_to_color = {t: cmap(i) for (i, t) in enumerate(unique_tasks)}
  (fig, ax) = plt.subplots(figsize=(8, 8))
  for task in unique_tasks:
    mask = np.array([t == task for t in tasks])
    ax.scatter(gts[mask], preds[mask], c=[task_to_color[task]], label=f'{task} (n={mask.sum()})', alpha=0.4, s=8)
  mn = min(gts.min(), preds.min())
  mx = max(gts.max(), preds.max())
  ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5)
  ax.set_xlabel('GT complexity score')
  ax.set_ylabel('Predicted complexity score')
  r = pearsonr(preds, gts)[0]
  ax.set_title(f'Pooled multi-subject - r={r:.3f}')
  ax.legend(fontsize=6, ncol=2, loc='upper left')
  fig.tight_layout()
  fig.savefig(save_dir / 'multisubj_calibration.png', dpi=150)
  fig.savefig(save_dir / 'multisubj_calibration.pdf', dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f"  Saved → {save_dir / 'multisubj_calibration.png'} (+pdf)")


def plot_per_category_calibration(preds: Array, gts: Array, tasks: list[str], save_dir: Path | None = None) -> None:
  """One scatter plot per category: predicted vs GT (un-residualized)."""
  if save_dir is None:
    save_dir = get_paths().output / 'figures' / 'per_category'
  else:
    save_dir = save_dir / 'per_category'
  save_dir.mkdir(parents=True, exist_ok=True)
  unique_tasks = sorted(set(tasks))
  tasks_arr = np.array(tasks)
  for task in unique_tasks:
    mask = tasks_arr == task
    n = mask.sum()
    if n < MIN_N_FOR_CATEGORY:
      continue
    (p, g) = (preds[mask], gts[mask])
    (r_val, r_p) = pearsonr(p, g)
    (rho_val, _) = spearmanr(p, g)
    (fig, ax) = plt.subplots(figsize=(6, 6))
    ax.scatter(g, p, alpha=0.35, s=14, edgecolors='none', c='steelblue')
    mn = min(g.min(), p.min()) - 0.05
    mx = max(g.max(), p.max()) + 0.05
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.4, linewidth=1)
    if n > 2:
      z = np.polyfit(g, p, 1)
      xs = np.linspace(mn, mx, 100)
      ax.plot(xs, np.polyval(z, xs), 'r-', alpha=0.6, linewidth=1.5, label=f'fit (slope={z[0]:.2f})')
    ax.set_xlabel('GT complexity score')
    ax.set_ylabel('Predicted complexity score')
    pstr = f'p={r_p:.2e}' if r_p < 0.001 else f'p={r_p:.4f}'
    ax.set_title(f'{task}  (n={n})\nr={r_val:.3f} {pstr} | ρ={rho_val:.3f}')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect('equal')
    fig.tight_layout()
    safe_name = task.replace(' ', '_')
    fig.savefig(save_dir / f'calibration_{safe_name}.png', dpi=150)
    fig.savefig(save_dir / f'calibration_{safe_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
  print(f'  Saved {len(unique_tasks)} per-category calibration plots → {save_dir} (+pdf)')


def plot_category_bars(cat_metrics: dict[str, Metrics], save_dir: Path | None = None) -> None:
  """Horizontal bar chart of per-category Pearson r, sorted best-to-worst."""
  if save_dir is None:
    save_dir = get_paths().output / 'figures'
  save_dir.mkdir(parents=True, exist_ok=True)
  cats = sorted(cat_metrics.keys(), key=lambda c: cat_metrics[c]['pearson'], reverse=True)
  rs = [cat_metrics[c]['pearson'] for c in cats]
  ns = [cat_metrics[c]['n'] for c in cats]
  sig = ['*' if cat_metrics[c].get('pearson_p_fdr', 1.0) < 0.05 else '' for c in cats]
  (fig, ax) = plt.subplots(figsize=(12, max(5, 0.45 * len(cats))))
  y_pos = np.arange(len(cats))
  colors = ['#2a9d8f' if r > 0 else '#e76f51' for r in rs]
  bars = ax.barh(y_pos, rs, color=colors, edgecolor='black', linewidth=0.5, height=0.7)
  for (i, (r, s, n)) in enumerate(zip(rs, sig, ns)):
    offset = 0.01 if r >= 0 else -0.01
    ha = 'left' if r >= 0 else 'right'
    ax.text(r + offset, i, f'{r:.3f}{s}  (n={n})', va='center', ha=ha, fontsize=8)
  ax.margins(x=0.2)
  ax.set_yticks(y_pos)
  ax.set_yticklabels(cats)
  ax.invert_yaxis()
  ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
  ax.set_xlabel('Pearson r')
  ax.set_title(f'Per-category Pearson r  (* = FDR q<0.05, min n≥{MIN_N_FOR_CATEGORY})')
  fig.tight_layout()
  fig.savefig(save_dir / 'category_bars.png', dpi=150)
  fig.savefig(save_dir / 'category_bars.pdf', dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f"  Saved → {save_dir / 'category_bars.png'} (+pdf)")


def plot_per_subject(subj_metrics: dict[str, Metrics], save_dir: Path | None = None) -> None:
  """Bar chart of per-subject Pearson r."""
  if save_dir is None:
    save_dir = get_paths().output / 'figures'
  save_dir.mkdir(parents=True, exist_ok=True)
  subjs = sorted(subj_metrics.keys())
  rs = [subj_metrics[s]['pearson'] for s in subjs]
  (fig, ax) = plt.subplots(figsize=(8, 5))
  bars = ax.bar(subjs, rs, color='steelblue', edgecolor='black')
  for (bar, r) in zip(bars, rs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
  ax.set_ylabel('Pearson r')
  ax.set_title('Per-subject complexity prediction (pooled model)')
  ax.axhline(0, color='gray', linewidth=0.5)
  ymax = max(rs) if rs else 0.5
  ax.set_ylim(top=ymax * 1.15)
  fig.tight_layout()
  fig.savefig(save_dir / 'per_subject_performance.png', dpi=150)
  fig.savefig(save_dir / 'per_subject_performance.pdf', dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f"  Saved → {save_dir / 'per_subject_performance.png'} (+pdf)")
