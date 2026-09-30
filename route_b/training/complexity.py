"""Phase 3 training: target-conditioned complexity prediction (multi-subject)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from route_b.config import paths as get_paths
from route_b.constants import (
  BATCH_SIZE,
  CATEGORY_EMBED_DIM,
  CLIP_EMBED_DIM,
  DINO_EMBED_DIM,
  FMRI_ENCODER_DROPOUT,
  FMRI_ENCODER_HIDDEN,
  FMRI_ENCODER_LATENT,
  LEARNING_RATE,
  NSD_SUBJECTS,
  NUM_CATEGORIES,
  NUM_EPOCHS_COMPLEXITY,
  NUM_SUBJECTS,
  NUM_WORKERS,
  N_CV_FOLDS,
  PATIENCE,
  SEED,
  SUBJECT_EMBED_DIM,
  WEIGHT_DECAY,
)
from route_b.data.datasets import (
  ComplexityDataset,
  complexity_split_by_image,
  complexity_stratified_group_kfold,
  load_complexity_records,
)
from route_b.models.complexity import BrainComplexityModel, HeteroscedasticLoss
from route_b.runtime import EpochRecovery, atomic_checkpoint, check_budget
from route_b.types import FoldResult, Sample, TensorMap


def set_seed(seed: int) -> None:
  import random
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _collate_fn(batch: list[Sample]) -> TensorMap:
  out = {}
  for key in batch[0]:
    vals = [b[key] for b in batch]
    if isinstance(vals[0], torch.Tensor):
      out[key] = torch.stack(vals)
    else:
      out[key] = torch.tensor(vals, dtype=torch.long)
  return out


def _compute_balanced_weights(train_datasets: list[ComplexityDataset]) -> torch.Tensor:
  """Inverse-frequency weights over (subject × category) cells."""
  cell_counts: dict[tuple[int, int], int] = defaultdict(int)
  sample_cells: list[tuple[int, int]] = []
  for ds in train_datasets:
    for (_, cat_idx, _, _) in ds.samples:
      cell = (ds.subj_idx, cat_idx)
      cell_counts[cell] += 1
      sample_cells.append(cell)
  weights = [1.0 / cell_counts[cell] for cell in sample_cells]
  return torch.DoubleTensor(weights)


def _build_fold_loaders(
  train_recs: list[Record],
  val_recs: list[Record],
  batch_size: int,
  category_means: dict[str, float] | None = None
) -> tuple[DataLoader, DataLoader, int, dict[str, float], list[ComplexityDataset], list[ComplexityDataset]]:
  """Build train/val DataLoaders for one fold with balanced sampling."""
  (train_datasets, val_datasets) = ([], [])
  computed_category_means = category_means
  for subj in NSD_SUBJECTS:
    tds = ComplexityDataset(subj, records=train_recs, category_means=computed_category_means)
    if computed_category_means is None:
      computed_category_means = tds.category_means
    train_datasets.append(tds)
    vds = ComplexityDataset(
      subj,
      records=val_recs,
      fmri_mean=tds._fmri_mean,
      fmri_std=tds._fmri_std,
      category_means=computed_category_means
    )
    val_datasets.append(vds)
  train_ds = ConcatDataset(train_datasets)
  val_ds = ConcatDataset(val_datasets)
  sample_weights = _compute_balanced_weights(train_datasets)
  sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
  train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    collate_fn=_collate_fn
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=_collate_fn
  )
  fmri_dim = train_datasets[0].fmri_dim
  return (train_loader, val_loader, fmri_dim, computed_category_means, train_datasets, val_datasets)


def _build_model(
  fmri_dim: int,
  device: torch.device | str,
  pretrained_path: Path | None,
  gpus: list[int] | None
) -> torch.nn.Module:
  """Instantiate BrainComplexityModel and optionally load pretrained encoder."""
  model = BrainComplexityModel(
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
  if pretrained_path and pretrained_path.exists():
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_pretrained_encoder(ckpt['model_state_dict'])
  if gpus and len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus)
  return model


def _train_one_fold(
  train_loader: DataLoader,
  val_loader: DataLoader,
  model: torch.nn.Module,
  device: torch.device | str,
  epochs: int,
  lr: float,
  patience: int,
  freeze_encoder_epochs: int,
  aux_weight: float,
  fold_idx: int = 0,
  n_folds: int = 1,
  recovery: EpochRecovery | None = None
) -> FoldResult:
  """Run the training loop for a single fold."""
  raw = model.module if isinstance(model, nn.DataParallel) else model
  complexity_loss_fn = HeteroscedasticLoss()
  cos_sim = nn.CosineSimilarity(dim=-1)
  mse_fn = nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  best_val = float('inf')
  best_state = None
  best_metrics = {}
  best_epoch = 0
  no_improve = 0
  fold_label = f'[Fold {fold_idx + 1}/{n_folds}] ' if n_folds > 1 else ''
  resumed = recovery.restore(raw, optimizer, scheduler) if recovery else None
  start = resumed['epoch'] + 1 if resumed else 1
  if resumed:
    progress = resumed['progress']
    best_val, best_state = progress['best_val'], progress['best_state']
    best_metrics, best_epoch = progress['best_metrics'], progress['best_epoch']
    no_improve = progress['no_improve']
  for epoch in range(start, epochs + 1):
    if no_improve >= patience:
      break
    check_budget()
    for p in raw.encoder.parameters():
      p.requires_grad = epoch > freeze_encoder_epochs
    model.train()
    train_losses = []
    for batch in train_loader:
      check_budget()
      fmri = batch['fmri'].to(device)
      subj_id = batch['subj_id'].to(device)
      category_idx = batch['category_idx'].to(device)
      clip_text = batch['clip_text'].to(device)
      score = batch['score'].to(device)
      out = raw.forward_with_auxiliary(fmri, subj_id, category_idx, clip_text)
      loss_c = complexity_loss_fn(out['mu'], out['log_var'], score)
      loss_aux = torch.tensor(0.0, device=device)
      for key in ['dino_early', 'dino_mid', 'dino_late', 'clip']:
        gt_key = key if key != 'clip' else 'clip_img'
        gt = batch[gt_key].to(device) if gt_key in batch else batch[key].to(device)
        loss_aux = loss_aux + (1 - cos_sim(out[key], gt)).mean() + mse_fn(out[key], gt)
      loss = loss_c + aux_weight * loss_aux
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      train_losses.append(loss.item())
    scheduler.step()
    model.eval()
    (val_preds, val_gts, val_losses) = ([], [], [])
    with torch.no_grad():
      for batch in val_loader:
        check_budget()
        fmri = batch['fmri'].to(device)
        subj_id = batch['subj_id'].to(device)
        category_idx = batch['category_idx'].to(device)
        clip_text = batch['clip_text'].to(device)
        score = batch['score'].to(device)
        (mu, log_var, _) = raw(fmri, subj_id, category_idx, clip_text)
        loss = complexity_loss_fn(mu, log_var, score)
        val_losses.append(loss.item())
        val_preds.append(mu.cpu().numpy())
        val_gts.append(score.cpu().numpy())
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    all_preds = np.concatenate(val_preds)
    all_gts = np.concatenate(val_gts)
    if len(all_preds) > 2:
      r = pearsonr(all_preds, all_gts)[0]
      rho = spearmanr(all_preds, all_gts)[0]
    else:
      (r, rho) = (0.0, 0.0)
    mae = np.mean(np.abs(all_preds - all_gts))
    if epoch % 5 == 0 or epoch == 1:
      print((
        '  '
        f'{fold_label}'
        'Epoch '
        f'{epoch:3d}'
        '/'
        f'{epochs}'
        ' │ train '
        f'{avg_train:.4f}'
        ' │ val '
        f'{avg_val:.4f}'
        ' │ r='
        f'{r:.3f}'
        ' │ ρ='
        f'{rho:.3f}'
        ' │ MAE='
        f'{mae:.4f}'
      ))
    if avg_val < best_val:
      best_val = avg_val
      best_epoch = epoch
      best_metrics = {'pearson_resid': r, 'spearman_resid': rho, 'mae': mae}
      best_state = {k: v.cpu().clone() for (k, v) in raw.state_dict().items()}
      no_improve = 0
    else:
      no_improve += 1
    if recovery:
      recovery.save(epoch, raw, optimizer, scheduler, {
        'best_val': best_val, 'best_state': best_state, 'best_metrics': best_metrics,
        'best_epoch': best_epoch, 'no_improve': no_improve,
      })
    if no_improve >= patience:
      break
  print(f'  {fold_label}Best val loss: {best_val:.4f} (epoch {best_epoch})')
  return {'best_val_loss': best_val, 'best_epoch': best_epoch, 'metrics': best_metrics, 'state_dict': best_state}


def train_complexity_model(
  pretrained_path: Path | None = None,
  epochs: int = NUM_EPOCHS_COMPLEXITY,
  batch_size: int = BATCH_SIZE,
  lr: float = LEARNING_RATE,
  patience: int = PATIENCE,
  freeze_encoder_epochs: int = 5,
  aux_weight: float = 0.1,
  run_name: str | None = None,
  gpus: list[int] | None = None,
  n_folds: int = N_CV_FOLDS,
  recovery_root: Path | None = None
) -> float:
  """Train multi-subject complexity model with stratified K-fold CV."""
  set_seed(SEED)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f"\n{'=' * 60}")
  if n_folds > 1:
    print(f'  Phase 3 · Complexity Model ({n_folds}-fold stratified CV)')
  else:
    print(f'  Phase 3 · Complexity Model (single split)')
  print(f"{'=' * 60}")
  records = load_complexity_records()
  print(f'  Total records: {len(records)}')
  cat_dist = Counter((r['task'] for r in records))
  print(f'  Category distribution: { {k: v for (k, v) in sorted(cat_dist.items())}}')
  if n_folds > 1:
    folds = complexity_stratified_group_kfold(records, n_folds, SEED)
  else:
    folds = [complexity_split_by_image(records, 0.1, SEED)]
  save_dir = get_paths().checkpoints / 'complexity'
  save_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_stem = _checkpoint_stem(pretrained_path, run_name)
  save_path = save_dir / f'{checkpoint_stem}.pt'
  print(f"  Pretraining: {(pretrained_path if pretrained_path is not None else 'scratch')}")
  print(f'  Freeze encoder epochs: {freeze_encoder_epochs}')
  print(f'  Auxiliary weight: {aux_weight}')
  print(f'  Checkpoint stem: {checkpoint_stem}')
  all_fold_results = []
  best_overall_val = float('inf')
  best_overall_state = None
  best_overall_metrics = None
  best_category_means = None
  best_fold_idx = -1
  for (fold_idx, (train_recs, val_recs)) in enumerate(folds):
    directory = recovery_root / checkpoint_stem / f'fold_{fold_idx + 1}' if recovery_root else None
    complete = directory / 'complete.pt' if directory else None
    if complete is not None and complete.exists():
      saved = torch.load(complete, map_location='cpu', weights_only=False)
      fold_result, category_means = saved['fold_result'], saved['category_means']
    else:
      set_seed(SEED + fold_idx)
      if n_folds > 1:
        train_cats = Counter((r['task'] for r in train_recs))
        val_cats = Counter((r['task'] for r in val_recs))
        print(f"\n{'─' * 60}")
        print(f'  Fold {fold_idx + 1}/{n_folds} │ Train: {len(train_recs)} │ Val: {len(val_recs)}')
        print(f'  Val categories: { {k: v for (k, v) in sorted(val_cats.items())}}')
        print(f"{'─' * 60}")
      else:
        print(f'  Train recs: {len(train_recs)}, Val recs: {len(val_recs)}')
      (
        train_loader,
        val_loader,
        fmri_dim,
        category_means,
        train_dsets,
        val_dsets
      ) = _build_fold_loaders(train_recs, val_recs, batch_size)
      pool_train = sum((len(d) for d in train_dsets))
      pool_val = sum((len(d) for d in val_dsets))
      if fold_idx == 0:
        print(f'  PCA fMRI dim: {fmri_dim}')
        print(f"  Category means: { {k: f'{v:.4f}' for (k, v) in category_means.items()}}")
      print(f'  Pooled train: {pool_train},  Pooled val: {pool_val}')
      model = _build_model(fmri_dim, device, pretrained_path, gpus)
      recovery = EpochRecovery(directory, {'fold': fold_idx, 'epochs': epochs, 'lr': lr,
        'patience': patience, 'freeze_encoder_epochs': freeze_encoder_epochs, 'aux_weight': aux_weight,
        'batch_size': batch_size, 'pretrained_path': str(pretrained_path)}) if directory else None
      fold_result = _train_one_fold(
        train_loader,
        val_loader,
        model,
        device,
        epochs,
        lr,
        patience,
        freeze_encoder_epochs,
        aux_weight,
        fold_idx=fold_idx,
        n_folds=n_folds,
        recovery=recovery
      )
      if complete is not None:
        atomic_checkpoint(complete, {'fold_result': fold_result, 'category_means': category_means})
    all_fold_results.append(fold_result)
    if fold_result['best_val_loss'] < best_overall_val:
      best_overall_val = fold_result['best_val_loss']
      best_overall_state = fold_result['state_dict']
      best_overall_metrics = fold_result['metrics']
      best_category_means = category_means
      best_fold_idx = fold_idx
  if n_folds > 1:
    print(f"\n{'=' * 60}")
    print(f'  Cross-Validation Summary ({n_folds} folds)')
    print(f"{'=' * 60}")
    val_losses = [r['best_val_loss'] for r in all_fold_results]
    pearson_vals = [r['metrics'].get('pearson_resid', 0) for r in all_fold_results]
    spearman_vals = [r['metrics'].get('spearman_resid', 0) for r in all_fold_results]
    mae_vals = [r['metrics'].get('mae', 0) for r in all_fold_results]
    for (i, r) in enumerate(all_fold_results):
      m = r['metrics']
      print(f'  Fold {i + 1}: loss={r["best_val_loss"]:.4f} │ r={m.get("pearson_resid", 0):.3f}'
            f' │ ρ={m.get("spearman_resid", 0):.3f} │ MAE={m.get("mae", 0):.4f} │ epoch={r["best_epoch"]}')
    print(f"  {'─' * 56}")
    print(f'  Mean : loss={np.mean(val_losses):.4f}±{np.std(val_losses):.4f}'
          f' │ r={np.mean(pearson_vals):.3f}±{np.std(pearson_vals):.3f}'
          f' │ ρ={np.mean(spearman_vals):.3f}±{np.std(spearman_vals):.3f}'
          f' │ MAE={np.mean(mae_vals):.4f}±{np.std(mae_vals):.4f}')
    print(f'  Best fold: {best_fold_idx + 1} (val_loss={best_overall_val:.4f})')
  checkpoint = {
    'epoch': all_fold_results[best_fold_idx]['best_epoch'],
    'model_state_dict': best_overall_state,
    'val_loss': best_overall_val,
    'metrics': best_overall_metrics,
    'category_means': best_category_means,
    'run_config': {
      'pretrained_path': str(pretrained_path) if pretrained_path is not None else None,
      'epochs': epochs,
      'batch_size': batch_size,
      'lr': lr,
      'patience': patience,
      'freeze_encoder_epochs': freeze_encoder_epochs,
      'aux_weight': aux_weight,
      'run_name': run_name,
      'checkpoint_stem': checkpoint_stem
    },
    'cv_results': {
      'n_folds': n_folds,
      'best_fold_idx': best_fold_idx,
      'fold_metrics': [r['metrics'] for r in all_fold_results],
      'fold_val_losses': [r['best_val_loss'] for r in all_fold_results]
    } if n_folds > 1 else None
  }
  if n_folds > 1:
    checkpoint['fold_state_dicts'] = [r['state_dict'] for r in all_fold_results]
  atomic_checkpoint(save_path, checkpoint)
  print(f'\n  Best val loss: {best_overall_val:.4f}')
  print(f'  Saved → {save_path}')
  return best_overall_val


def _checkpoint_stem(pretrained_path: Path | None, run_name: str | None) -> str:
  if run_name:
    sanitized = ''.join((ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in run_name.strip())).strip('_')
    if sanitized:
      return sanitized
  if pretrained_path is None:
    return 'multisubj_scratch_best'
  return 'multisubj_best'
