"""Phase 2 DNN training: multi-subject fMRI encoder → DINO tier heads + CLIP head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances, r2_score
from torch.utils.data import DataLoader

from route_b.config import paths as get_paths
from route_b.constants import (
  BATCH_SIZE,
  CLIP_EMBED_DIM,
  DINO_EMBED_DIM,
  FMRI_ENCODER_DROPOUT,
  FMRI_ENCODER_HIDDEN,
  FMRI_ENCODER_LATENT,
  INFONCE_TEMPERATURE,
  LEARNING_RATE,
  NUM_EPOCHS_PRETRAIN,
  NUM_SUBJECTS,
  NUM_WORKERS,
  PATIENCE,
  SEED,
  SUBJECT_EMBED_DIM,
  WEIGHT_DECAY,
)
from route_b.data.datasets import build_multi_subject_feature_dataset, get_complexity_val_nsd_ids
from route_b.models.encoder import BrainFeatureDecoder, FeatureDecodingLoss
from route_b.runtime import EpochRecovery, atomic_checkpoint, check_budget
from route_b.types import Array, Metrics, Sample, TensorMap


def set_seed(seed: int) -> None:
  import random
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _collate_fn(batch: list[Sample]) -> TensorMap:
  """Collate list of dicts into dict of tensors."""
  out = {}
  for key in batch[0]:
    vals = [b[key] for b in batch]
    if isinstance(vals[0], torch.Tensor):
      out[key] = torch.stack(vals)
    else:
      out[key] = torch.tensor(vals, dtype=torch.long)
  return out


def train_dnn_decoder(
  epochs: int = NUM_EPOCHS_PRETRAIN,
  batch_size: int = BATCH_SIZE,
  lr: float = LEARNING_RATE,
  patience: int = PATIENCE,
  visual_loss_weight: float = 1.0,
  fmri_recon_weight: float = 0.0,
  dino_loss_weight: float = 1.0,
  clip_loss_weight: float = 1.0,
  dino_loss_reduction: str = 'sum',
  run_name: str | None = None,
  gpus: list[int] | None = None,
  recovery_root: Path | None = None
) -> float:
  """Train multi-subject DNN feature decoder with InfoNCE loss."""
  set_seed(SEED)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f"\n{'=' * 60}")
  print(f'  Phase 2 · DNN Feature Decoder (multi-subject)')
  print(f"{'=' * 60}")
  complexity_val_ids = get_complexity_val_nsd_ids()
  print(f'  Excluding {len(complexity_val_ids)} complexity-val NSD IDs from encoder training')
  print(f'  Visual loss weight: {visual_loss_weight}')
  print(f'  DINO loss weight: {dino_loss_weight}')
  print(f'  DINO loss reduction: {dino_loss_reduction}')
  print(f'  CLIP loss weight: {clip_loss_weight}')
  print(f'  fMRI recon weight: {fmri_recon_weight}')
  (train_ds, train_stats) = build_multi_subject_feature_dataset('train', exclude_nsd_ids=complexity_val_ids)
  (val_ds, _) = build_multi_subject_feature_dataset('val')
  train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
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
  fmri_dim = train_ds.datasets[0].fmri_dim
  print(f'  PCA fMRI dim: {fmri_dim}')
  print(f'  Train: {len(train_ds)},  Val: {len(val_ds)}')
  model = BrainFeatureDecoder(
    fmri_dim=fmri_dim,
    dino_dim=DINO_EMBED_DIM,
    clip_dim=CLIP_EMBED_DIM,
    n_subjects=NUM_SUBJECTS,
    hidden_dim=FMRI_ENCODER_HIDDEN,
    latent_dim=FMRI_ENCODER_LATENT,
    dropout=FMRI_ENCODER_DROPOUT,
    subject_embed_dim=SUBJECT_EMBED_DIM
  ).to(device)
  if gpus and len(gpus) > 1:
    model = nn.DataParallel(model, device_ids=gpus)
    print(f'  Using {len(gpus)} GPUs: {gpus}')
  criterion = FeatureDecodingLoss(
    temperature=INFONCE_TEMPERATURE,
    visual_weight=visual_loss_weight,
    fmri_recon_weight=fmri_recon_weight,
    dino_weight=dino_loss_weight,
    clip_weight=clip_loss_weight,
    dino_reduction=dino_loss_reduction
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  best_val = float('inf')
  no_improve = 0
  save_dir = get_paths().checkpoints / 'feature_decoder'
  save_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_stem = _checkpoint_stem(visual_loss_weight, fmri_recon_weight, run_name)
  save_path = save_dir / f'{checkpoint_stem}.pt'
  recovery = EpochRecovery(recovery_root / checkpoint_stem, {
    'epochs': epochs, 'batch_size': batch_size, 'lr': lr, 'patience': patience,
    'visual_loss_weight': visual_loss_weight, 'fmri_recon_weight': fmri_recon_weight,
    'dino_loss_weight': dino_loss_weight, 'clip_loss_weight': clip_loss_weight,
    'dino_loss_reduction': dino_loss_reduction,
  }) if recovery_root is not None else None
  raw_model = model.module if isinstance(model, nn.DataParallel) else model
  resumed = recovery.restore(raw_model, optimizer, scheduler) if recovery else None
  start = resumed['epoch'] + 1 if resumed else 1
  if resumed:
    best_val, no_improve = resumed['progress']['best_val'], resumed['progress']['no_improve']
  for epoch in range(start, epochs + 1):
    if no_improve >= patience:
      break
    check_budget()
    model.train()
    train_losses = []
    for batch in train_loader:
      check_budget()
      fmri = batch['fmri'].to(device)
      subj_id = batch['subj_id'].to(device)
      targets = {k: batch[k].to(device) for k in ['dino_early', 'dino_mid', 'dino_late', 'clip']}
      raw_model = model.module if isinstance(model, nn.DataParallel) else model
      outputs = raw_model(fmri, subj_id)
      (loss, _) = criterion(outputs, targets, fmri=fmri)
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      train_losses.append(loss.item())
    scheduler.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
      for batch in val_loader:
        check_budget()
        fmri = batch['fmri'].to(device)
        subj_id = batch['subj_id'].to(device)
        targets = {k: batch[k].to(device) for k in ['dino_early', 'dino_mid', 'dino_late', 'clip']}
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        outputs = raw_model(fmri, subj_id)
        (loss, _) = criterion(outputs, targets, fmri=fmri)
        val_losses.append(loss.item())
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    if epoch % 5 == 0 or epoch == 1:
      print(f'  Epoch {epoch:3d}/{epochs} │ train {avg_train:.4f} │ val {avg_val:.4f}')
    if avg_val < best_val:
      best_val = avg_val
      no_improve = 0
      raw_model = model.module if isinstance(model, nn.DataParallel) else model
      atomic_checkpoint(
        save_path,
        {
          'epoch': epoch,
          'model_state_dict': raw_model.state_dict(),
          'val_loss': best_val,
          'fmri_dim': fmri_dim,
          'run_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'patience': patience,
            'visual_loss_weight': visual_loss_weight,
            'fmri_recon_weight': fmri_recon_weight,
            'dino_loss_weight': dino_loss_weight,
            'clip_loss_weight': clip_loss_weight,
            'dino_loss_reduction': dino_loss_reduction,
            'run_name': run_name,
            'checkpoint_stem': checkpoint_stem
          }
        }
      )
    else:
      no_improve += 1
    if recovery:
      recovery.save(epoch, raw_model, optimizer, scheduler, {'best_val': best_val, 'no_improve': no_improve})
    if no_improve >= patience:
      break
  print(f'  Best val loss: {best_val:.4f}')
  print(f'  Saved → {save_path}')
  raw_model = model.module if isinstance(model, nn.DataParallel) else model
  ckpt = torch.load(save_path, map_location=device, weights_only=False)
  raw_model.load_state_dict(ckpt['model_state_dict'])
  evaluation = evaluate_dnn_decoder(
    raw_model,
    val_loader,
    device,
    visual_loss_weight=visual_loss_weight,
    fmri_recon_weight=fmri_recon_weight
  )
  ckpt['evaluation'] = evaluation
  atomic_checkpoint(save_path, ckpt)
  return best_val


@torch.no_grad()
def evaluate_dnn_decoder(
  model: torch.nn.Module,
  val_loader: DataLoader,
  device: torch.device | str,
  visual_loss_weight: float,
  fmri_recon_weight: float
) -> Metrics:
  """Compute Pearson r, R², and retrieval accuracy."""
  model.eval()
  all_preds = {k: [] for k in ['dino_early', 'dino_mid', 'dino_late', 'clip']}
  all_gts = {k: [] for k in ['dino_early', 'dino_mid', 'dino_late', 'clip']}
  (all_recons, all_fmri) = ([], [])
  for batch in val_loader:
    check_budget()
    fmri = batch['fmri'].to(device)
    subj_id = batch['subj_id'].to(device)
    outputs = model(fmri, subj_id)
    if visual_loss_weight > 0:
      for key in all_preds:
        all_preds[key].append(outputs[key].cpu().numpy())
        all_gts[key].append(batch[key].numpy())
    if fmri_recon_weight > 0:
      all_recons.append(outputs['fmri_recon'].cpu().numpy())
      all_fmri.append(batch['fmri'].numpy())
  summary = {'visual': {}, 'fmri_recon': None}
  if visual_loss_weight > 0:
    print(f'\n  DNN Feature-space evaluation (val):')
    for key in all_preds:
      pred = np.concatenate(all_preds[key])
      gt = np.concatenate(all_gts[key])
      r = _pearson_mean(pred, gt)
      r2 = r2_score(gt, pred, multioutput='uniform_average')
      (top1, top5) = _retrieval_acc(pred, gt)
      summary['visual'][key] = {'pearson': r, 'r2': float(r2), 'top1': top1, 'top5': top5}
      print(f'    {key:12s}  r={r:.4f}  R²={r2:.4f}  top1={top1:.4f}  top5={top5:.4f}')
  if fmri_recon_weight > 0:
    recon = np.concatenate(all_recons)
    gt_fmri = np.concatenate(all_fmri)
    recon_mse = float(np.mean((recon - gt_fmri) ** 2))
    recon_r2 = float(r2_score(gt_fmri, recon, multioutput='uniform_average'))
    recon_r = _pearson_mean(recon, gt_fmri)
    summary['fmri_recon'] = {'mse': recon_mse, 'r2': recon_r2, 'pearson': recon_r}
    print(f'\n  fMRI reconstruction evaluation (val):')
    print(f'    recon         r={recon_r:.4f}  R²={recon_r2:.4f}  MSE={recon_mse:.4f}')
  return summary


def _checkpoint_stem(visual_loss_weight: float, fmri_recon_weight: float, run_name: str | None) -> str:
  if run_name:
    sanitized = ''.join((ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in run_name.strip())).strip('_')
    if sanitized:
      return sanitized
  if visual_loss_weight == 1.0 and fmri_recon_weight == 0.0:
    return 'dnn_multisubj_best'
  return f'dnn_vw{visual_loss_weight:g}_fw{fmri_recon_weight:g}'


def _pearson_mean(pred: Array, gt: Array) -> float:
  rs = [pearsonr(pred[:, d], gt[:, d])[0] for d in range(pred.shape[1])]
  return float(np.nanmean(rs))


def _retrieval_acc(pred: Array, gt: Array) -> tuple[float, float]:
  dists = pairwise_distances(pred, gt, metric='cosine')
  n = len(pred)
  correct = np.arange(n)
  top1 = float((np.argmin(dists, axis=1) == correct).mean())
  top5_idx = np.argsort(dists, axis=1)[:, :5]
  top5 = float(np.mean([i in top5_idx[j] for (j, i) in enumerate(correct)]))
  return (top1, top5)
