"""K-fold training of the embedding-conditioned complexity model.

Trains :func:`route_a.embedding.model.make_model` with k-fold cross-validation.
The paper's main model (Fig. 5/6, Table 6) uses ``--arch small --features all
--noise 0``. The feature-family ablation (Appendix Fig./Table B1) reruns this
with ``--features clip_only`` and ``--features dino_only``.

Noise augmentation (``--noise``) was only used to test its effect on training
and is not part of the reported model: all paper results use ``--noise 0``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from route_a.embedding.model import ARCH_CONFIGS, FeatureSet, ImageComplexityDataset, make_model
from shared.complexity_head import HeteroscedasticLoss
from shared.constants import DEFAULT_SEED


class _NoisyDataset(Dataset):
  """Wraps a dataset, adding fresh Gaussian noise to the embedding fields on every read."""

  def __init__(self, base: Dataset, noise_std: float) -> None:
    self.base = base
    self.noise_std = noise_std

  def __len__(self) -> int:
    return len(self.base)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    item = dict(self.base[index])
    if self.noise_std > 0:
      for key in ('dino_early', 'dino_mid', 'dino_late', 'clip_img'):
        item[key] = item[key] + torch.randn_like(item[key]) * self.noise_std
    return item


def _run_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: HeteroscedasticLoss,
  optimizer: torch.optim.Optimizer | None,
  device: torch.device,
  is_train: bool,
) -> float:
  model.train() if is_train else model.eval()
  total, n = 0.0, 0
  context = torch.enable_grad() if is_train else torch.no_grad()
  with context:
    for batch in loader:
      mu, log_var = model(
        batch['dino_early'].to(device), batch['dino_mid'].to(device), batch['dino_late'].to(device),
        batch['clip_img'].to(device), batch['category_idx'].to(device), batch['clip_text'].to(device),
      )
      loss = criterion(mu, log_var, batch['score'].to(device))
      if is_train:
        assert optimizer is not None
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
      total += loss.item()
      n += 1
  return total / max(n, 1)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
  model.eval()
  preds: list[float] = []
  targets: list[float] = []
  for batch in loader:
    mu, _ = model(
      batch['dino_early'].to(device), batch['dino_mid'].to(device), batch['dino_late'].to(device),
      batch['clip_img'].to(device), batch['category_idx'].to(device), batch['clip_text'].to(device),
    )
    preds.extend(mu.cpu().tolist())
    targets.extend(batch['score'].tolist())
  return np.array(preds, dtype=np.float32), np.array(targets, dtype=np.float32)


def _train_fold(
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  epochs: int,
  lr: float,
  patience: int,
  device: torch.device,
) -> tuple[float, float]:
  """Train until early stopping.

  :returns: ``(best_val_loss, best_pearson_r)`` at the best-val-loss epoch.
  """
  criterion = HeteroscedasticLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  best_val_loss = float('inf')
  best_r = float('nan')
  patience_counter = 0
  epoch = 0

  for epoch in range(1, epochs + 1):
    _run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
    val_loss = _run_epoch(model, val_loader, criterion, None, device, is_train=False)
    scheduler.step()

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      preds, targets = _predict(model, val_loader, device)
      if len(preds) >= 4 and preds.std() > 0 and targets.std() > 0:
        best_r, _ = stats.pearsonr(preds, targets)
      patience_counter = 0
    else:
      patience_counter += 1
      if patience_counter >= patience:
        break

  print(f'  Epochs trained: {epoch}')
  return best_val_loss, float(best_r)


def _kfold_indices(n: int, k: int, seed: int) -> Sequence[tuple[list[int], list[int]]]:
  rng = np.random.RandomState(seed)
  indices = rng.permutation(n)
  folds = np.array_split(indices, k)
  return [
    (np.concatenate([folds[j] for j in range(k) if j != i]).tolist(), folds[i].tolist())
    for i in range(k)
  ]


def run_kfold(
  *,
  dataset: ImageComplexityDataset,
  arch: str,
  features: FeatureSet,
  folds: int,
  epochs: int,
  lr: float,
  patience: int,
  batch_size: int,
  noise: float,
  seed: int,
  device: torch.device,
  checkpoint_dir: Path,
  results_path: Path,
) -> None:
  """Run k-fold cross-validated training and write per-fold and deployment checkpoints.

  Each per-fold checkpoint stores its ``val_idx``, so evaluation code can
  reconstruct the exact held-out set without leakage. The deployment
  checkpoint is retrained on the full dataset afterwards and must not be
  evaluated on any held-out split; the k-fold Pearson r is the honest
  performance estimate.
  """
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  fold_rs: list[float] = []
  start = time.time()

  for fold, (train_idx, val_idx) in enumerate(_kfold_indices(len(dataset), folds, seed), start=1):
    print(f'\n  Fold {fold}/{folds}  (train={len(train_idx)}, val={len(val_idx)})')

    train_subset: Dataset = Subset(dataset, train_idx)
    if noise > 0:
      train_subset = _NoisyDataset(train_subset, noise_std=noise)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)

    model = make_model(arch, features).to(device)
    best_loss, best_r = _train_fold(model, train_loader, val_loader, epochs=epochs, lr=lr, patience=patience, device=device)
    fold_rs.append(best_r)
    print(f'    Best r={best_r:.4f}  val_loss={best_loss:.4f}  ({(time.time() - start) / 60:.1f} min total)')

    fold_ckpt = checkpoint_dir / f'route_a_embedding_{features}_{arch}_noise{noise:.2f}_fold{fold}of{folds}.pt'
    torch.save({
      'model': model.state_dict(),
      'arch': arch,
      'features': features,
      'fold': fold,
      'n_folds': folds,
      'val_idx': val_idx,
      'best_r': best_r,
      'val_loss': best_loss,
      'noise': noise,
    }, fold_ckpt)
    print(f'    Fold checkpoint -> {fold_ckpt}')

  fold_rs_arr = np.array(fold_rs)
  print(f'\n  {folds}-fold results (arch={arch}, features={features}, noise={noise})')
  print(f'  Pearson r: {fold_rs_arr.mean():.4f} +/- {fold_rs_arr.std():.4f}')

  print('\n  Retraining on the full dataset (deployment checkpoint, do not evaluate on held-out data)...')
  full_train: Dataset = _NoisyDataset(dataset, noise_std=noise) if noise > 0 else dataset
  full_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, num_workers=0)

  model = make_model(arch, features).to(device)
  criterion = HeteroscedasticLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  for epoch in range(1, epochs + 1):
    loss = _run_epoch(model, full_loader, criterion, optimizer, device, is_train=True)
    scheduler.step()
    if epoch % 20 == 0:
      print(f'    Epoch {epoch}/{epochs}  train_loss={loss:.4f}')

  deploy_ckpt = checkpoint_dir / f'route_a_embedding_{features}_{arch}_noise{noise:.2f}_deploy_only.pt'
  torch.save({
    'model': model.state_dict(),
    'arch': arch,
    'features': features,
    'fold_pearson_mean': float(fold_rs_arr.mean()),
    'fold_pearson_std': float(fold_rs_arr.std()),
    'noise': noise,
    'warning': 'Trained on the full dataset. Do not evaluate on any held-out set.',
  }, deploy_ckpt)
  print(f'  Deployment checkpoint -> {deploy_ckpt}')

  results_path.parent.mkdir(parents=True, exist_ok=True)
  with open(results_path, 'w') as handle:
    json.dump({
      'arch': arch,
      'features': features,
      'arch_config': ARCH_CONFIGS.get(arch, {'note': 'direct feed, no encoder'}),
      'folds': folds,
      'noise': noise,
      'lr': lr,
      'epochs': epochs,
      'fold_pearson_r': fold_rs_arr.tolist(),
      'mean_r': float(fold_rs_arr.mean()),
      'std_r': float(fold_rs_arr.std()),
    }, handle, indent=2)
  print(f'  Results -> {results_path}')


def build_dataset(args: argparse.Namespace) -> ImageComplexityDataset:
  return ImageComplexityDataset(
    nsd_root=args.nsd_root,
    dino_dir=args.dino_dir,
    clip_img_dir=args.clip_img_dir,
    clip_text_path=args.clip_text_path,
    complexity_csv=args.complexity_csv,
  )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
  """Add the data-location flags shared by every route_a.embedding entry point."""
  parser.add_argument('--nsd-root', type=Path, required=True, help='Root of the NSD/Algonauts-2023 data tree.')
  parser.add_argument('--dino-dir', type=Path, required=True, help='Directory with {subject}_dino_layers.npy files.')
  parser.add_argument('--clip-img-dir', type=Path, required=True, help='Directory with {subject}_clip_img.npy files.')
  parser.add_argument('--clip-text-path', type=Path, required=True, help='Path to the CLIP text embeddings .npz file.')
  parser.add_argument('--complexity-csv', type=Path, required=True, help='Complexity-ranking CSV.')


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Train the embedding-conditioned complexity model with k-fold CV.')
  add_common_arguments(parser)
  parser.add_argument('--out', type=Path, required=True, help='Output directory for checkpoints and results.')
  parser.add_argument('--arch', default='small', choices=[*ARCH_CONFIGS, 'direct'])
  parser.add_argument(
    '--features', default='all', choices=['all', 'clip_only', 'dino_only'],
    help="Which image embeddings feed the model. 'all' is the paper's main model (Fig. 5/6, Table 6); "
         "'clip_only'/'dino_only' reproduce the ablation in Appendix Fig./Table B1.",
  )
  parser.add_argument('--folds', type=int, default=5)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--patience', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument(
    '--noise', type=float, default=0.0,
    help='Std of Gaussian noise added to embeddings during training (0 = off, used for all paper results).',
  )
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--device', default='cuda')
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(args.seed)

  dataset = build_dataset(args)
  results_path = args.out / f'kfold_results_{args.features}_{args.arch}_noise{args.noise:.2f}.json'
  run_kfold(
    dataset=dataset,
    arch=args.arch,
    features=args.features,
    folds=args.folds,
    epochs=args.epochs,
    lr=args.lr,
    patience=args.patience,
    batch_size=args.batch_size,
    noise=args.noise,
    seed=args.seed,
    device=device,
    checkpoint_dir=args.out / 'checkpoints',
    results_path=results_path,
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
