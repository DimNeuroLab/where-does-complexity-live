"""Datasets preserving the original record order and normalization."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from route_b.config import paths as get_paths
from route_b.constants import CLIP_EMBED_DIM, COCO_SEARCH18_CATEGORIES, DINO_TIER_LAYERS, NSD_SUBJECTS, SEED
from route_b.data.nsd import list_training_images, parse_image_filename
from route_b.types import Array, Record, Sample

_CLIP_TEXT_CACHE: dict[str, np.ndarray] | None = None
_CLIP_TEXT_CACHE_PATH: Path | None = None


def get_clip_text_embeddings() -> dict[str, np.ndarray]:
  """Load & cache precomputed CLIP text embeddings for each category."""
  global _CLIP_TEXT_CACHE, _CLIP_TEXT_CACHE_PATH
  path = get_paths().clip_text / 'clip_text_embeddings.npz'
  if _CLIP_TEXT_CACHE is None or _CLIP_TEXT_CACHE_PATH != path:
    data = np.load(path)
    _CLIP_TEXT_CACHE = {cat: data[cat].astype(np.float32) for cat in data.files}
    _CLIP_TEXT_CACHE_PATH = path
  return _CLIP_TEXT_CACHE
CATEGORY_TO_IDX = {cat: i for (i, cat) in enumerate(COCO_SEARCH18_CATEGORIES)}


def get_complexity_val_nsd_ids(val_frac: float = 0.1, seed: int = SEED) -> set[int]:
  """Return the original fixed image holdout used for encoder pretraining."""
  records = load_complexity_records(train_only=True)
  (_, val_recs) = complexity_split_by_image(records, val_frac, seed)
  return {r['nsd_id'] for r in val_recs}


def get_nsd_id_to_index(subj: str) -> dict[int, int]:
  """Return {nsd_id: 0-based row index} for a subject's training images."""
  mapping = {}
  for p in list_training_images(subj):
    (_, idx, nsd_id) = parse_image_filename(p.name)
    mapping[nsd_id] = idx - 1
  return mapping


class FeatureDecodingDataset(Dataset):
  """Single-subject dataset for Phase 2: (pca_fmri, subj_id, dino_tiers, clip)."""

  def __init__(
    self,
    subj: str,
    split: str = 'train',
    fmri_mean: np.ndarray | None = None,
    fmri_std: np.ndarray | None = None,
    exclude_nsd_ids: set[int] | None = None
  ) -> None:
    self.subj = subj
    self.subj_idx = NSD_SUBJECTS.index(subj)
    pca_fmri = np.load(get_paths().pca / f'{subj}_pca_fmri.npy')
    train_idx = np.load(get_paths().pca / f'{subj}_train_idx.npy')
    val_idx = np.load(get_paths().pca / f'{subj}_val_idx.npy')
    idx = train_idx if split == 'train' else val_idx
    if exclude_nsd_ids and split == 'train':
      nsd_to_row = get_nsd_id_to_index(subj)
      rows_to_exclude = {nsd_to_row[nid] for nid in exclude_nsd_ids if nid in nsd_to_row}
      idx = np.array([i for i in idx if i not in rows_to_exclude])
    pca_fmri = pca_fmri[idx]
    if fmri_mean is None:
      self._fmri_mean = pca_fmri.mean(axis=0)
      self._fmri_std = pca_fmri.std(axis=0) + 1e-06
    else:
      self._fmri_mean = fmri_mean
      self._fmri_std = fmri_std
    pca_fmri = (pca_fmri - self._fmri_mean) / self._fmri_std
    dino_layers = np.load(get_paths().dino / f'{subj}_dino_layers.npy')
    dino_layers = dino_layers[idx]
    self.dino_early = dino_layers[:, DINO_TIER_LAYERS['early'], :].mean(axis=1).astype(np.float32)
    self.dino_mid = dino_layers[:, DINO_TIER_LAYERS['mid'], :].mean(axis=1).astype(np.float32)
    self.dino_late = dino_layers[:, DINO_TIER_LAYERS['late'], :].mean(axis=1).astype(np.float32)
    del dino_layers
    clip_img = np.load(get_paths().clip_image / f'{subj}_clip_img.npy')[idx]
    self.fmri = torch.from_numpy(pca_fmri.astype(np.float32))
    self.clip_img = torch.from_numpy(clip_img.astype(np.float32))

  def __len__(self) -> int:
    return len(self.fmri)

  def __getitem__(self, idx: int) -> Sample:
    return {
      'fmri': self.fmri[idx],
      'subj_id': self.subj_idx,
      'dino_early': torch.from_numpy(self.dino_early[idx]),
      'dino_mid': torch.from_numpy(self.dino_mid[idx]),
      'dino_late': torch.from_numpy(self.dino_late[idx]),
      'clip': self.clip_img[idx]
    }

  @property
  def fmri_dim(self) -> int:
    return self.fmri.shape[1]


def build_multi_subject_feature_dataset(
  split: str = 'train',
  subjects: list[str] | None = None,
  exclude_nsd_ids: set[int] | None = None
) -> tuple[ConcatDataset, dict[str, tuple[Array, Array]]]:
  """Build a ConcatDataset pooling all subjects for Phase 2."""
  subjects = subjects or NSD_SUBJECTS
  datasets = []
  train_stats = {}
  for subj in subjects:
    ds = FeatureDecodingDataset(subj, split='train', exclude_nsd_ids=exclude_nsd_ids)
    train_stats[subj] = (ds._fmri_mean, ds._fmri_std)
    if split == 'train':
      datasets.append(ds)
  if split == 'val':
    for subj in subjects:
      (mean, std) = train_stats[subj]
      datasets.append(FeatureDecodingDataset(subj, split='val', fmri_mean=mean, fmri_std=std))
  return (ConcatDataset(datasets), train_stats)
_IMG_RE = re.compile('(?:train|test)-(\\d+)_nsd-(\\d+)(?:_.+)?\\.png')


def load_complexity_records(train_only: bool = True) -> list[Record]:
  """Parse the complexity CSV. Reads task from the 'task' column."""
  records = []
  with open(get_paths().ranking) as f:
    reader = csv.DictReader(f)
    for row in reader:
      m = _IMG_RE.match(row['image'])
      if not m:
        continue
      split = 'train' if row['image'].startswith('train-') else 'test'
      if train_only and split != 'train':
        continue
      records.append({
        'train_idx': int(m.group(1)),
        'nsd_id': int(m.group(2)),
        'task': row['task'],
        'score': float(row['score']),
        'image': row['image'],
        'split': split
      })
  return records


def complexity_split_by_image(
  records: list[Record], val_frac: float = 0.1, seed: int = 42,
) -> tuple[list[Record], list[Record]]:
  """Split complexity records by unique NSD image ID."""
  rng = np.random.RandomState(seed)
  nsd_ids = sorted({r['nsd_id'] for r in records})
  rng.shuffle(nsd_ids)
  n_val = int(len(nsd_ids) * val_frac)
  val_ids = set(nsd_ids[:n_val])
  train_recs = [r for r in records if r['nsd_id'] not in val_ids]
  val_recs = [r for r in records if r['nsd_id'] in val_ids]
  return (train_recs, val_recs)


def complexity_stratified_group_kfold(
  records: list[Record], n_folds: int = 5, seed: int = 42,
) -> list[tuple[list[Record], list[Record]]]:
  """K-fold splits stratified by category, grouped by NSD image ID."""
  from sklearn.model_selection import StratifiedGroupKFold
  categories = np.array([r['task'] for r in records])
  groups = np.array([r['nsd_id'] for r in records])
  X = np.arange(len(records))
  sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
  folds = []
  for (train_idx, val_idx) in sgkf.split(X, categories, groups):
    train_recs = [records[i] for i in train_idx]
    val_recs = [records[i] for i in val_idx]
    folds.append((train_recs, val_recs))
  return folds


class ComplexityDataset(Dataset):
  """Multi-subject complexity dataset using PCA-transformed fMRI."""

  def __init__(
    self,
    subj: str,
    records: list[Record] | None = None,
    fmri_mean: np.ndarray | None = None,
    fmri_std: np.ndarray | None = None,
    category_means: dict[str, float] | None = None,
    visual_targets: bool = True,
  ) -> None:
    self.subj = subj
    self.subj_idx = NSD_SUBJECTS.index(subj)
    pca_fmri = np.load(get_paths().pca / f'{subj}_pca_fmri.npy')
    if fmri_mean is None:
      self._fmri_mean = pca_fmri.mean(axis=0)
      self._fmri_std = pca_fmri.std(axis=0) + 1e-06
    else:
      self._fmri_mean = fmri_mean
      self._fmri_std = fmri_std
    pca_fmri = (pca_fmri - self._fmri_mean) / self._fmri_std
    self.fmri_all = torch.from_numpy(pca_fmri.astype(np.float32))
    self.visual_targets = visual_targets
    if visual_targets:
      self._load_visual_targets(subj)
    self._clip_text = get_clip_text_embeddings()
    self._build_samples(subj, records, category_means)

  def _load_visual_targets(self, subj: str) -> None:
    """Load image representations used only by the training objectives."""
    dino_layers = np.load(get_paths().dino / f'{subj}_dino_layers.npy')
    self._dino_early = dino_layers[:, DINO_TIER_LAYERS['early'], :].mean(axis=1).astype(np.float32)
    self._dino_mid = dino_layers[:, DINO_TIER_LAYERS['mid'], :].mean(axis=1).astype(np.float32)
    self._dino_late = dino_layers[:, DINO_TIER_LAYERS['late'], :].mean(axis=1).astype(np.float32)
    del dino_layers
    self._clip_img = np.load(get_paths().clip_image / f'{subj}_clip_img.npy').astype(np.float32)

  def _build_samples(
    self, subj: str, records: list[Record] | None, category_means: dict[str, float] | None,
  ) -> None:
    """Retain the historical subject, record, and target ordering."""
    nsd_to_row: dict[int, int] = {}
    for p in list_training_images(subj):
      (_, idx, nsd_id) = parse_image_filename(p.name)
      nsd_to_row[nsd_id] = idx - 1
    if records is None:
      records = load_complexity_records()
    if category_means is None:
      cat_scores: dict[str, list[float]] = defaultdict(list)
      for rec in records:
        cat_scores[rec['task']].append(rec['score'])
      self.category_means = {cat: np.mean(scores) for (cat, scores) in cat_scores.items()}
    else:
      self.category_means = category_means
    self.samples: list[tuple[int, int, float, str]] = []
    for rec in records:
      row = nsd_to_row.get(rec['nsd_id'])
      if row is not None:
        cat_idx = CATEGORY_TO_IDX.get(rec['task'])
        if cat_idx is not None:
          residual = rec['score'] - self.category_means[rec['task']]
          self.samples.append((row, cat_idx, residual, rec['task']))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Sample:
    (row, cat_idx, residual, task) = self.samples[idx]
    clip_text = self._clip_text.get(task, np.zeros(CLIP_EMBED_DIM, dtype=np.float32))
    sample = {
      'fmri': self.fmri_all[row],
      'subj_id': self.subj_idx,
      'category_idx': cat_idx,
      'score': torch.tensor(residual, dtype=torch.float32),
      'clip_text': torch.from_numpy(clip_text),
    }
    if self.visual_targets:
      sample.update({
      'dino_early': torch.from_numpy(self._dino_early[row]),
      'dino_mid': torch.from_numpy(self._dino_mid[row]),
      'dino_late': torch.from_numpy(self._dino_late[row]),
      'clip_img': torch.from_numpy(self._clip_img[row])
      })
    return sample

  @property
  def fmri_dim(self) -> int:
    return self.fmri_all.shape[1]
