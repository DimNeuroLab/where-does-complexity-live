"""ROI utilities: load stream masks, group voxels into tiers, fit & apply PCA."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from route_b.config import paths as get_paths
from route_b.constants import NSD_SUBJECTS, PCA_COMPONENTS_PER_TIER, ROI_TIER_LABELS


def load_streams_mask(subj: str) -> np.ndarray:
  """Return the concatenated (lh + rh) streams mask in challenge space."""
  roi_dir = get_paths().nsd / subj / 'roi_masks'
  lh = np.load(roi_dir / 'lh.streams_challenge_space.npy')
  rh = np.load(roi_dir / 'rh.streams_challenge_space.npy')
  return np.concatenate([lh, rh])


def tier_voxel_indices(streams_mask: np.ndarray) -> dict[str, np.ndarray]:
  """Return {tier_name: array_of_voxel_indices} for the 3-tier grouping."""
  tier_idx = {}
  for (tier, labels) in ROI_TIER_LABELS.items():
    mask = np.isin(streams_mask, labels)
    tier_idx[tier] = np.where(mask)[0]
  return tier_idx


def all_roi_voxel_indices(streams_mask: np.ndarray) -> np.ndarray:
  """Return sorted indices of all voxels that belong to *any* ROI (label > 0)."""
  return np.where(streams_mask > 0)[0]


def fit_subject_pca(
  subj: str,
  fmri: np.ndarray,
  train_idx: np.ndarray,
  n_components: int = PCA_COMPONENTS_PER_TIER
) -> dict[str, PCA]:
  """Fit PCA per tier for a single subject."""
  streams = load_streams_mask(subj)
  tiers = tier_voxel_indices(streams)
  pca_models = {}
  for (tier_name, vox_idx) in tiers.items():
    fmri_tier = fmri[train_idx][:, vox_idx]
    n_comp = min(n_components, fmri_tier.shape[0], fmri_tier.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(fmri_tier)
    pca_models[tier_name] = pca
    explained = pca.explained_variance_ratio_.sum()
    print(f'  [{subj}] {tier_name:10s}: {len(vox_idx):6d} voxels → {n_comp} PCs '
          f'(explained variance: {explained:.3f})')
  return pca_models


def apply_subject_pca(subj: str, fmri: np.ndarray, pca_models: dict[str, PCA]) -> np.ndarray:
  """Apply per-tier PCA to get a fixed-dim representation."""
  streams = load_streams_mask(subj)
  tiers = tier_voxel_indices(streams)
  parts = []
  for tier_name in ['early', 'mid', 'late']:
    vox_idx = tiers[tier_name]
    fmri_tier = fmri[:, vox_idx]
    transformed = pca_models[tier_name].transform(fmri_tier)
    parts.append(transformed.astype(np.float32))
  return np.concatenate(parts, axis=1)


def save_pca_models(subj: str, pca_models: dict[str, PCA]) -> Path:
  """Pickle PCA models for a subject."""
  out = get_paths().pca / f'{subj}_pca_models.pkl'
  with open(out, 'wb') as f:
    pickle.dump(pca_models, f)
  return out


def load_pca_models(subj: str) -> dict[str, PCA]:
  """Load pickled PCA models for a subject."""
  path = (get_paths().pca_models or get_paths().pca) / f'{subj}_pca_models.pkl'
  with open(path, 'rb') as f:
    return pickle.load(f)


def prepare_all_subjects_pca(
  val_frac: float = 0.1, n_components: int = PCA_COMPONENTS_PER_TIER,
  seed: int = 42, subjects: list[str] | None = None,
) -> None:
  """Fit per-tier PCA for every subject and save transformed data."""
  from route_b.data.nsd import load_training_fmri
  get_paths().pca.mkdir(parents=True, exist_ok=True)
  for subj in subjects or NSD_SUBJECTS:
    print(f"\n{'=' * 60}")
    print(f'  PCA preparation · {subj}')
    print(f"{'=' * 60}")
    (lh, rh) = load_training_fmri(subj)
    fmri = np.concatenate([lh, rh], axis=1)
    del lh, rh
    n_images = fmri.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_images)
    n_val = int(n_images * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    np.save(get_paths().pca / f'{subj}_train_idx.npy', train_idx)
    np.save(get_paths().pca / f'{subj}_val_idx.npy', val_idx)
    pca_models = fit_subject_pca(subj, fmri, train_idx, n_components)
    save_pca_models(subj, pca_models)
    pca_fmri = apply_subject_pca(subj, fmri, pca_models)
    np.save(get_paths().pca / f'{subj}_pca_fmri.npy', pca_fmri)
    print(f'  → {subj} PCA fMRI shape: {pca_fmri.shape}')
    del fmri, pca_fmri
