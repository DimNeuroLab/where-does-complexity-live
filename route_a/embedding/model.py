"""Image-embedding complexity dataset, encoder, and full model.

Predicts complexity directly from DINOv2 tiered embeddings and/or CLIP image
embeddings (no fMRI), with FiLM target conditioning from
:class:`shared.complexity_head.ComplexityHead`.

The encoder input is selected by ``features``: ``'all'`` (DINO + CLIP, the
paper's main model, Fig. 5/6 and Table 6) or ``'clip_only'`` / ``'dino_only'``
(the feature-family ablation, Appendix Fig./Table B1).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from shared.complexity_data import CATEGORY_TO_IDX, ComplexityRecord, get_clip_text_embeddings, load_complexity_records
from shared.complexity_head import ComplexityHead
from shared.constants import CLIP_EMBED_DIM, DINO_TIER_LAYERS, NSD_SUBJECTS
from shared.nsd_utils import list_training_images, parse_image_filename

DINO_DIM = 1024
CLIP_IMG_DIM = 768

FeatureSet = Literal['all', 'clip_only', 'dino_only']

#: Encoder input dimension for each feature-set ablation.
FEATURE_SET_DIMS: dict[str, int] = {
  'all': 3 * DINO_DIM + CLIP_IMG_DIM,
  'clip_only': CLIP_IMG_DIM,
  'dino_only': 3 * DINO_DIM,
}

#: Encoder width/depth presets swept in the paper; ``small`` is used for all
#: reported results.
ARCH_CONFIGS: dict[str, dict[str, float]] = {
  'tiny': {'latent': 256, 'hidden': 128, 'dropout': 0.5},
  'small': {'latent': 512, 'hidden': 256, 'dropout': 0.5},
  'medium': {'latent': 1024, 'hidden': 512, 'dropout': 0.4},
}


def select_features(
  features: FeatureSet,
  dino_early: torch.Tensor,
  dino_mid: torch.Tensor,
  dino_late: torch.Tensor,
  clip_img: torch.Tensor,
) -> torch.Tensor:
  """Concatenate the embedding tensors used by one feature-set ablation.

  :raises ValueError: If ``features`` is not a recognised feature set.
  """
  if features == 'all':
    return torch.cat([dino_early, dino_mid, dino_late, clip_img], dim=-1)
  if features == 'clip_only':
    return clip_img
  if features == 'dino_only':
    return torch.cat([dino_early, dino_mid, dino_late], dim=-1)
  raise ValueError(f'Unknown feature set: {features!r}')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageComplexityDataset(Dataset):
  """Image-embedding complexity dataset (no fMRI, no subject ID).

  Every unique ``(nsd_id, task)`` pair in the complexity records becomes one
  sample. DINO/CLIP image features are subject-independent (the same NSD
  image yields the same embedding regardless of which subject viewed it),
  so each image's features are loaded once, from whichever subject saw it
  first.

  :param nsd_root: Root of the NSD/Algonauts-2023 data tree, used only to
    enumerate each subject's training image filenames.
  :param dino_dir: Directory containing ``{subject}_dino_layers.npy`` files.
  :param clip_img_dir: Directory containing ``{subject}_clip_img.npy`` files.
  :param clip_text_path: Path to the prompt-ensembled CLIP text embeddings.
  :param complexity_csv: Complexity-ranking CSV, required if ``records`` is
    omitted.
  :param records: Pre-loaded complexity records; loaded from
    ``complexity_csv`` if omitted.
  :param category_means: ``{task: mean_score}`` computed on the training
    records. Pass ``None`` for the training split (computed here), then pass
    ``train_dataset.category_means`` to the validation split, to avoid
    leaking validation targets into the residualisation.
  :param subjects: Which subjects' feature files to scan for images.
    Defaults to all NSD subjects, so every NSD image seen by any subject is
    covered.
  """

  def __init__(
    self,
    nsd_root: Path,
    dino_dir: Path,
    clip_img_dir: Path,
    clip_text_path: Path,
    complexity_csv: Path | None = None,
    records: list[ComplexityRecord] | None = None,
    category_means: dict[str, float] | None = None,
    subjects: list[str] | None = None,
  ) -> None:
    self.subjects = subjects or NSD_SUBJECTS

    if records is None:
      if complexity_csv is None:
        raise ValueError('Provide either records or complexity_csv.')
      records = load_complexity_records(complexity_csv)

    if category_means is None:
      category_scores: dict[str, list[float]] = defaultdict(list)
      for record in records:
        category_scores[record['task']].append(record['score'])
      self.category_means: dict[str, float] = {
        category: float(np.mean(scores)) for category, scores in category_scores.items()
      }
    else:
      self.category_means = category_means

    # Union NSD ID -> (subject, row) map across all subjects: first subject
    # to have seen an image wins, since the features are identical either way.
    nsd_to_subject_row: dict[int, tuple[str, int]] = {}
    for subject in self.subjects:
      for path in list_training_images(nsd_root, subject):
        _, index, nsd_id = parse_image_filename(path.name)
        if nsd_id not in nsd_to_subject_row:
          nsd_to_subject_row[nsd_id] = (subject, index - 1)

    needed_ids = {
      record['nsd_id'] for record in records
      if record['nsd_id'] in nsd_to_subject_row and CATEGORY_TO_IDX.get(record['task']) is not None
    }

    subject_to_needed: dict[str, list[int]] = defaultdict(list)
    for nsd_id in needed_ids:
      subject, _ = nsd_to_subject_row[nsd_id]
      subject_to_needed[subject].append(nsd_id)

    nsd_to_feat_idx: dict[int, int] = {}
    dino_early_list: list[np.ndarray] = []
    dino_mid_list: list[np.ndarray] = []
    dino_late_list: list[np.ndarray] = []
    clip_img_list: list[np.ndarray] = []

    feat_counter = 0
    for subject in self.subjects:
      ids_for_subject = subject_to_needed.get(subject, [])
      if not ids_for_subject:
        continue

      dino_layers = np.load(dino_dir / f'{subject}_dino_layers.npy')  # (N_subject, n_layers, 1024)
      clip_arr = np.load(clip_img_dir / f'{subject}_clip_img.npy')  # (N_subject, 768)

      for nsd_id in ids_for_subject:
        _, row = nsd_to_subject_row[nsd_id]
        dino_early_list.append(dino_layers[row, DINO_TIER_LAYERS['early'], :].mean(axis=0))
        dino_mid_list.append(dino_layers[row, DINO_TIER_LAYERS['mid'], :].mean(axis=0))
        dino_late_list.append(dino_layers[row, DINO_TIER_LAYERS['late'], :].mean(axis=0))
        clip_img_list.append(clip_arr[row])
        nsd_to_feat_idx[nsd_id] = feat_counter
        feat_counter += 1

      del dino_layers, clip_arr

    self._dino_early = torch.tensor(np.array(dino_early_list, dtype=np.float32).tolist(), dtype=torch.float32)
    self._dino_mid = torch.tensor(np.array(dino_mid_list, dtype=np.float32).tolist(), dtype=torch.float32)
    self._dino_late = torch.tensor(np.array(dino_late_list, dtype=np.float32).tolist(), dtype=torch.float32)
    self._clip_img = torch.tensor(np.array(clip_img_list, dtype=np.float32).tolist(), dtype=torch.float32)

    self.samples: list[tuple[int, int, float, str]] = []  # (feat_idx, cat_idx, residual, task)
    skipped = 0
    for record in records:
      feat_idx = nsd_to_feat_idx.get(record['nsd_id'])
      if feat_idx is None:
        skipped += 1
        continue
      cat_idx = CATEGORY_TO_IDX.get(record['task'])
      if cat_idx is None:
        skipped += 1
        continue
      residual = record['score'] - self.category_means[record['task']]
      self.samples.append((feat_idx, cat_idx, residual, record['task']))

    if skipped:
      print(f'[ImageComplexityDataset] skipped {skipped} records (NSD ID not found or unknown category)')
    print(
      f'[ImageComplexityDataset] {len(self.samples)} samples from {len(nsd_to_feat_idx)} unique images '
      f'across {len(subject_to_needed)} subjects'
    )

    # Exposed so evaluation code can recover the NSD ID from a feature index
    # without rebuilding this mapping independently (which risks a different order).
    self.nsd_to_feat_idx: dict[int, int] = nsd_to_feat_idx
    self.feat_idx_to_nsd: dict[int, int] = {v: k for k, v in nsd_to_feat_idx.items()}

    raw_clip_text = get_clip_text_embeddings(clip_text_path)
    self._clip_text: dict[str, torch.Tensor] = {
      category: torch.tensor(embedding.tolist(), dtype=torch.float32)
      for category, embedding in raw_clip_text.items()
    }
    self._clip_text_zero = torch.zeros(CLIP_EMBED_DIM, dtype=torch.float32)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    feat_idx, cat_idx, residual, task = self.samples[index]
    return {
      'dino_early': self._dino_early[feat_idx],
      'dino_mid': self._dino_mid[feat_idx],
      'dino_late': self._dino_late[feat_idx],
      'clip_img': self._clip_img[feat_idx],
      'clip_text': self._clip_text.get(task, self._clip_text_zero),
      'category_idx': torch.tensor(cat_idx, dtype=torch.long),
      'score': torch.tensor(residual, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
  """Two-block MLP mapping the selected image embeddings to a latent representation."""

  def __init__(self, features: FeatureSet, latent_dim: int, dropout: float) -> None:
    super().__init__()
    self.features = features
    input_dim = FEATURE_SET_DIMS[features]
    self.net = nn.Sequential(
      nn.Linear(input_dim, latent_dim),
      nn.LayerNorm(latent_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(latent_dim, latent_dim),
      nn.LayerNorm(latent_dim),
    )

  def forward(
    self,
    dino_early: torch.Tensor,
    dino_mid: torch.Tensor,
    dino_late: torch.Tensor,
    clip_img: torch.Tensor,
  ) -> torch.Tensor:
    return self.net(select_features(self.features, dino_early, dino_mid, dino_late, clip_img))


class ImageComplexityModel(nn.Module):
  """End-to-end: selected image embeddings -> :class:`Encoder` -> FiLM complexity head.

  This is the architecture used for all paper results (``arch='small'``).
  """

  def __init__(
    self,
    arch: str = 'small',
    features: FeatureSet = 'all',
    n_categories: int = 16,
    clip_text_dim: int = 768,
    cat_embed_dim: int = 16,
  ) -> None:
    super().__init__()
    config = ARCH_CONFIGS[arch]
    self.encoder = Encoder(features=features, latent_dim=int(config['latent']), dropout=config['dropout'])
    self.complexity_head = ComplexityHead(
      brain_dim=int(config['latent']),
      clip_text_dim=clip_text_dim,
      n_categories=n_categories,
      cat_embed_dim=cat_embed_dim,
      hidden_dim=int(config['hidden']),
      dropout=config['dropout'],
    )

  def forward(
    self,
    dino_early: torch.Tensor,
    dino_mid: torch.Tensor,
    dino_late: torch.Tensor,
    clip_img: torch.Tensor,
    category_idx: torch.Tensor,
    clip_text: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(mu, log_var)``, each shape ``(batch,)``."""
    z = self.encoder(dino_early, dino_mid, dino_late, clip_img)
    return self.complexity_head(z, category_idx, clip_text)


class DirectFeedModel(nn.Module):
  """No encoder MLP: the selected image embeddings feed the complexity head directly.

  The most regularised option: a single learned projection (the head's own
  ``brain_proj``) rather than the two stacked ones in :class:`Encoder`.
  """

  def __init__(
    self,
    features: FeatureSet = 'all',
    n_categories: int = 16,
    clip_text_dim: int = 768,
    head_hidden: int = 256,
    cat_embed_dim: int = 16,
    dropout: float = 0.3,
  ) -> None:
    super().__init__()
    self.features = features
    self.complexity_head = ComplexityHead(
      brain_dim=FEATURE_SET_DIMS[features],
      clip_text_dim=clip_text_dim,
      n_categories=n_categories,
      cat_embed_dim=cat_embed_dim,
      hidden_dim=head_hidden,
      dropout=dropout,
    )

  def forward(
    self,
    dino_early: torch.Tensor,
    dino_mid: torch.Tensor,
    dino_late: torch.Tensor,
    clip_img: torch.Tensor,
    category_idx: torch.Tensor,
    clip_text: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    z = select_features(self.features, dino_early, dino_mid, dino_late, clip_img)
    return self.complexity_head(z, category_idx, clip_text)


def make_model(arch: str, features: FeatureSet = 'all', n_categories: int = 16) -> nn.Module:
  """Build the model for a given architecture and feature-set ablation.

  :param arch: ``'tiny'``, ``'small'`` (used for all paper results), or ``'medium'`` for an
    :class:`Encoder` + :class:`ComplexityHead <shared.complexity_head.ComplexityHead>` model, or
    ``'direct'`` for a :class:`DirectFeedModel` with no encoder MLP.
  :param features: ``'all'`` (DINO + CLIP, Fig. 5/6 and Table 6),
    ``'clip_only'`` or ``'dino_only'`` (Appendix Fig./Table B1).
  """
  if arch == 'direct':
    return DirectFeedModel(features=features, n_categories=n_categories)
  return ImageComplexityModel(arch=arch, features=features, n_categories=n_categories)
