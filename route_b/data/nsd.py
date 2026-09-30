"""Utilities for loading and manipulating NSD / Algonauts-2023 data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from route_b.config import paths as get_paths
from route_b.constants import NSD_SUBJECTS

_TRAIN_RE = re.compile('train-(\\d+)_nsd-(\\d+)\\.png')
_TEST_RE = re.compile('test-(\\d+)_nsd-(\\d+)\\.png')


def parse_image_filename(fname: str) -> tuple[str, int, int]:
  """Return (split, index, nsd_id) from a filename like train-0001_nsd-00013.png."""
  m = _TRAIN_RE.match(fname)
  if m:
    return ('train', int(m.group(1)), int(m.group(2)))
  m = _TEST_RE.match(fname)
  if m:
    return ('test', int(m.group(1)), int(m.group(2)))
  raise ValueError(f'Cannot parse NSD filename: {fname}')


def subject_dir(subj: str) -> Path:
  """Return the root directory for a subject, e.g. 'subj01'."""
  return get_paths().nsd / subj


def training_images_dir(subj: str) -> Path:
  return subject_dir(subj) / 'training_split' / 'training_images'


def test_images_dir(subj: str) -> Path:
  return subject_dir(subj) / 'test_split' / 'test_images'


def list_training_images(subj: str) -> list[Path]:
  """Return sorted list of training image paths for *subj*."""
  d = training_images_dir(subj)
  return sorted(d.glob('train-*_nsd-*.png'))


def list_test_images(subj: str) -> list[Path]:
  d = test_images_dir(subj)
  return sorted(d.glob('test-*_nsd-*.png'))


def load_training_fmri(subj: str) -> tuple[np.ndarray, np.ndarray]:
  """Return (lh, rh) fMRI arrays, each shape (n_images, n_vertices)."""
  fmri_dir = subject_dir(subj) / 'training_split' / 'training_fmri'
  lh = np.load(fmri_dir / 'lh_training_fmri.npy')
  rh = np.load(fmri_dir / 'rh_training_fmri.npy')
  return (lh, rh)


def load_roi_mask(subj: str, hemi: str, roi_class: str, space: str = 'challenge') -> np.ndarray:
  """Load an ROI mask array."""
  fname = f'{hemi}.{roi_class}_{space}_space.npy'
  return np.load(subject_dir(subj) / 'roi_masks' / fname)


def load_roi_mapping(subj: str, roi_class: str) -> dict[int, str]:
  """Load the integer→ROI-name mapping dict for *roi_class*."""
  p = subject_dir(subj) / 'roi_masks' / f'mapping_{roi_class}.npy'
  return np.load(p, allow_pickle=True).item()


def build_nsd_id_index(subj: str) -> dict[int, int]:
  """Return {nsd_id: train_index (0-based)} for a subject's training images."""
  mapping: dict[int, int] = {}
  for p in list_training_images(subj):
    (_, idx, nsd_id) = parse_image_filename(p.name)
    mapping[nsd_id] = idx - 1
  return mapping


def load_image(path: Path, size: int | None = None) -> Image.Image:
  """Load an image as RGB, optionally resize to *size* × *size*."""
  img = Image.open(path).convert('RGB')
  if size is not None:
    img = img.resize((size, size), Image.BICUBIC)
  return img


def all_unique_nsd_ids() -> dict[int, list[str]]:
  """Return {nsd_id: [subj, ...]} mapping across all subjects."""
  from collections import defaultdict
  mapping: dict[int, list[str]] = defaultdict(list)
  for subj in NSD_SUBJECTS:
    for p in list_training_images(subj):
      (_, _, nsd_id) = parse_image_filename(p.name)
      mapping[nsd_id].append(subj)
  return dict(mapping)
