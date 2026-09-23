"""NSD image filename parsing and listing.

Only what the embedding-extraction and dataset-building code needs: mapping
an NSD image ID to its on-disk training image file. fMRI and ROI loading are
Route B concerns and are not part of this module.
"""

from __future__ import annotations

import re
from pathlib import Path

_TRAIN_RE = re.compile(r'train-(\d+)_nsd-(\d+)\.png')
_TEST_RE = re.compile(r'test-(\d+)_nsd-(\d+)\.png')


def parse_image_filename(fname: str) -> tuple[str, int, int]:
  """Parse an NSD training/test image filename.

  :param fname: A filename such as ``train-0001_nsd-00013.png``.
  :returns: A ``(split, index, nsd_id)`` tuple.
  :raises ValueError: If the filename does not match the expected pattern.
  """
  match = _TRAIN_RE.match(fname)
  if match:
    return 'train', int(match.group(1)), int(match.group(2))
  match = _TEST_RE.match(fname)
  if match:
    return 'test', int(match.group(1)), int(match.group(2))
  raise ValueError(f'Cannot parse NSD filename: {fname}')


def subject_dir(nsd_root: Path, subject: str) -> Path:
  """Return the root directory for one subject, for example ``subj01``."""
  return nsd_root / subject


def training_images_dir(nsd_root: Path, subject: str) -> Path:
  return subject_dir(nsd_root, subject) / 'training_split' / 'training_images'


def list_training_images(nsd_root: Path, subject: str) -> list[Path]:
  """Return the sorted list of training image paths for one subject.

  :param nsd_root: Root of the NSD/Algonauts-2023 data tree.
  :param subject: Subject ID, for example ``subj01``.
  """
  directory = training_images_dir(nsd_root, subject)
  return sorted(directory.glob('train-*_nsd-*.png'))
