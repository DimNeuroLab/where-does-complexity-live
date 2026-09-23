"""Complexity-score records and CLIP text embeddings.

Reads the target-conditioned complexity ranking (an image-target-score CSV
produced by the ``complexity/`` component) and the cached, prompt-ensembled
CLIP text embeddings produced by :mod:`shared.features`.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from shared.constants import COCO_SEARCH18_CATEGORIES, DEFAULT_SEED

#: Category name -> integer index, for a learned category embedding.
CATEGORY_TO_IDX: dict[str, int] = {category: index for index, category in enumerate(COCO_SEARCH18_CATEGORIES)}

_IMAGE_FILENAME_RE = re.compile(r'(?:train|test)-(\d+)_nsd-(\d+)(?:_.+)?\.png')


class ComplexityRecord(dict):
  """A single (image, target category, complexity score) record.

  Kept as a plain ``dict`` subclass (keys: ``train_idx``, ``nsd_id``,
  ``task``, ``score``, ``image``, ``split``) so existing dict-style access
  keeps working, while still giving callers a named type to annotate with.
  """


def load_complexity_records(csv_path: Path, train_only: bool = True) -> list[ComplexityRecord]:
  """Parse the complexity-ranking CSV.

  :param csv_path: Path to the complexity CSV (``image,task,score`` columns,
    where ``image`` encodes the split, running index, and NSD ID, for
    example ``train-2152_nsd-18280_cup.png``).
  :param train_only: If true, drop rows from the held-out test split.
  :returns: One record per row that matches the expected filename pattern.
  """
  records: list[ComplexityRecord] = []
  with open(csv_path) as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      match = _IMAGE_FILENAME_RE.match(row['image'])
      if not match:
        continue
      split = 'train' if row['image'].startswith('train-') else 'test'
      if train_only and split != 'train':
        continue
      records.append(ComplexityRecord(
        train_idx=int(match.group(1)),
        nsd_id=int(match.group(2)),
        task=row['task'],
        score=float(row['score']),
        image=row['image'],
        split=split,
      ))
  return records


def complexity_split_by_image(
  records: list[ComplexityRecord],
  val_frac: float = 0.1,
  seed: int = DEFAULT_SEED,
) -> tuple[list[ComplexityRecord], list[ComplexityRecord]]:
  """Split complexity records by unique NSD image ID.

  Splitting by image, not by row, keeps every target category for a given
  image on the same side of the split.
  """
  rng = np.random.RandomState(seed)
  nsd_ids = sorted({record['nsd_id'] for record in records})
  rng.shuffle(nsd_ids)
  n_val = int(len(nsd_ids) * val_frac)
  val_ids = set(nsd_ids[:n_val])
  train_records = [record for record in records if record['nsd_id'] not in val_ids]
  val_records = [record for record in records if record['nsd_id'] in val_ids]
  return train_records, val_records


@lru_cache(maxsize=8)
def get_clip_text_embeddings(clip_text_path: Path) -> dict[str, np.ndarray]:
  """Load the prompt-ensembled CLIP text embeddings written by
  :func:`shared.features.extract_and_save_clip`, cached per path.
  """
  data = np.load(clip_text_path)
  return {category: data[category].astype(np.float32) for category in data.files}
