"""Canonical image-target rows for the engineered Route A predictor."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_TARGETS: tuple[str, ...] = (
  'bottle',
  'car',
  'chair',
  'clock',
  'cup',
  'fork',
  'keyboard',
  'laptop',
  'microwave',
  'mouse',
  'oven',
  'potted plant',
  'sink',
  'stop sign',
  'toilet',
  'tv',
)

_TARGET_ALIASES: dict[str, str] = {target: target for target in SUPPORTED_TARGETS}
_TARGET_ALIASES.update({
  'automobile': 'car',
  'cars': 'car',
  'chairs': 'chair',
  'clocks': 'clock',
  'cups': 'cup',
  'forks': 'fork',
  'keyboards': 'keyboard',
  'laptops': 'laptop',
  'microwaves': 'microwave',
  'mice': 'mouse',
  'ovens': 'oven',
  'pottedplant': 'potted plant',
  'plant': 'potted plant',
  'plants': 'potted plant',
  'potted plants': 'potted plant',
  'sink basin': 'sink',
  'sinks': 'sink',
  'stopsign': 'stop sign',
  'stop signs': 'stop sign',
  'toilets': 'toilet',
  'television': 'tv',
  'televisions': 'tv',
  'monitor': 'tv',
  'monitors': 'tv',
  'tvs': 'tv',
})


def normalize_target(value: str) -> str:
  """Return a supported target name from a canonical name or known alias.

  :param value: Search target supplied by a user or source table.
  :raises ValueError: If the name is empty or unsupported.
  """
  if not isinstance(value, str):
    raise ValueError(f'Target must be text; received {value!r}.')
  cleaned = re.sub(r'[^a-z0-9]+', ' ', value.lower()).strip()
  target = _TARGET_ALIASES.get(cleaned)
  if target is None:
    raise ValueError(f'Unsupported target {value!r}. Choose one of: {", ".join(SUPPORTED_TARGETS)}.')
  return target


def _clean_identifier(value: object, column: str, row_index: object) -> str:
  if pd.isna(value):
    raise ValueError(f'Row {row_index}: {column} is missing.')
  if isinstance(value, (float, np.floating)) and float(value).is_integer():
    value = int(value)
  cleaned = str(value).strip()
  if not cleaned:
    raise ValueError(f'Row {row_index}: {column} is empty.')
  return cleaned


def validate_image_identity(frame: pd.DataFrame) -> None:
  """Reject conflicting physical IDs for the same resolved image path.

  Distinct subject-specific paths may share one physical image ID. Paths are
  resolved so relative paths and symlink aliases cannot evade this check.

  :param frame: Rows containing ``image_path`` and ``image_id``.
  :raises ValueError: If one resolved path maps to different image IDs.
  """
  missing = sorted({'image_path', 'image_id'} - set(frame.columns))
  if missing:
    raise ValueError(f'Image identity check needs columns: {", ".join(missing)}.')
  id_by_path: dict[str, str] = {}
  for row_index, path_value, id_value in zip(frame.index, frame['image_path'], frame['image_id']):
    image_path = str(Path(_clean_identifier(path_value, 'image_path', row_index)).expanduser().resolve())
    image_id = _clean_identifier(id_value, 'image_id', row_index)
    previous = id_by_path.get(image_path)
    if previous is not None and previous != image_id:
      raise ValueError(
        f'Image path {image_path} has conflicting image_id values {previous!r} and {image_id!r}; '
        'one physical image must stay in one validation group.'
      )
    id_by_path[image_path] = image_id


def validate_pairs(
  frame: pd.DataFrame,
  *,
  require_score: bool = False,
  image_root: Path | str | None = None,
  check_images: bool = True,
) -> pd.DataFrame:
  """Validate and canonicalize an image-target table without changing row order.

  Image paths become absolute. Relative paths are resolved under ``image_root``
  when supplied, otherwise under the current working directory. Training rows
  require ``image_id`` and a finite ``score``. A repeated
  ``(image_id, target, subject)`` training key is rejected. Each resolved
  image path must map to one physical image ID.

  :param frame: Input rows with at least ``image_path`` and ``target``.
  :param require_score: Require labeled training rows.
  :param image_root: Base for relative image paths.
  :param check_images: Require each path to name an existing file.
  :returns: A canonicalized copy, retaining any additional metadata columns.
  :raises ValueError: If rows do not satisfy the canonical schema.
  :raises FileNotFoundError: If an image is missing.
  """
  if not isinstance(frame, pd.DataFrame):
    raise TypeError('Image-target pairs must be a pandas DataFrame.')
  required = {'image_path', 'target'} | ({'image_id', 'score'} if require_score else set())
  missing = sorted(required - set(frame.columns))
  if missing:
    raise ValueError(f'Image-target table is missing required columns: {", ".join(missing)}.')
  if frame.empty:
    raise ValueError('Image-target table has no rows.')

  root = Path(image_root).expanduser().resolve() if image_root is not None else Path.cwd()
  result = frame.copy()
  paths: list[str] = []
  targets: list[str] = []
  for row_index, path_value, target_value in zip(frame.index, frame['image_path'], frame['target']):
    raw_path = _clean_identifier(path_value, 'image_path', row_index)
    path = Path(raw_path).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if check_images and not path.is_file():
      raise FileNotFoundError(f'Row {row_index}: image file does not exist: {path}')
    paths.append(str(path))
    try:
      targets.append(normalize_target(target_value))
    except ValueError as exc:
      raise ValueError(f'Row {row_index}: {exc}') from exc
  result['image_path'] = paths
  result['target'] = targets

  for column in ('image_id', 'subject'):
    if column in result:
      result[column] = [
        _clean_identifier(value, column, row_index)
        for row_index, value in zip(frame.index, frame[column])
      ]

  if 'score' in result:
    try:
      scores = pd.to_numeric(result['score'], errors='raise').astype(float)
    except (TypeError, ValueError) as exc:
      raise ValueError('Score values must be finite numbers.') from exc
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
      raise ValueError('Score values must be finite numbers.')
    result['score'] = scores

  if require_score:
    validate_image_identity(result)
    key_columns = ['image_id', 'target']
    if 'subject' in result:
      key_columns.append('subject')
    duplicates = result.duplicated(subset=key_columns, keep=False)
    if duplicates.any():
      sample = result.loc[duplicates, key_columns].iloc[0].to_dict()
      raise ValueError(f'Duplicate training key {sample!r}; expected one score per image-target-subject.')

  return result
