"""Adapters from the original COCO and NSD tables to canonical training rows."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from route_a.engineered.schema import normalize_target, validate_pairs


IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'})
_NSD_ID_PATTERN = re.compile(r'nsd-(\d+)', re.IGNORECASE)
_SUBJECT_PATTERN = re.compile(r'^subj0*(\d+)$', re.IGNORECASE)


def _require_file(path: Path | str, description: str) -> Path:
  resolved = Path(path).expanduser().resolve()
  if not resolved.is_file():
    raise FileNotFoundError(f'{description} does not exist: {resolved}')
  return resolved


def _require_root(path: Path | str) -> Path:
  resolved = Path(path).expanduser().resolve()
  if not resolved.is_dir():
    raise FileNotFoundError(f'Image root does not exist: {resolved}')
  return resolved


def _normalize_subject(value: object) -> str:
  if pd.isna(value) or not str(value).strip():
    raise ValueError('NSD subject is missing.')
  text = str(value).strip().lower()
  match = _SUBJECT_PATTERN.fullmatch(text)
  return f'subj{int(match.group(1)):02d}' if match else text


def _extract_nsd_id(value: object) -> int:
  if pd.isna(value):
    raise ValueError('NSD image ID is missing.')
  text = str(value).strip()
  match = _NSD_ID_PATTERN.search(text)
  if match:
    return int(match.group(1))
  if re.fullmatch(r'\d+(?:\.0)?', text):
    return int(float(text))
  raise ValueError(f'Cannot read NSD image ID from {value!r}.')


def _iter_images(root: Path) -> list[Path]:
  images = sorted(path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
  if not images:
    raise ValueError(f'No supported image files found under {root}.')
  return images


def _coco_image_id(stem: str, target: str, source_task: str) -> str:
  nsd_match = _NSD_ID_PATTERN.search(stem)
  if nsd_match:
    return f'nsd-{int(nsd_match.group(1)):05d}'
  for label in (target, source_task):
    variants = {label.lower().replace(' ', separator) for separator in (' ', '_', '-', '')}
    for variant in sorted(variants, key=len, reverse=True):
      for separator in ('_', '-', ' '):
        suffix = separator + variant
        if stem.lower().endswith(suffix) and len(stem) > len(suffix):
          return stem[:-len(suffix)]
  return stem


def load_coco_pairs(csv_path: Path | str, image_root: Path | str) -> pd.DataFrame:
  """Load the COCO ranking CSV used for Route A training.

  The source columns are ``image``, ``task``, and ``score``; the supplied
  ``score`` becomes the training label. COCO's ``bowl`` and ``knife`` rows
  are deliberately excluded because the 16-target model
  does not include them. Other unknown target names are errors. Every retained
  image must resolve uniquely under ``image_root``.

  :param csv_path: Source ranking CSV.
  :param image_root: Explicit root of COCO-Search18 images.
  :returns: ``image_id``, ``image_path``, ``target``, ``score`` rows.
  """
  source = _require_file(csv_path, 'COCO ranking CSV')
  root = _require_root(image_root)
  frame = pd.read_csv(source)
  missing = sorted({'image', 'task', 'score'} - set(frame.columns))
  if missing:
    raise ValueError(f'COCO ranking CSV is missing columns: {", ".join(missing)}.')

  image_index: dict[str, list[Path]] = {}
  for path in _iter_images(root):
    image_index.setdefault(path.name, []).append(path)

  rows: list[dict[str, object]] = []
  for row_index, record in frame.iterrows():
    source_task = str(record['task']).strip() if pd.notna(record['task']) else ''
    cleaned_task = re.sub(r'[^a-z0-9]+', ' ', source_task.lower()).strip()
    if cleaned_task in {'bowl', 'knife'}:
      continue
    try:
      target = normalize_target(source_task)
    except ValueError as exc:
      raise ValueError(f'COCO row {row_index}: {exc}') from exc
    if pd.isna(record['image']) or not str(record['image']).strip():
      raise ValueError(f'COCO row {row_index}: image name is missing.')
    raw_image = Path(str(record['image']).strip())
    filename = raw_image.name
    candidate = raw_image if raw_image.is_absolute() else root / raw_image
    if candidate.is_file():
      resolved = candidate.resolve()
    else:
      direct = [root / target / filename, root / source_task / filename, root / filename]
      found = next((path.resolve() for path in direct if path.is_file()), None)
      if found is not None:
        resolved = found
      else:
        matches = image_index.get(filename, [])
        if len(matches) > 1:
          target_names = {target, cleaned_task}
          target_matches = [
            path for path in matches
            if re.sub(r'[^a-z0-9]+', ' ', path.parent.name.lower()).strip() in target_names
          ]
          matches = target_matches if target_matches else matches
        if len(matches) != 1:
          reason = 'missing' if not matches else 'ambiguous'
          raise ValueError(f'COCO row {row_index}: image {filename!r} is {reason} under {root}.')
        resolved = matches[0].resolve()
    rows.append({
      'image_id': _coco_image_id(raw_image.stem, target, source_task),
      'image_path': str(resolved),
      'target': target,
      'score': record['score'],
    })

  if not rows:
    raise ValueError('COCO ranking CSV has no rows for the supported 16 targets.')
  return validate_pairs(pd.DataFrame(rows), require_score=True)


def load_nsd_pairs(
  inventory_path: Path | str,
  image_root: Path | str,
  subject: str | None = None,
) -> pd.DataFrame:
  """Load labeled rows from the original NSD image inventory.

  ``complexity_score`` becomes the training label; rows with
  ``has_complexity`` false are excluded. The subject is retained. Physical NSD
  images share one ``image_id`` across subjects for grouped validation.
  Image filenames must contain ``nsd-N`` or have a numeric stem. If an NSD
  image appears in more than one path, the subject directory resolves it.

  :param inventory_path: Inventory CSV with NSD IDs and complexity labels.
  :param image_root: Explicit root containing subject image directories.
  :param subject: Optional subject filter such as ``subj01``.
  :returns: Canonical rows including ``subject``.
  """
  source = _require_file(inventory_path, 'NSD inventory CSV')
  root = _require_root(image_root)
  frame = pd.read_csv(source)
  required = {'nsd_id', 'subject', 'complexity_category', 'complexity_score', 'has_complexity'}
  missing = sorted(required - set(frame.columns))
  if missing:
    raise ValueError(f'NSD inventory CSV is missing columns: {", ".join(missing)}.')

  has_complexity = frame['has_complexity'].map(
    lambda value: str(value).strip().lower() in {'true', '1', 'yes', 'y'}
  )
  selected = frame.loc[has_complexity].copy()
  if subject is not None:
    wanted_subject = _normalize_subject(subject)
    selected = selected.loc[selected['subject'].map(_normalize_subject) == wanted_subject]
  if selected.empty:
    raise ValueError('NSD inventory has no matching rows with complexity labels.')

  image_index: dict[int, list[tuple[Path, str | None]]] = {}
  for path in _iter_images(root):
    try:
      nsd_id = _extract_nsd_id(path.stem)
    except ValueError:
      continue
    path_subject = next(
      (_normalize_subject(part) for part in path.parts if _SUBJECT_PATTERN.fullmatch(part)),
      None,
    )
    image_index.setdefault(nsd_id, []).append((path, path_subject))

  rows: list[dict[str, object]] = []
  for row_index, record in selected.iterrows():
    try:
      nsd_id = _extract_nsd_id(record['nsd_id'])
      row_subject = _normalize_subject(record['subject'])
      target = normalize_target(record['complexity_category'])
    except ValueError as exc:
      raise ValueError(f'NSD inventory row {row_index}: {exc}') from exc
    candidates = image_index.get(nsd_id, [])
    subject_candidates = [path for path, path_subject in candidates if path_subject == row_subject]
    matching = subject_candidates if subject_candidates else [path for path, _ in candidates]
    if len(matching) != 1:
      reason = 'missing' if not matching else 'ambiguous'
      raise ValueError(f'NSD inventory row {row_index}: image nsd-{nsd_id:05d} is {reason} under {root}.')
    rows.append({
      'image_id': f'nsd-{nsd_id:05d}',
      'image_path': str(matching[0].resolve()),
      'target': target,
      'score': record['complexity_score'],
      'subject': row_subject,
    })

  return validate_pairs(pd.DataFrame(rows), require_score=True)
