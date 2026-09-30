"""Record standalone run inputs, software, and completed-stage fingerprints."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from route_b.config import RunConfig, paths
from route_b.runtime import atomic_json, digest


def source_hashes() -> dict[str, str]:
  root = Path(__file__).resolve().parent
  return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*.py'))
          if '.venv' not in path.parts and '__pycache__' not in path.parts}


def freeze(config: RunConfig) -> None:
  """Create or verify a manifest without an original repository or saved experiment."""
  destination = paths().output / 'manifest.json'
  packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
  current = {'schema': 2, 'config': config, 'source_sha256': source_hashes(),
             'python': sys.version, 'packages': packages}
  if destination.exists():
    if json.loads(destination.read_text()) != current:
      raise ValueError('Run source, configuration, or environment changed; use a separate output directory')
  else:
    atomic_json(destination, current)
    (paths().output / 'requirements.freeze.txt').write_text(packages)


def input_files(stage: str) -> list[Path]:
  """Identify data dependencies without requiring artifacts of unrelated stages."""
  from route_b.constants import NSD_SUBJECTS
  p = paths()
  files = [p.ranking]
  for subject in NSD_SUBJECTS:
    base = p.nsd / subject
    images = sorted((base / 'training_split/training_images').glob('*.png'))
    if stage in ('features', 'pretrain', 'train', 'evaluate', 'variance'):
      files.extend(images)
    if stage in ('prepare', 'transform'):
      files.extend(base / f'training_split/training_fmri/{hemisphere}_training_fmri.npy' for hemisphere in ('lh', 'rh'))
      files.extend(base / f'roi_masks/{hemisphere}.streams_challenge_space.npy' for hemisphere in ('lh', 'rh'))
    if stage in ('pretrain', 'train', 'evaluate', 'variance'):
      files.append(p.pca / f'{subject}_pca_fmri.npy')
    if stage == 'pretrain':
      files.extend(p.pca / f'{subject}_{split}_idx.npy' for split in ('train', 'val'))
    if stage in ('pretrain', 'train'):
      files.extend([p.dino / f'{subject}_dino_layers.npy', p.clip_image / f'{subject}_clip_img.npy'])
    if stage == 'transform':
      files.append((p.pca_models or p.pca) / f'{subject}_pca_models.pkl')
  if stage in ('train', 'evaluate', 'variance'):
    files.append(p.clip_text / 'clip_text_embeddings.npz')
  return files


def fingerprints(files: list[Path]) -> dict[str, dict[str, int | str]]:
  """Hash scientific inputs and retain file metadata for inexpensive resume checks."""
  return {str(path): {'sha256': digest(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
          for path in files}


def verify_files(saved: dict[str, dict[str, int | str]]) -> None:
  for name, expected in saved.items():
    path = Path(name)
    stat = path.stat()
    if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
      if digest(path) != expected['sha256']:
        raise ValueError(f'Run artifact changed: {path}')
