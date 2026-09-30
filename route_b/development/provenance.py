"""Freeze the original source, ported source, environment, and experiment inputs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from route_b.config import RunConfig, paths
from route_b.constants import NSD_SUBJECTS
from route_b.data.datasets import complexity_stratified_group_kfold, load_complexity_records
from route_b.data.nsd import build_nsd_id_index
from route_b.evaluation.reproduction import REFERENCE_HEADS
from route_b.runtime import atomic_json, digest


def freeze(config_path: Path, config: RunConfig) -> None:
  """Create an authoritative source snapshot before the first GPU worker starts."""
  root = paths().output
  root.mkdir(parents=True, exist_ok=True)
  if (root / 'manifest.json').exists():
    verify(config_path, config)
    return
  original = Path(config['original_repository']).resolve()
  publication = Path(__file__).resolve().parents[2]
  snapshot = root / 'source_snapshot'
  source_files = subprocess.check_output(
    ['git', '-C', str(original), 'ls-tree', '-r', '--name-only', 'HEAD', 'src', 'scripts'], text=True).splitlines()
  for name in source_files:
    target = snapshot / 'original' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original / name, target)
  shutil.copy2(paths().ranking, snapshot / 'original/full_ranking_M2_n_studentT.csv')
  link = snapshot / 'original/outputs'
  if not link.exists():
    link.symlink_to(original / 'outputs', target_is_directory=True)
  shutil.copytree(publication / 'route_b', snapshot / 'publication/route_b', dirs_exist_ok=True,
                  ignore=shutil.ignore_patterns('.venv', '__pycache__'))
  target = snapshot / 'publication/complexity/evaluation'
  target.mkdir(parents=True, exist_ok=True)
  shutil.copy2(publication / 'complexity/evaluation/compute_variance.py', target / 'compute_variance.py')
  shutil.copy2(config_path, root / 'config.json') if config_path.resolve() != root / 'config.json' else None
  requirements = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
  (root / 'requirements.freeze.txt').write_text(requirements)
  inputs = [paths().ranking]
  for name in ('analysis_metrics.json', 'routeb_phase3_sweep_metrics.json'):
    reference = original / 'outputs' / name
    inputs.append(reference)
    (root / 'reference').mkdir(exist_ok=True)
    shutil.copy2(reference, root / 'reference' / name)
  reference_pca = Path(config['inputs']['reference_pca'])
  checkpoints = Path(config['inputs']['reference_checkpoints'])
  for name in set(REFERENCE_HEADS.values()) | {'routeb_pretrained_aux0_frozen.pt'}:
    inputs.append(checkpoints / 'complexity' / name)
  inputs.extend(sorted((checkpoints / 'feature_decoder').glob('*.pt')))
  for subject in NSD_SUBJECTS:
    inputs.extend([paths().dino / f'{subject}_dino_layers.npy', paths().clip_image / f'{subject}_clip_img.npy'])
    inputs.extend(sorted(reference_pca.glob(f'{subject}_*')))
    for hemi in ('lh', 'rh'):
      inputs.extend([paths().nsd / subject / f'training_split/training_fmri/{hemi}_training_fmri.npy',
                     paths().nsd / subject / f'roi_masks/{hemi}.streams_challenge_space.npy'])
  inputs.append(paths().clip_text / 'clip_text_embeddings.npz')
  input_info = {}
  for index, path in enumerate(inputs, 1):
    print(f'Fingerprint input {index}/{len(inputs)}: {path.name}', flush=True)
    stat = path.stat()
    input_info[str(path.resolve())] = {'sha256': digest(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
  code = {str(path.relative_to(snapshot)): digest(path) for path in sorted(snapshot.rglob('*'))
           if path.is_file() and '__pycache__' not in path.parts and 'outputs' not in path.relative_to(snapshot).parts}
  records = load_complexity_records()
  indices = {id(record): index for index, record in enumerate(records)}
  folds = [[indices[id(record)] for record in validation]
           for _, validation in complexity_stratified_group_kfold(records, 5, 42)]
  rows = {subject: build_nsd_id_index(subject) for subject in NSD_SUBJECTS}
  atomic_json(root / 'manifest.json', {
    'created': time.time(), 'config_sha256': digest(root / 'config.json'), 'code_sha256': code,
    'input_files': input_info, 'subject_rows': rows, 'fold_record_indices': folds,
    'source_git_commit': subprocess.check_output(['git', '-C', str(original), 'rev-parse', 'HEAD'], text=True).strip(),
    'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
    'environment_sha256': digest(root / 'requirements.freeze.txt'),
    'baseline_settings_basis': 'Tensor-identical auxiliary-zero checkpoint configuration; encoder defaults where unrecorded.',
    'gpu_budget_includes': 'validation, baseline, four sweep points, and prediction variance',
  })


def verify(config_path: Path, config: RunConfig) -> None:
  """Reject modified code, settings, packages, or experiment inputs on resume."""
  import hashlib
  root = Path(config['output']).resolve()
  manifest = json.loads((root / 'manifest.json').read_text())
  if digest(config_path) != manifest['config_sha256']:
    raise ValueError('Configuration changed; use a separate output directory')
  if sys.version != manifest['python'] or torch.__version__ != manifest['torch']:
    raise ValueError('Python or PyTorch changed')
  requirements = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
  if hashlib.sha256(requirements.encode()).hexdigest() != manifest['environment_sha256']:
    raise ValueError('Installed packages changed')
  for name, expected in manifest['code_sha256'].items():
    if digest(root / 'source_snapshot' / name) != expected:
      raise ValueError(f'Frozen implementation changed: {name}')
  for name, expected in manifest['input_files'].items():
    path = Path(name)
    stat = path.stat()
    if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
      if digest(path) != expected['sha256']:
        raise ValueError(f'Preserved input changed: {path}')
  rows = {subject: build_nsd_id_index(subject) for subject in NSD_SUBJECTS}
  if json.loads(json.dumps(rows)) != manifest['subject_rows']:
    raise ValueError('Subject image to fMRI row mapping changed')
