"""Explicit input and output configuration without directory creation on import."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict


class Inputs(TypedDict):
  nsd: str
  features: str
  reference_pca: NotRequired[str]
  reference_checkpoints: NotRequired[str]
  pca: NotRequired[str]
  checkpoints: NotRequired[str]
  pca_models: NotRequired[str]
  ranking: str


class RunConfig(TypedDict):
  inputs: Inputs
  output: str
  devices: list[int]
  gpu_hours: float
  cpu_threads: int
  original_repository: NotRequired[str]
  dino_repository: NotRequired[str]
  encoder: EncoderSettings
  head: HeadSettings


class EncoderSettings(TypedDict):
  epochs: int
  batch_size: int
  lr: float
  patience: int
  dino_loss_weight: float
  clip_loss_weight: float
  dino_loss_reduction: str


class HeadSettings(TypedDict):
  epochs: int
  batch_size: int
  lr: float
  patience: int
  freeze_encoder_epochs: int
  aux_weight: float
  n_folds: int


@dataclass(frozen=True)
class Paths:
  """Resolved locations for one invocation, with shared inputs kept external."""

  nsd: Path
  dino: Path
  clip_image: Path
  clip_text: Path
  pca: Path
  checkpoints: Path
  ranking: Path
  output: Path
  dino_repository: str = ''
  pca_models: Path | None = None


_paths: Paths | None = None


def read_config(filename: Path) -> RunConfig:
  """Read a JSON configuration and reject missing input or run fields."""
  config: RunConfig = json.loads(filename.read_text())
  config.setdefault('devices', [0])
  config.setdefault('cpu_threads', 8)
  config.setdefault('gpu_hours', 72)
  config['output'] = str(Path(os.path.expandvars(config['output'])).expanduser().resolve())
  if '$' in config['output']:
    raise ValueError('Define the output environment variable or supply an explicit path')
  config['inputs'].setdefault('features', str(Path(config['output']) / 'features'))
  config['inputs'].setdefault('ranking', str(Path(__file__).resolve().parents[1] / 'complexity/nsd_m2_ranking.csv'))
  for key, value in config['inputs'].items():
    if '$' in os.path.expandvars(value):
      raise ValueError(f'Undefined environment variable in input {key}: {value}')
    config['inputs'][key] = str(Path(os.path.expandvars(value)).expanduser().resolve())
  for key in ('nsd', 'features', 'ranking'):
    if not config['inputs'].get(key):
      raise ValueError(f'Missing input: {key}')
  if config['gpu_hours'] <= 0 or not config['devices'] or len(set(config['devices'])) != len(config['devices']):
    raise ValueError('GPU hours must be positive and devices must be nonempty and unique')
  if config['head']['n_folds'] != 5:
    raise ValueError('The paper reproduction requires five folds')
  if '$' in os.path.expandvars(config.get('dino_repository', '')):
    raise ValueError('Define DINOV2_ROOT or omit dino_repository to use the pinned Torch Hub revision')
  return config


def configure(filename: Path, reference: bool = False) -> RunConfig:
  """Select original or newly fitted PCA/checkpoints before importing data modules."""
  global _paths
  config = read_config(filename)
  inputs = config['inputs']
  output = Path(config['output']).expanduser().resolve()
  features = Path(inputs['features']).expanduser().resolve()
  _paths = Paths(
    nsd=Path(inputs['nsd']).expanduser().resolve(), dino=features / 'dino',
    clip_image=features / 'clip_image', clip_text=features / 'clip_text',
    pca=Path(inputs['reference_pca']) if reference else Path(inputs.get('pca', output / 'pca')),
    checkpoints=(Path(inputs['reference_checkpoints']) if reference
                 else Path(inputs.get('checkpoints', output / 'checkpoints'))),
    ranking=Path(inputs['ranking']).expanduser().resolve(), output=output,
    dino_repository=os.path.expandvars(config.get('dino_repository', '')),
    pca_models=Path(inputs['pca_models']) if 'pca_models' in inputs else None,
  )
  protected = [_paths.nsd, _paths.ranking]
  protected.extend(Path(inputs[key]) for key in ('reference_pca', 'reference_checkpoints') if key in inputs)
  protected.extend(location for location in (features, _paths.pca, _paths.checkpoints)
                   if not location.is_relative_to(output))
  if any(output == location or output.is_relative_to(location) or location.is_relative_to(output) for location in protected):
    raise ValueError('Output directory must be separate from preserved inputs')
  return config


def paths() -> Paths:
  """Require callers to configure paths explicitly before accessing any data."""
  if _paths is None:
    raise RuntimeError('Configure Route B paths before accessing data')
  return _paths
