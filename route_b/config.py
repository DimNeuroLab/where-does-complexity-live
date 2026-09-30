"""Explicit input and output configuration without directory creation on import."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict


class Inputs(TypedDict):
  nsd_root: NotRequired[str]
  features_dir: NotRequired[str]
  pca_dir: NotRequired[str]
  checkpoints_dir: NotRequired[str]
  pca_models_dir: NotRequired[str]
  ranking_file: NotRequired[str]


class RunConfig(TypedDict):
  inputs: Inputs
  output_dir: str
  devices: list[int]
  gpu_hours: float
  cpu_threads: int
  dino_root: NotRequired[str]
  encoder: NotRequired[EncoderSettings]
  head: NotRequired[HeadSettings]


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
  cv_folds: int


@dataclass(frozen=True)
class Paths:
  """Resolved locations for one invocation, with shared inputs kept external."""

  nsd: Path | None
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


def configured_path(value: str, base: Path) -> str:
  """Expand a configured path relative to the JSON file, independently of cwd."""
  if not value:
    raise ValueError('Configured paths must not be empty')
  expanded = os.path.expandvars(value)
  if '$' in expanded:
    raise ValueError(f'Undefined environment variable in configured path: {value}')
  path = Path(expanded).expanduser()
  return str((path if path.is_absolute() else base / path).resolve())


def read_config(filename: Path, stages: tuple[str, ...] | None = None) -> RunConfig:
  """Read normalized settings and require only paths used by selected stages."""
  config: RunConfig = json.loads(filename.read_text())
  selected = set(stages) if stages is not None else {
    'features', 'prepare', 'pretrain', 'train', 'evaluate', 'sweep', 'variance'
  }
  if 'sweep' in selected:
    selected.update(('pretrain', 'train', 'evaluate'))
  allowed = {'inputs', 'output_dir', 'devices', 'gpu_hours', 'cpu_threads', 'dino_root', 'encoder', 'head'}
  if unknown := config.keys() - allowed:
    raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
  inputs = config.setdefault('inputs', {})
  if unknown := inputs.keys() - Inputs.__annotations__.keys():
    raise ValueError(f'Unknown input fields: {sorted(unknown)}')
  if 'n_folds' in config.get('head', {}):
    raise ValueError('Use head.cv_folds for the cross-validation fold count')
  base = filename.resolve().parent
  config.setdefault('devices', [0])
  config.setdefault('cpu_threads', 8)
  config.setdefault('gpu_hours', 72)
  config['output_dir'] = configured_path(config['output_dir'], base)
  inputs.setdefault('features_dir', str(Path(config['output_dir']) / 'features'))
  inputs.setdefault('ranking_file', str(Path(__file__).resolve().parents[1] / 'complexity/nsd_m2_ranking.csv'))
  required = {'nsd_root'} if selected - {'text'} else set()
  if selected & {'pretrain', 'train', 'evaluate', 'variance', 'prepare', 'transform'}:
    required.add('ranking_file')
  for key in required:
    if not inputs.get(key):
      raise ValueError(f'Missing input for selected stages: {key}')
  for key, value in list(inputs.items()):
    try:
      inputs[key] = configured_path(value, base)
    except ValueError:
      used = key in required or (
        key == 'features_dir' and bool(selected & {'features', 'text', 'pretrain', 'train', 'evaluate', 'variance'})
      ) or (key in {'pca_dir', 'checkpoints_dir'} and bool(selected - {'text', 'features'})) or (
        key == 'pca_models_dir' and 'transform' in selected
      )
      if used:
        raise
      del inputs[key]
  if config['gpu_hours'] <= 0 or not config['devices'] or len(set(config['devices'])) != len(config['devices']):
    raise ValueError('GPU hours must be positive and devices must be nonempty and unique')
  if 'head' in config and config['head']['cv_folds'] != 5:
    raise ValueError('The paper reproduction requires five folds')
  if config.get('dino_root'):
    try:
      config['dino_root'] = configured_path(config['dino_root'], base)
    except ValueError:
      if 'features' in selected:
        raise
      del config['dino_root']
  return config


def configure(filename: Path, stages: tuple[str, ...] | None = None) -> RunConfig:
  """Resolve external inputs separately from writable run products."""
  global _paths
  config = read_config(filename, stages)
  inputs = config['inputs']
  output = Path(config['output_dir'])
  features = Path(inputs.get('features_dir', output / 'features'))
  _paths = Paths(
    nsd=Path(inputs['nsd_root']) if 'nsd_root' in inputs else None,
    dino=features / 'dino', clip_image=features / 'clip_image', clip_text=features / 'clip_text',
    pca=Path(inputs.get('pca_dir', output / 'pca')),
    checkpoints=Path(inputs.get('checkpoints_dir', output / 'checkpoints')),
    ranking=Path(inputs['ranking_file']), output=output,
    dino_repository=config.get('dino_root', ''),
    pca_models=Path(inputs['pca_models_dir']) if 'pca_models_dir' in inputs else None,
  )
  protected = [_paths.ranking]
  if _paths.nsd is not None:
    protected.append(_paths.nsd)
  if _paths.pca_models is not None:
    protected.append(_paths.pca_models)
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
