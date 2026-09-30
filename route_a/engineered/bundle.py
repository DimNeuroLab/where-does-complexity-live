"""Portable model bundle for engineered Route A inference."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
from xgboost import XGBRegressor

from route_a.engineered.features import FeatureConfig
from route_a.engineered.preprocessing import FeaturePreprocessor
from route_a.engineered.schema import SUPPORTED_TARGETS


BUNDLE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _software_versions() -> dict[str, str]:
  versions = {'python': platform.python_version()}
  for distribution in ('where-does-complexity-live', 'numpy', 'pandas', 'scikit-learn', 'xgboost',
                       'torch', 'torchvision', 'timm', 'open-clip-torch'):
    try:
      versions[distribution] = version(distribution)
    except PackageNotFoundError:
      continue
  return versions


@dataclass
class ModelBundle:
  """Model, fitted transforms, and the feature configuration they require."""

  model: XGBRegressor
  preprocessor: FeaturePreprocessor
  feature_config: FeatureConfig
  training_info: dict[str, Any] = field(default_factory=dict)

  def save(self, directory: str | Path) -> Path:
    """Write the complete model bundle to a directory.

    :param directory: Destination for ``model.json``, ``preprocessing.joblib``, and ``metadata.json``.
    :returns: The bundle directory.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    if not self.preprocessor.output_columns:
      raise ValueError('Cannot save a bundle with an unfitted preprocessor.')
    model_path = destination / 'model.json'
    preprocessing_path = destination / 'preprocessing.joblib'
    self.model.save_model(model_path)
    joblib.dump(self.preprocessor, preprocessing_path)
    metadata = {
      'schema_version': BUNDLE_SCHEMA_VERSION,
      'created_at_utc': datetime.now(timezone.utc).isoformat(),
      'model_kind': 'engineered_xgboost',
      'feature_config': asdict(self.feature_config),
      'targets': list(SUPPORTED_TARGETS),
      'model_feature_names': self.preprocessor.output_columns,
      'raw_feature_columns': self.preprocessor.columns,
      'training_info': self.training_info,
      'software_versions': _software_versions(),
      'files_sha256': {
        model_path.name: _sha256(model_path),
        preprocessing_path.name: _sha256(preprocessing_path),
      },
    }
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return destination

  @classmethod
  def load(cls, directory: str | Path) -> ModelBundle:
    """Load and check a trusted model bundle.

    :param directory: Bundle created by :meth:`save`.
    :returns: Loaded bundle ready for inference.
    :raises ValueError: If the bundle format or file hashes are invalid.
    """
    source = Path(directory)
    metadata_path = source / 'metadata.json'
    model_path = source / 'model.json'
    preprocessing_path = source / 'preprocessing.joblib'
    for path in (metadata_path, model_path, preprocessing_path):
      if not path.is_file():
        raise FileNotFoundError(f'Model bundle is missing {path}.')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('schema_version') != BUNDLE_SCHEMA_VERSION:
      raise ValueError(f'Unsupported model bundle schema version: {metadata.get("schema_version")!r}.')
    if metadata.get('model_kind') != 'engineered_xgboost':
      raise ValueError('This bundle is not an engineered Route A XGBoost model.')
    if metadata.get('targets') != list(SUPPORTED_TARGETS):
      raise ValueError('Model target vocabulary differs from this package.')
    for path in (model_path, preprocessing_path):
      expected = metadata.get('files_sha256', {}).get(path.name)
      if expected != _sha256(path):
        raise ValueError(f'Model bundle file checksum failed: {path.name}.')
    preprocessor = joblib.load(preprocessing_path)
    if not isinstance(preprocessor, FeaturePreprocessor):
      raise ValueError('Model bundle preprocessing object has an unexpected type.')
    if preprocessor.output_columns != metadata.get('model_feature_names'):
      raise ValueError('Model bundle feature names differ from preprocessing metadata.')
    model = XGBRegressor()
    model.load_model(model_path)
    if model.get_booster().num_features() != len(preprocessor.output_columns):
      raise ValueError('Model bundle model width differs from preprocessing width.')
    feature_config = FeatureConfig(**metadata['feature_config'])
    return cls(
      model=model,
      preprocessor=preprocessor,
      feature_config=feature_config,
      training_info=metadata.get('training_info', {}),
    )
