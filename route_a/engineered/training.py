"""Grouped training and validation for the engineered stimulus readout."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from xgboost import XGBRegressor

from route_a.engineered.bundle import ModelBundle
from route_a.engineered.features import FeatureConfig, expected_feature_columns, feature_columns
from route_a.engineered.preprocessing import FeaturePreprocessor
from route_a.engineered.schema import normalize_target, validate_image_identity


@dataclass(frozen=True)
class TrainingConfig:
  """Settings for grouped validation and final XGBoost fitting."""

  pca_components: int = 64
  n_estimators: int = 300
  cv_folds: int = 5
  seed: int = 42
  learning_rate: float = 0.03
  max_depth: int = 6
  subsample: float = 0.8
  colsample_bytree: float = 0.8
  reg_lambda: float = 1.0
  n_jobs: int = 4


@dataclass
class TrainingResult:
  """Fitted bundle and optional held-out predictions."""

  bundle: ModelBundle
  validation_predictions: pd.DataFrame


def _validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
  required = {'image_id', 'image_path', 'target', 'score'}
  missing = required.difference(frame.columns)
  if missing:
    raise ValueError(f'Training data is missing columns: {sorted(missing)}')
  if frame.empty:
    raise ValueError('Training data has no rows.')
  data = frame.copy().reset_index(drop=True)
  data['image_id'] = data['image_id'].astype(str).str.strip()
  if data['image_id'].eq('').any() or data['image_id'].eq('nan').any():
    raise ValueError('Every training row needs a physical image_id.')
  validate_image_identity(data)
  data['target'] = data['target'].map(normalize_target)
  data['score'] = pd.to_numeric(data['score'], errors='raise')
  if not np.isfinite(data['score'].to_numpy(dtype=np.float64)).all():
    raise ValueError('Training scores must be finite.')
  key = ['image_id', 'target']
  if 'subject' in data.columns:
    key.insert(0, 'subject')
  if data.duplicated(key).any():
    raise ValueError(f'Duplicate training rows for key {key}.')
  return data


def make_grouped_folds(frame: pd.DataFrame, config: TrainingConfig) -> np.ndarray:
  """Assign each physical image to one validation fold.

  :param frame: Canonical training rows, with ``image_id`` and ``target``.
  :param config: Cross-validation settings.
  :returns: One-based fold index for each row.
  """
  if config.cv_folds < 2:
    raise ValueError('cv_folds must be at least 2 for grouped validation.')
  groups = frame['image_id'].astype(str).to_numpy()
  n_groups = pd.Series(groups).nunique()
  if n_groups < 2:
    raise ValueError('Grouped validation needs at least two physical images.')
  n_splits = min(config.cv_folds, n_groups)
  per_target_groups = frame.groupby('target')['image_id'].nunique()
  if per_target_groups.min() >= n_splits:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    splits = splitter.split(frame, frame['target'], groups=groups)
  else:
    splitter = GroupKFold(n_splits=n_splits)
    splits = splitter.split(frame, groups=groups)
  folds = np.zeros(len(frame), dtype=np.int32)
  for fold, (_, validation_index) in enumerate(splits, start=1):
    folds[validation_index] = fold
  if np.any(folds == 0):
    raise RuntimeError('Some training rows were not assigned to a validation fold.')
  if pd.DataFrame({'image_id': groups, 'fold': folds}).groupby('image_id')['fold'].nunique().gt(1).any():
    raise RuntimeError('Physical images crossed validation folds.')
  return folds


def _make_regressor(config: TrainingConfig) -> XGBRegressor:
  if config.n_estimators < 1:
    raise ValueError('n_estimators must be positive.')
  return XGBRegressor(
    objective='reg:absoluteerror',
    eval_metric='mae',
    n_estimators=config.n_estimators,
    learning_rate=config.learning_rate,
    max_depth=config.max_depth,
    subsample=config.subsample,
    colsample_bytree=config.colsample_bytree,
    reg_lambda=config.reg_lambda,
    random_state=config.seed,
    tree_method='hist',
    n_jobs=config.n_jobs,
  )


def train_engineered(
  frame: pd.DataFrame,
  feature_config: FeatureConfig | None = None,
  training_config: TrainingConfig | None = None,
) -> TrainingResult:
  """Fit and validate an image-and-target XGBoost model.

  :param frame: Canonical labeled rows with extracted raw feature columns.
  :param feature_config: Feature extraction settings saved for later inference.
  :param training_config: Preprocessing and model settings.
  :returns: A final full-data model bundle and grouped held-out predictions.
  """
  config = training_config or TrainingConfig()
  extractor_config = feature_config or FeatureConfig()
  data = _validate_training_frame(frame)
  raw_columns = feature_columns(data)
  dino_width = sum(column.startswith('dino_') for column in raw_columns)
  openclip_width = sum(column.startswith('openclip_') for column in raw_columns)
  if dino_width != extractor_config.dino_dim or openclip_width != extractor_config.openclip_dim:
    raise ValueError(
      f'Feature widths do not match the saved extractor config: DINO {dino_width}/{extractor_config.dino_dim}, '
      f'OpenCLIP {openclip_width}/{extractor_config.openclip_dim}.'
    )
  expected_columns = expected_feature_columns(extractor_config)
  if raw_columns != expected_columns:
    mismatch = next(index for index, (actual, expected) in enumerate(zip(raw_columns, expected_columns)) if actual != expected)
    raise ValueError(
      f'Raw feature schema does not match FeatureConfig at column {mismatch}: '
      f'expected {expected_columns[mismatch]!r}, found {raw_columns[mismatch]!r}.'
    )
  if config.pca_components < 1:
    raise ValueError('pca_components must be positive.')
  y = data['score'].to_numpy(dtype=np.float32)
  validation = pd.DataFrame()
  if config.cv_folds >= 2:
    folds = make_grouped_folds(data, config)
    predictions = np.full(len(data), np.nan, dtype=np.float32)
    for fold in sorted(np.unique(folds)):
      train_index = np.flatnonzero(folds != fold)
      validation_index = np.flatnonzero(folds == fold)
      preprocessor = FeaturePreprocessor(config.pca_components, config.seed)
      x_train = preprocessor.fit_transform(data.iloc[train_index])
      x_validation = preprocessor.transform(data.iloc[validation_index])
      model = _make_regressor(config)
      model.fit(x_train, y[train_index], verbose=False)
      predictions[validation_index] = model.predict(x_validation)
    if not np.isfinite(predictions).all():
      raise RuntimeError('Held-out prediction is missing for at least one row.')
    metadata_columns = [column for column in ('image_id', 'image_path', 'subject', 'target') if column in data]
    validation = data[metadata_columns].copy()
    validation['score'] = y
    validation['prediction'] = predictions
    validation['fold'] = folds
  final_preprocessor = FeaturePreprocessor(config.pca_components, config.seed)
  x_all = final_preprocessor.fit_transform(data)
  final_model = _make_regressor(config)
  final_model.fit(x_all, y, verbose=False)
  training_info = {
    'n_rows': len(data),
    'n_physical_images': int(data['image_id'].nunique()),
    'training_config': asdict(config),
  }
  if not validation.empty:
    training_info['validation_mae'] = float(mean_absolute_error(y, validation['prediction']))
    training_info['validation_folds'] = int(validation['fold'].nunique())
  bundle = ModelBundle(final_model, final_preprocessor, extractor_config, training_info)
  return TrainingResult(bundle, validation)
