"""Checks for grouped fitting and complete model bundles."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from route_a.engineered.bundle import ModelBundle
from route_a.engineered.features import DETECTOR_FEATURE_NAMES, FeatureConfig
from route_a.engineered.preprocessing import FeaturePreprocessor
from route_a.engineered.schema import SUPPORTED_TARGETS
from route_a.engineered.training import TrainingConfig, make_grouped_folds, train_engineered


def _feature_frame() -> pd.DataFrame:
  rows: list[dict[str, object]] = []
  for image_index in range(6):
    for target in ('car', 'chair'):
      row: dict[str, object] = {
        'image_id': f'nsd-{image_index:05d}',
        'image_path': f'/unused/image-{image_index}.jpg',
        'target': target,
        'score': image_index * 0.15 + (0.25 if target == 'chair' else -0.1),
      }
      for name in SUPPORTED_TARGETS:
        row[f'task_{name.replace(" ", "_")}'] = float(name == target)
      for index in range(3):
        row[f'dino_{index:03d}'] = image_index * (index + 1) * 0.1
      for index in range(4):
        row[f'openclip_vit_l_14_{index:03d}'] = image_index * (index + 1) * 0.07
      for name in DETECTOR_FEATURE_NAMES:
        row[f'det_{name}'] = float(image_index % 3) * 0.2
      rows.append(row)
  return pd.DataFrame(rows)


def test_physical_image_is_never_split_across_folds() -> None:
  frame = _feature_frame()
  folds = make_grouped_folds(frame, TrainingConfig(cv_folds=3, n_estimators=3))
  assigned = pd.DataFrame({'image_id': frame['image_id'], 'fold': folds})
  assert len(set(folds)) == 3
  assert assigned.groupby('image_id')['fold'].nunique().eq(1).all()


def test_preprocessor_fits_pca_on_training_rows_only() -> None:
  frame = _feature_frame()
  training = frame.iloc[:10]
  validation = frame.iloc[10:].copy()
  validation['dino_000'] = 10000.0
  preprocessor = FeaturePreprocessor(pca_components=2).fit(training)
  assert preprocessor.pcas['dino'].mean_[0] == pytest.approx(training['dino_000'].mean())
  assert preprocessor.transform(validation).shape == (2, 16 + 26 + 2 + 2)


def test_complete_bundle_round_trip(tmp_path) -> None:
  frame = _feature_frame()
  feature_config = FeatureConfig(dino_dim=3, openclip_dim=4)
  result = train_engineered(frame, feature_config, TrainingConfig(cv_folds=3, n_estimators=5, pca_components=2))
  assert len(result.validation_predictions) == len(frame)
  assert result.validation_predictions['prediction'].notna().all()
  directory = result.bundle.save(tmp_path / 'bundle')
  restored = ModelBundle.load(directory)
  x_original = result.bundle.preprocessor.transform(frame)
  x_restored = restored.preprocessor.transform(frame)
  np.testing.assert_allclose(x_original, x_restored, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(result.bundle.model.predict(x_original), restored.model.predict(x_restored),
                             rtol=1e-6, atol=1e-6)
  metadata = json.loads((directory / 'metadata.json').read_text())
  assert metadata['software_versions']['python']
  assert metadata['model_feature_names'] == restored.preprocessor.output_columns


def test_training_rejects_wrong_extractor_dimensions() -> None:
  with pytest.raises(ValueError, match='Feature widths'):
    train_engineered(_feature_frame(), FeatureConfig(), TrainingConfig(cv_folds=0, n_estimators=2))
