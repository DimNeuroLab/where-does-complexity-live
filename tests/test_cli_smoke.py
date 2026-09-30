"""Small CLI workflow using a deterministic stand-in for pretrained encoders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from route_a.engineered import __main__ as cli
from route_a.engineered import predictor as predictor_module
from route_a.engineered.features import DETECTOR_FEATURE_NAMES, FeatureConfig
from route_a.engineered.schema import SUPPORTED_TARGETS, validate_pairs


class _StubExtractor:
  def __init__(self, config: FeatureConfig | None = None, cache_dir: str | Path | None = None) -> None:
    self.config = config or FeatureConfig()

  def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
    canonical = validate_pairs(pairs)
    rows: list[dict[str, float]] = []
    for image_path, target in zip(canonical['image_path'], canonical['target']):
      image_number = int(Path(image_path).stem.split('-')[-1])
      row: dict[str, float] = {}
      for name in SUPPORTED_TARGETS:
        row[f'task_{name.replace(" ", "_")}'] = float(name == target)
      for index in range(self.config.dino_dim):
        row[f'dino_{index:03d}'] = float(np.sin(image_number * 0.5 + index * 0.01))
      for index in range(self.config.openclip_dim):
        row[f'openclip_vit_l_14_{index:03d}'] = float(np.cos(image_number * 0.3 + index * 0.01))
      for name in DETECTOR_FEATURE_NAMES:
        row[f'det_{name}'] = float(image_number % 3) * 0.1
      rows.append(row)
    return pd.concat([canonical.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def test_train_then_predict_cli(tmp_path, monkeypatch, capsys) -> None:
  monkeypatch.setattr(cli, 'ImageTargetFeatureExtractor', _StubExtractor)
  monkeypatch.setattr(predictor_module, 'ImageTargetFeatureExtractor', _StubExtractor)
  image_root = tmp_path / 'images'
  image_root.mkdir()
  labels: list[dict[str, object]] = []
  for image_number in range(6):
    name = f'image-{image_number}.png'
    Image.new('RGB', (4, 4), color=(image_number * 20, 0, 0)).save(image_root / name)
    for target in ('car', 'chair'):
      labels.append({
        'image_id': f'nsd-{image_number:05d}',
        'image_path': name,
        'target': target,
        'score': image_number * 0.2 + (0.3 if target == 'chair' else 0),
      })
  labels_path = tmp_path / 'labels.csv'
  pd.DataFrame(labels).to_csv(labels_path, index=False)
  model_dir = tmp_path / 'model_bundle'
  assert cli.main([
    'train', '--data', str(labels_path), '--image-root', str(image_root), '--out', str(model_dir),
    '--n-estimators', '3', '--pca-components', '2', '--cv-folds', '2',
  ]) == 0
  assert (model_dir / 'model.json').is_file()
  assert (model_dir / 'preprocessing.joblib').is_file()
  assert (model_dir / 'metadata.json').is_file()
  capsys.readouterr()
  assert cli.main([
    'predict', '--model', str(model_dir), '--image', str(image_root / 'image-0.png'), '--target', 'chair',
  ]) == 0
  response = json.loads(capsys.readouterr().out)
  assert response['target'] == 'chair'
  assert np.isfinite(response['prediction'])
  pair_path = tmp_path / 'pairs.csv'
  pd.DataFrame([{'image_path': 'image-1.png', 'target': 'car'}]).to_csv(pair_path, index=False)
  predictions_path = tmp_path / 'predictions.csv'
  assert cli.main([
    'predict', '--model', str(model_dir), '--input', str(pair_path), '--image-root', str(image_root),
    '--out', str(predictions_path),
  ]) == 0
  predictions = pd.read_csv(predictions_path)
  assert len(predictions) == 1
  assert np.isfinite(predictions['prediction'].iloc[0])
  original_input = pair_path.read_text(encoding='utf-8')
  with pytest.raises(SystemExit, match='2'):
    cli.main([
      'predict', '--model', str(model_dir), '--input', str(pair_path), '--out', str(pair_path),
    ])
  assert pair_path.read_text(encoding='utf-8') == original_input
  with pytest.raises(SystemExit, match='2'):
    cli.main([
      'predict', '--model', str(model_dir), '--input', str(pair_path), '--out', str(predictions_path),
    ])
