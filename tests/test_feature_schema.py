"""Lightweight contract tests for engineered Route A feature inputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from route_a.engineered.datasets import load_coco_pairs, load_nsd_pairs
from route_a.engineered.features import (
  DETECTOR_FEATURE_NAMES,
  FeatureConfig,
  ImageTargetFeatureExtractor,
  _ImageFeatures,
  expected_feature_columns,
  feature_columns,
)
from route_a.engineered.schema import SUPPORTED_TARGETS, normalize_target, validate_pairs
from route_a.engineered.training import TrainingConfig, _validate_training_frame, train_engineered


class SchemaTests(unittest.TestCase):
  def test_targets_and_aliases(self) -> None:
    self.assertEqual(len(SUPPORTED_TARGETS), 16)
    self.assertEqual(normalize_target('Potted-Plant'), 'potted plant')
    self.assertEqual(normalize_target('television'), 'tv')
    self.assertEqual(normalize_target('stop_sign'), 'stop sign')
    with self.assertRaisesRegex(ValueError, 'Unsupported target'):
      normalize_target('bowl')

  def test_training_validation_resolves_paths_and_rejects_duplicates(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'scene.jpg').write_bytes(b'fixture')
      frame = pd.DataFrame({
        'image_id': ['one'],
        'image_path': ['scene.jpg'],
        'target': ['Chair'],
        'score': [0.3],
        'subject': ['subj01'],
      })
      validated = validate_pairs(frame, require_score=True, image_root=root)
      self.assertEqual(validated.loc[0, 'image_path'], str((root / 'scene.jpg').resolve()))
      self.assertEqual(validated.loc[0, 'target'], 'chair')
      self.assertEqual(frame.loc[0, 'target'], 'Chair')
      with self.assertRaisesRegex(ValueError, 'Duplicate training key'):
        validate_pairs(pd.concat((frame, frame), ignore_index=True), require_score=True, image_root=root)
      with self.assertRaisesRegex(ValueError, 'finite numbers'):
        validate_pairs(frame.assign(score=[float('nan')]), require_score=True, image_root=root)

  def test_one_resolved_path_cannot_have_multiple_training_ids(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      first_path = root / 'scene.jpg'
      first_path.write_bytes(b'fixture')
      alias_path = root / 'alias.jpg'
      alias_path.symlink_to(first_path)
      conflicting = pd.DataFrame({
        'image_id': ['scene-a', 'scene-b'],
        'image_path': [str(first_path), str(alias_path)],
        'target': ['chair', 'cup'],
        'score': [0.1, 0.2],
      })
      with self.assertRaisesRegex(ValueError, 'conflicting image_id'):
        validate_pairs(conflicting, require_score=True)
      with self.assertRaisesRegex(ValueError, 'conflicting image_id'):
        _validate_training_frame(conflicting)

      second_path = root / 'subject-copy.jpg'
      second_path.write_bytes(b'fixture')
      shared_id = conflicting.assign(
        image_id=['scene-a', 'scene-a'],
        image_path=[str(first_path), str(second_path)],
        subject=['subj01', 'subj02'],
        target=['chair', 'chair'],
      )
      self.assertEqual(len(validate_pairs(shared_id, require_score=True)), 2)
      self.assertEqual(len(_validate_training_frame(shared_id)), 2)


class FeatureTests(unittest.TestCase):
  def test_default_feature_schema_matches_source_widths(self) -> None:
    names = ImageTargetFeatureExtractor()._feature_names()
    self.assertEqual(len(names), 16 + 384 + 768 + 26)
    self.assertEqual(names[16], 'dino_000')
    self.assertEqual(names[16 + 384], 'openclip_vit_l_14_000')
    self.assertEqual(names[-1], 'det_top1_is_target')

  def test_training_rejects_same_width_with_wrong_openclip_prefix(self) -> None:
    config = FeatureConfig(dino_dim=2, openclip_dim=3)
    row: dict[str, object] = {
      'image_id': 'scene',
      'image_path': '/unused/scene.jpg',
      'target': 'chair',
      'score': 0.1,
    }
    row.update({name: 0.0 for name in expected_feature_columns(config)})
    frame = pd.DataFrame([row]).rename(columns={
      f'openclip_vit_l_14_{index:03d}': f'openclip_other_model_{index:03d}' for index in range(3)
    })
    with self.assertRaisesRegex(ValueError, 'Raw feature schema does not match FeatureConfig'):
      train_engineered(frame, config, TrainingConfig(cv_folds=0, n_estimators=1, pca_components=1))

  def _fake_features(self, image_path: str) -> _ImageFeatures:
    return _ImageFeatures(
      dino=np.asarray([1.0, 2.0], dtype=np.float32),
      openclip=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
      scores=np.asarray([0.9, 0.6], dtype=np.float32),
      labels=np.asarray([1, 3], dtype=np.int64),
      boxes=np.asarray([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32),
      width=20.0,
      height=20.0,
    )

  def test_features_are_ordered_and_target_conditioned(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'scene.jpg'
      Image.new('RGB', (20, 20)).save(path)
      config = FeatureConfig(dino_dim=2, openclip_dim=3)
      extractor = ImageTargetFeatureExtractor(config)
      pairs = pd.DataFrame({
        'image_path': [str(path), str(path)],
        'target': ['bottle', 'chair'],
        'subject': ['subj01', 'subj02'],
      })
      category_ids = {target: index + 1 for index, target in enumerate(SUPPORTED_TARGETS)}
      with patch.object(extractor, '_compute_image_features', wraps=self._fake_features) as compute:
        with patch.object(extractor, '_detector_category_ids', return_value=category_ids):
          frame = extractor.transform(pairs)
      self.assertEqual(compute.call_count, 1)
      self.assertEqual(frame['subject'].tolist(), ['subj01', 'subj02'])
      self.assertEqual(frame['dino_000'].tolist(), [1.0, 1.0])
      self.assertEqual(frame['task_bottle'].tolist(), [1.0, 0.0])
      self.assertEqual(frame['task_chair'].tolist(), [0.0, 1.0])
      self.assertEqual(frame['det_target_count_lo'].tolist(), [1.0, 1.0])
      self.assertAlmostEqual(frame.loc[0, 'det_target_max_score'], 0.9)
      self.assertAlmostEqual(frame.loc[1, 'det_target_max_score'], 0.6)
      expected = [f'task_{target.replace(" ", "_")}' for target in SUPPORTED_TARGETS]
      expected += ['dino_000', 'dino_001']
      expected += [f'openclip_vit_l_14_{index:03d}' for index in range(3)]
      expected += [f'det_{name}' for name in DETECTOR_FEATURE_NAMES]
      self.assertEqual(feature_columns(frame), expected)
      self.assertEqual(frame.columns.tolist()[-len(expected):], expected)

  def test_disk_cache_reuses_image_only_features(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      path = root / 'scene.jpg'
      Image.new('RGB', (20, 20)).save(path)
      pairs = pd.DataFrame({'image_path': [str(path)], 'target': ['bottle']})
      config = FeatureConfig(dino_dim=2, openclip_dim=3)
      first = ImageTargetFeatureExtractor(config, cache_dir=root / 'cache')
      with patch.object(first, '_compute_image_features', wraps=self._fake_features):
        original = first.transform(pairs)
      second = ImageTargetFeatureExtractor(config, cache_dir=root / 'cache')
      with patch.object(second, '_compute_image_features', side_effect=AssertionError('cache miss')):
        with patch.object(second, '_load_detector', side_effect=AssertionError('unneeded detector load')):
          cached = second.transform(pairs)
      pd.testing.assert_frame_equal(original, cached)


class AdapterTests(unittest.TestCase):
  def test_coco_adapter_excludes_only_the_two_extra_categories(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'tv').mkdir()
      Image.new('RGB', (4, 4)).save(root / 'tv' / '00001.jpg')
      csv_path = root / 'rankings.csv'
      pd.DataFrame({
        'image': ['00001.jpg', 'unused.jpg'],
        'task': ['television', 'bowl'],
        'score': [0.5, 0.1],
      }).to_csv(csv_path, index=False)
      frame = load_coco_pairs(csv_path, root)
      self.assertEqual(frame[['image_id', 'target', 'score']].iloc[0].tolist(), ['00001', 'tv', 0.5])

  def test_coco_adapter_groups_nsd_augmented_filenames_by_physical_image(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for target in ('chair', 'cup'):
        folder = root / target
        folder.mkdir()
        Image.new('RGB', (4, 4)).save(folder / f'train-2152_nsd-18280_{target}.png')
      csv_path = root / 'rankings.csv'
      pd.DataFrame({
        'image': ['train-2152_nsd-18280_chair.png', 'train-2152_nsd-18280_cup.png'],
        'task': ['chair', 'cup'],
        'score': [0.5, 0.7],
      }).to_csv(csv_path, index=False)
      frame = load_coco_pairs(csv_path, root)
      self.assertEqual(frame['image_id'].tolist(), ['nsd-18280', 'nsd-18280'])

  def test_coco_adapter_removes_target_suffix_from_image_id(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for target in ('chair', 'cup'):
        folder = root / target
        folder.mkdir()
        Image.new('RGB', (4, 4)).save(folder / f'scene42_{target}.jpg')
      csv_path = root / 'rankings.csv'
      pd.DataFrame({
        'image': ['scene42_chair.jpg', 'scene42_cup.jpg'],
        'task': ['chair', 'cup'],
        'score': [0.5, 0.7],
      }).to_csv(csv_path, index=False)
      self.assertEqual(load_coco_pairs(csv_path, root)['image_id'].tolist(), ['scene42', 'scene42'])

  def test_nsd_adapter_uses_physical_id_across_subjects(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for subject in ('subj01', 'subj02'):
        path = root / subject / 'training_split' / 'training_images'
        path.mkdir(parents=True)
        Image.new('RGB', (4, 4)).save(path / f'{subject}_nsd-00013_chair.png')
      inventory = root / 'inventory.csv'
      pd.DataFrame({
        'nsd_id': [13, 13, 14],
        'subject': ['subj01', 'subj02', 'subj01'],
        'complexity_category': ['chair', 'chair', 'chair'],
        'complexity_score': [0.2, 0.3, None],
        'has_complexity': [True, True, False],
      }).to_csv(inventory, index=False)
      frame = load_nsd_pairs(inventory, root)
      self.assertEqual(frame['image_id'].tolist(), ['nsd-00013', 'nsd-00013'])
      self.assertEqual(frame['subject'].tolist(), ['subj01', 'subj02'])
      self.assertNotEqual(frame.loc[0, 'image_path'], frame.loc[1, 'image_path'])


if __name__ == '__main__':
  unittest.main()
