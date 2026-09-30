"""Standalone configuration, dependency isolation, and resume integrity checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from route_b import config
from route_b.config import Paths, configure
from route_b.data.datasets import ComplexityDataset
from route_b.provenance import fingerprints, verify_files


class PublicPipelineTests(unittest.TestCase):
  def test_fresh_configuration_needs_no_legacy_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      value = {'inputs': {'nsd': str(root / 'nsd')}, 'output': str(root / 'run'), 'head': {'n_folds': 5}}
      filename = root / 'config.json'
      filename.write_text(json.dumps(value))
      with patch.object(config, '_paths', None):
        loaded = configure(filename)
        self.assertNotIn('original_repository', loaded)
        self.assertEqual(config.paths().ranking.name, 'nsd_m2_ranking.csv')
        self.assertEqual(config.paths().ranking.parent.name, 'complexity')
        self.assertTrue(config.paths().ranking.is_file())
        self.assertEqual(config.paths().dino, root / 'run/features/dino')
        self.assertEqual(config.paths().pca, root / 'run/pca')
      self.assertFalse((root / 'run').exists())

  def test_external_pca_models_do_not_redirect_generated_observations(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      value = {'inputs': {'nsd': str(root / 'nsd'), 'pca_models': str(root / 'bundle/pca')},
               'output': str(root / 'run'), 'head': {'n_folds': 5}}
      filename = root / 'config.json'
      filename.write_text(json.dumps(value))
      with patch.object(config, '_paths', None):
        configure(filename)
        self.assertEqual(config.paths().pca, root / 'run/pca')
        self.assertEqual(config.paths().pca_models, root / 'bundle/pca')

  def test_evaluation_samples_match_training_without_loading_image_features(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      rng = np.random.RandomState(23)
      np.save(root / 'subj01_pca_fmri.npy', rng.normal(size=(3, 6)).astype(np.float32))
      np.save(root / 'subj01_dino_layers.npy', rng.normal(size=(3, 24, 2)).astype(np.float32))
      np.save(root / 'subj01_clip_img.npy', rng.normal(size=(3, 3)).astype(np.float32))
      records = [{'nsd_id': index, 'score': index / 10, 'task': 'bottle'} for index in (1, 2, 1)]
      paths = Paths(root, root, root, root, root, root, root, root)
      with patch.object(config, '_paths', paths), patch('route_b.data.datasets.get_clip_text_embeddings',
          return_value={'bottle': np.ones(3, dtype=np.float32)}), patch('route_b.data.datasets.list_training_images',
          return_value=[root / f'train-{i+1}_nsd-{i}.png' for i in range(3)]):
        training = ComplexityDataset('subj01', records=records)
        (root / 'subj01_dino_layers.npy').unlink()
        (root / 'subj01_clip_img.npy').unlink()
        inference = ComplexityDataset('subj01', records=records, visual_targets=False)
        self.assertEqual(training.samples, inference.samples)
        for i in range(len(inference)):
          for key, value in inference[i].items():
            if isinstance(value, torch.Tensor):
              torch.testing.assert_close(value, training[i][key], atol=0, rtol=0)
            else:
              self.assertEqual(value, training[i][key])

  def test_resume_rejects_changed_artifact(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'input'
      path.write_text('original')
      saved = fingerprints([path])
      verify_files(saved)
      path.write_text('changed data')
      with self.assertRaises(ValueError):
        verify_files(saved)

  def test_preparation_cannot_overwrite_external_pca(self) -> None:
    from route_b.run import execute
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      value = {'inputs': {'nsd': str(root / 'nsd'), 'pca': str(root / 'preserved')},
               'output': str(root / 'run'), 'head': {'n_folds': 5}}
      filename = root / 'config.json'
      filename.write_text(json.dumps(value))
      with patch.object(config, '_paths', None), self.assertRaisesRegex(ValueError, 'read-only'):
        execute('prepare', 1., filename)
      self.assertFalse((root / 'preserved').exists())


if __name__ == '__main__':
  unittest.main()
