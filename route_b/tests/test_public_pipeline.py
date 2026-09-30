"""Standalone configuration, dependency isolation, and resume integrity checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
      value = {'inputs': {'nsd_root': str(root / 'nsd')}, 'output_dir': str(root / 'run'), 'head': {'cv_folds': 5}}
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
      value = {'inputs': {'nsd_root': str(root / 'nsd'), 'pca_models_dir': str(root / 'bundle/pca')},
               'output_dir': str(root / 'run'), 'head': {'cv_folds': 5}}
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
      value = {'inputs': {'nsd_root': str(root / 'nsd'), 'pca_dir': str(root / 'preserved')},
               'output_dir': str(root / 'run'), 'head': {'cv_folds': 5}}
      filename = root / 'config.json'
      filename.write_text(json.dumps(value))
      with patch.object(config, '_paths', None), self.assertRaisesRegex(ValueError, 'read-only'):
        execute('prepare', 1., filename)
      self.assertFalse((root / 'preserved').exists())

  def test_configuration_and_resume_are_independent_of_launch_directory(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      settings = root / 'settings'
      settings.mkdir()
      filename = settings / 'run.json'
      filename.write_text(json.dumps({'output_dir': 'output', 'inputs': {}, 'dino_root': '${UNUSED_DINO_ROOT}'}))
      command = [sys.executable, '-m', 'route_b', 'run', '--config', str(filename),
                 '--stages', 'text', '--initialize-only']
      environment = dict(os.environ, PYTHONPATH=str(Path(config.__file__).resolve().parents[1]))
      result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
      self.assertEqual(result.returncode, 0, result.stderr)
      result = subprocess.run([*command, '--resume'], cwd=settings, env=environment, capture_output=True, text=True)
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertFalse((root / 'output').exists())
      manifest = json.loads((settings / 'output/manifest.json').read_text())
      self.assertEqual(manifest['config']['output_dir'], str(settings / 'output'))
      # Changing effective settings must still reject resume.
      value = json.loads(filename.read_text())
      value['gpu_hours'] = 1
      filename.write_text(json.dumps(value))
      result = subprocess.run([*command, '--resume'], cwd=settings, env=environment, capture_output=True, text=True)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn('configuration, or environment changed', result.stderr)

  def test_all_configured_paths_expand_beside_the_configuration(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      filename = root / 'settings.json'
      filename.write_text(json.dumps({'output_dir': 'run', 'inputs': {
        'nsd_root': '${TEST_NSD_ROOT}', 'features_dir': 'features', 'pca_dir': 'pca',
        'checkpoints_dir': 'checkpoints', 'pca_models_dir': 'models', 'ranking_file': 'ranking.csv',
      }, 'dino_root': 'dinov2', 'head': {'cv_folds': 5}}))
      with patch.dict(os.environ, {'TEST_NSD_ROOT': 'nsd'}), patch.object(config, '_paths', None):
        loaded = configure(filename)
        for key, value in loaded['inputs'].items():
          self.assertTrue(Path(value).is_relative_to(root), key)
        self.assertEqual(config.paths().nsd, root / 'nsd')
        self.assertEqual(loaded['dino_root'], str(root / 'dinov2'))
      value = json.loads(filename.read_text())
      value['original_repository'] = 'old-checkout'
      filename.write_text(json.dumps(value))
      with self.assertRaisesRegex(ValueError, 'Unknown configuration fields'):
        configure(filename)

  def test_text_only_inputs_do_not_access_nsd_or_ranking(self) -> None:
    from route_b.provenance import input_files
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      filename = root / 'config.json'
      filename.write_text(json.dumps({'output_dir': 'run'}))
      with patch.object(config, '_paths', None):
        configure(filename, ('text',))
        self.assertIsNone(config.paths().nsd)
        self.assertEqual(input_files('text'), [])

  def test_canonical_commands_dispatch_without_running_other_workflows(self) -> None:
    from route_b.__main__ import COMMANDS, main
    for command, module in COMMANDS.items():
      with self.subTest(command=command), patch(module + '.main') as entry:
        self.assertEqual(main([command, '--help']), 0)
        entry.assert_called_once_with(['--help'])

  def test_variance_utility_accepts_configuration_before_resolving_default_checkpoint(self) -> None:
    from route_b.evaluation import variance
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      filename = root / 'config.json'
      filename.write_text(json.dumps({'output_dir': 'run', 'inputs': {'nsd_root': 'nsd'}}))
      checkpoint = root / 'run/checkpoints/complexity/visual_1.pt'
      checkpoint.parent.mkdir(parents=True)
      checkpoint.touch()
      with patch.object(config, '_paths', None), \
           patch.object(sys, 'argv', ['variance', '--config', str(filename), '--device', 'cpu']), \
           patch.object(variance, 'predict_oof_pair_scores', return_value={}) as predict, \
           patch.object(variance, 'decompose_variance', return_value=(1., .25, .75, 2)):
        variance.main()
        self.assertEqual(predict.call_args.args[0], checkpoint)

  def test_run_summary_uses_current_results_only(self) -> None:
    from route_b.evaluation.reproduction import report
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      current = root / 'results/visual_1'
      old = root / 'replay/visual_1/original'
      for path, value in [(current, .6), (old, .9)]:
        path.mkdir(parents=True)
        (path / 'metrics.json').write_text(json.dumps({'pooled': {
          'pearson': value, 'spearman': value, 'pearson_residual': value, 'mae': .2}}))
      report(root)
      summary = json.loads((root / 'summary.json').read_text())
      self.assertEqual(summary, {'1': {'pooled': {
        'pearson': .6, 'spearman': .6, 'pearson_residual': .6, 'mae': .2}}})
      self.assertTrue((root / 'summary.png').is_file())


if __name__ == '__main__':
  unittest.main()
