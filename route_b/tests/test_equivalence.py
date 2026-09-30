"""Numerical equivalence to the original implementation and interruption tests.

Set ``ROUTE_B_ORIGINAL`` to the preserved preds checkout for comparison tests.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from route_b import config
from route_b.config import Paths
from route_b.models.complexity import BrainComplexityModel
from route_b.models.encoder import BrainFeatureDecoder, FeatureDecodingLoss
from route_b.runtime import Budget, BudgetExhausted, EpochRecovery, atomic_checkpoint
from route_b.training.complexity import _train_one_fold, set_seed


ORIGINAL = os.environ.get('ROUTE_B_ORIGINAL')
if ORIGINAL:
  sys.path.insert(0, ORIGINAL)


def head() -> BrainComplexityModel:
  """Use production layers at small dimensions to test exact training recovery."""
  return BrainComplexityModel(fmri_dim=6, dino_dim=3, clip_dim=3, clip_text_dim=3, n_categories=2,
                              n_subjects=2, encoder_hidden=8, encoder_latent=8, head_hidden=8)


def samples() -> list[dict[str, torch.Tensor]]:
  generator = torch.Generator().manual_seed(19)
  return [{'fmri': torch.randn(6, generator=generator), 'subj_id': torch.tensor(index % 2),
           'category_idx': torch.tensor(index % 2), 'clip_text': torch.randn(3, generator=generator),
           'score': torch.tensor(index / 100.),
           **{key: torch.randn(3, generator=generator) for key in ('dino_early', 'dino_mid', 'dino_late', 'clip')}}
          for index in range(32)]


class PortTests(unittest.TestCase):
  def setUp(self) -> None:
    torch.set_num_threads(1)

  def test_budget_cannot_reset_after_a_crash(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / 'budget.json'
      budget = Budget(path, 120 / 3600, devices=2)
      budget.check()
      self.assertEqual(budget.charged, 120.)
      restarted = Budget(path, 120 / 3600, devices=2)
      with self.assertRaises(BudgetExhausted):
        restarted.check()
      with self.assertRaises(ValueError):
        Budget(path, 1.)

  def test_head_resume_preserves_final_weights_and_global_rng(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)

      def train(directory: Path) -> dict:
        set_seed(42)
        model = head()
        before = {key: value.clone() for key, value in model.encoder.state_dict().items()}
        rows = samples()
        result = _train_one_fold(DataLoader(rows[:24], batch_size=8, shuffle=True),
          DataLoader(rows[24:], batch_size=8), model, torch.device('cpu'), 4, .0003, 15, 100, 0.,
          recovery=EpochRecovery(directory, {'fixture': True}))
        for key, value in before.items():
          torch.testing.assert_close(value, model.encoder.state_dict()[key], atol=0, rtol=0)
        return result

      full = train(root / 'full')

      def interrupt(path: Path, state: dict) -> None:
        atomic_checkpoint(path, state)
        if state['epoch'] == 1:
          raise InterruptedError('Test process interruption')

      with patch('route_b.runtime.atomic_checkpoint', side_effect=interrupt):
        with self.assertRaises(InterruptedError):
          train(root / 'resumed')
      resumed = train(root / 'resumed')
      self.assertEqual(full['best_epoch'], resumed['best_epoch'])
      for filename in ('last.pt',):
        left = torch.load(root / 'full' / filename, weights_only=False)
        right = torch.load(root / 'resumed' / filename, weights_only=False)
        for key, value in left['model'].items():
          torch.testing.assert_close(value, right['model'][key], atol=0, rtol=0)
        torch.testing.assert_close(left['rng']['torch'], right['rng']['torch'], atol=0, rtol=0)
        self.assertEqual(left['scheduler'], right['scheduler'])

  @unittest.skipUnless(ORIGINAL, 'Set ROUTE_B_ORIGINAL for original-code comparison')
  def test_head_training_matches_original(self) -> None:
    original = importlib.import_module('src.training.train_complexity')
    results = []
    for fitting in (original._train_one_fold, _train_one_fold):
      set_seed(42)
      model = head()
      rows = samples()
      result = fitting(DataLoader(rows[:24], batch_size=8, shuffle=True),
        DataLoader(rows[24:], batch_size=8), model, torch.device('cpu'), 4, .0003, 15, 100, 0.)
      results.append((result, model.state_dict(), torch.get_rng_state()))
    self.assertEqual(results[0][0]['best_val_loss'], results[1][0]['best_val_loss'])
    for key, value in results[0][1].items():
      torch.testing.assert_close(value, results[1][1][key], atol=0, rtol=0)
    torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)

  @unittest.skipUnless(ORIGINAL, 'Set ROUTE_B_ORIGINAL for original-code comparison')
  def test_encoder_layers_losses_and_gradients_match_original(self) -> None:
    original = importlib.import_module('src.models.fmri_encoder')
    rows = next(iter(DataLoader(samples(), batch_size=8)))
    for weight in (0., .25, .5, .75, 1.):
      outputs = []
      for model_class, criterion_class in ((original.BrainFeatureDecoder, original.FeatureDecodingLoss),
                                            (BrainFeatureDecoder, FeatureDecodingLoss)):
        set_seed(42)
        model = model_class(fmri_dim=6, dino_dim=3, clip_dim=3, n_subjects=2, hidden_dim=8, latent_dim=8)
        loss, _ = criterion_class(visual_weight=weight, fmri_recon_weight=1-weight)(
          model(rows['fmri'], rows['subj_id']), rows, fmri=rows['fmri'])
        loss.backward()
        outputs.append((loss.detach(), {key: parameter.grad for key, parameter in model.named_parameters()}))
      torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
      for key, value in outputs[0][1].items():
        if value is None:
          self.assertIsNone(outputs[1][1][key])
        else:
          torch.testing.assert_close(value, outputs[1][1][key], rtol=0, atol=0)

  @unittest.skipUnless(ORIGINAL, 'Set ROUTE_B_ORIGINAL for original-code comparison')
  def test_pca_math_and_subject_splits_match_original(self) -> None:
    original = importlib.import_module('src.data.roi_utils')
    ported = importlib.import_module('route_b.data.roi')
    fmri = np.random.RandomState(42).normal(size=(40, 18)).astype(np.float32)
    streams = np.repeat([1, 2, 5], 6)
    train_rows = np.random.RandomState(42).permutation(40)[4:]
    outputs = []
    for module in (original, ported):
      with patch.object(module, 'load_streams_mask', return_value=streams):
        fitted = module.fit_subject_pca('subj01', fmri, train_rows, 4)
        outputs.append(module.apply_subject_pca('subj01', fmri, fitted))
    np.testing.assert_array_equal(*outputs)

  @unittest.skipUnless(ORIGINAL, 'Set ROUTE_B_ORIGINAL for original-code comparison')
  def test_encoder_training_and_resume_match_original(self) -> None:
    from torch.utils.data import ConcatDataset, Dataset
    from route_b.training import encoder as ported
    original = importlib.import_module('src.training.train_feature_decoder')
    rows = samples()

    class Fixture(Dataset):
      fmri_dim = 6

      def __init__(self, values: list[dict[str, torch.Tensor]]) -> None:
        self.values = values

      def __len__(self) -> int:
        return len(self.values)

      def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.values[index]

    def data(split: str, **_kwargs: object) -> tuple[ConcatDataset, dict]:
      return ConcatDataset([Fixture(rows[:24] if split == 'train' else rows[24:])]), {}

    def model(**_kwargs: object) -> BrainFeatureDecoder:
      return BrainFeatureDecoder(fmri_dim=6, dino_dim=3, clip_dim=3, n_subjects=2, hidden_dim=8, latent_dim=8)

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)

      def fit(module: object, directory: Path, recovery: bool) -> None:
        from contextlib import ExitStack
        settings = Paths(root, root, root, root, root, directory, root, root)
        with ExitStack() as stack:
          stack.enter_context(patch.object(config, '_paths', settings))
          if module is original:
            stack.enter_context(patch.object(module, 'CHECKPOINTS_DIR', directory))
          for name, value in [('NUM_WORKERS', 0), ('BrainFeatureDecoder', model),
                              ('build_multi_subject_feature_dataset', data), ('get_complexity_val_nsd_ids', lambda: set())]:
            stack.enter_context(patch.object(module, name, value))
          kwargs = {'recovery_root': directory / 'recovery'} if recovery else {}
          module.train_dnn_decoder(epochs=3, batch_size=8, gpus=None, **kwargs)

      fit(original, root / 'original', False)
      fit(ported, root / 'full', True)

      def interrupt(path: Path, state: dict) -> None:
        atomic_checkpoint(path, state)
        if state['epoch'] == 1:
          raise InterruptedError('Test encoder interruption')

      with patch('route_b.runtime.atomic_checkpoint', side_effect=interrupt):
        with self.assertRaises(InterruptedError):
          fit(ported, root / 'resumed', True)
      fit(ported, root / 'resumed', True)
      checkpoints = [torch.load(root / name / 'feature_decoder/dnn_multisubj_best.pt', weights_only=False)
                     for name in ('original', 'full', 'resumed')]
      for candidate in checkpoints[1:]:
        self.assertEqual(checkpoints[0]['val_loss'], candidate['val_loss'])
        for key, value in checkpoints[0]['model_state_dict'].items():
          torch.testing.assert_close(value, candidate['model_state_dict'][key], rtol=0, atol=0)
      left = torch.load(root / 'full/recovery/dnn_multisubj_best/last.pt', weights_only=False)
      right = torch.load(root / 'resumed/recovery/dnn_multisubj_best/last.pt', weights_only=False)
      for key, value in left['model'].items():
        torch.testing.assert_close(value, right['model'][key], rtol=0, atol=0)
      torch.testing.assert_close(left['rng']['torch'], right['rng']['torch'], rtol=0, atol=0)

  @unittest.skipUnless(ORIGINAL, 'Set ROUTE_B_ORIGINAL for original-code comparison')
  def test_datasets_keep_original_order_means_and_normalization(self) -> None:
    from contextlib import ExitStack
    original = importlib.import_module('src.data.datasets')
    ported = importlib.import_module('route_b.data.datasets')
    rng = np.random.RandomState(1)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      np.save(root / 'subj01_pca_fmri.npy', rng.normal(size=(20, 6)).astype(np.float32))
      np.save(root / 'subj01_train_idx.npy', np.arange(15))
      np.save(root / 'subj01_val_idx.npy', np.arange(15, 20))
      np.save(root / 'subj01_dino_layers.npy', rng.normal(size=(20, 24, 3)).astype(np.float32))
      np.save(root / 'subj01_clip_img.npy', rng.normal(size=(20, 3)).astype(np.float32))
      text = {'bottle': rng.normal(size=3).astype(np.float32), 'car': rng.normal(size=3).astype(np.float32)}
      records = [{'train_idx': index + 1, 'nsd_id': index, 'task': 'bottle' if index % 2 else 'car',
                  'score': index / 10., 'image': f'train-{index + 1}_nsd-{index}.png', 'split': 'train'}
                 for index in (1, 2, 2, 4, 7, 9, 15)]
      outputs = []
      for module in (original, ported):
        with ExitStack() as stack:
          stack.enter_context(patch.object(config, '_paths', Paths(root, root, root, root, root, root, root, root)))
          if module is original:
            for key in ('PCA_DIR', 'DINO_FEATURES_DIR', 'CLIP_IMG_FEATURES_DIR'):
              stack.enter_context(patch.object(module, key, root))
          stack.enter_context(patch.object(module, 'get_clip_text_embeddings', return_value=text))
          stack.enter_context(patch.object(module, 'list_training_images', return_value=[
            root / f'train-{index + 1}_nsd-{index}.png' for index in range(20)]))
          head_data = module.ComplexityDataset('subj01', records=records)
          encoder_data = module.FeatureDecodingDataset('subj01', exclude_nsd_ids={2, 4})
          outputs.append((head_data, encoder_data))
      for left, right in zip(outputs[0], outputs[1], strict=True):
        self.assertEqual(len(left), len(right))
        np.testing.assert_array_equal(left._fmri_mean, right._fmri_mean)
        np.testing.assert_array_equal(left._fmri_std, right._fmri_std)
        for index in range(len(left)):
          for key, value in left[index].items():
            if isinstance(value, torch.Tensor):
              torch.testing.assert_close(value, right[index][key], rtol=0, atol=0)
            else:
              self.assertEqual(value, right[index][key])
      self.assertEqual(outputs[0][0].samples, outputs[1][0].samples)



  def test_runner_checkpoint_names_match_both_trainers(self) -> None:
    from route_b.run import checkpoint_stem
    from route_b.training.encoder import _checkpoint_stem as encoder_stem
    from route_b.training.complexity import _checkpoint_stem as head_stem
    for weight in (0., .25, .5, .75, 1.):
      name = f'visual_{weight:g}'
      self.assertEqual(checkpoint_stem(weight), encoder_stem(weight, 1 - weight, name))
      self.assertEqual(checkpoint_stem(weight), head_stem(None, name))

  def test_replay_rejects_changed_observations_and_predictions(self) -> None:
    from route_b.evaluation.reproduction import compare
    truth = np.array([.1, .3, .5, .8, 1.2])
    original = {'truth': truth, 'prediction': truth + .01, 'residual_truth': truth - .2,
                'residual_prediction': truth - .19, 'log_variance': np.full(5, -2.),
                'record_index': np.arange(5), 'nsd_id': np.arange(5)}
    self.assertTrue(all(value == 0 for value in compare(original, original).values()))
    for field in ('nsd_id', 'prediction'):
      changed = {key: value.copy() for key, value in original.items()}
      changed[field][0] += 1
      with self.assertRaises(AssertionError):
        compare(original, changed)


if __name__ == '__main__':
  unittest.main()
