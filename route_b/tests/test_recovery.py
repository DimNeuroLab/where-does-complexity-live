"""Standalone interruption, budget and checkpoint naming checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from route_b import config
from route_b.config import Paths
from route_b.models.complexity import BrainComplexityModel
from route_b.models.encoder import BrainFeatureDecoder
from route_b.runtime import Budget, BudgetExhausted, EpochRecovery, atomic_checkpoint
from route_b.training.complexity import _train_one_fold, set_seed


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


class RecoveryTests(unittest.TestCase):
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


  def test_encoder_resume_preserves_weights_and_global_rng(self) -> None:
    from torch.utils.data import ConcatDataset, Dataset
    from route_b.training import encoder as ported
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
          for name, value in [('NUM_WORKERS', 0), ('BrainFeatureDecoder', model),
                              ('build_multi_subject_feature_dataset', data), ('get_complexity_val_nsd_ids', lambda: set())]:
            stack.enter_context(patch.object(module, name, value))
          kwargs = {'recovery_root': directory / 'recovery'} if recovery else {}
          module.train_dnn_decoder(epochs=3, batch_size=8, gpus=None, **kwargs)

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
                     for name in ('full', 'resumed')]
      for candidate in checkpoints[1:]:
        self.assertEqual(checkpoints[0]['val_loss'], candidate['val_loss'])
        for key, value in checkpoints[0]['model_state_dict'].items():
          torch.testing.assert_close(value, candidate['model_state_dict'][key], rtol=0, atol=0)
      left = torch.load(root / 'full/recovery/dnn_multisubj_best/last.pt', weights_only=False)
      right = torch.load(root / 'resumed/recovery/dnn_multisubj_best/last.pt', weights_only=False)
      for key, value in left['model'].items():
        torch.testing.assert_close(value, right['model'][key], rtol=0, atol=0)
      torch.testing.assert_close(left['rng']['torch'], right['rng']['torch'], rtol=0, atol=0)


  def test_runner_checkpoint_names_match_both_trainers(self) -> None:
    from route_b.run import checkpoint_stem
    from route_b.training.encoder import _checkpoint_stem as encoder_stem
    from route_b.training.complexity import _checkpoint_stem as head_stem
    for weight in (0., .25, .5, .75, 1.):
      name = f'visual_{weight:g}'
      self.assertEqual(checkpoint_stem(weight), encoder_stem(weight, 1 - weight, name))
      self.assertEqual(checkpoint_stem(weight), head_stem(None, name))


if __name__ == '__main__':
  unittest.main()
