"""Bounded command dispatch and argument mapping checks."""

import unittest
from unittest.mock import patch

from route_a.embedding import __main__ as cli
from route_a.embedding import evaluation, evaluation_per_subject, training


class CommandTests(unittest.TestCase):
  def test_command_dispatch_keeps_workflow_arguments(self) -> None:
    for name, module in [('train', training), ('evaluate-global', evaluation),
                         ('evaluate-per-subject', evaluation_per_subject)]:
      with self.subTest(command=name), patch.object(module, 'main', return_value=0) as entry:
        self.assertEqual(cli.main([name, '--help']), 0)
        entry.assert_called_once_with(['--help'])

  def test_normalized_paths_and_folds_reach_submitted_training_arguments(self) -> None:
    args = training.build_parser().parse_args([
      '--nsd-root', 'nsd', '--dino-dir', 'dino', '--clip-image-dir', 'clip',
      '--clip-text-file', 'text.npz', '--ranking-file', 'ranking.csv', '--output-dir', 'run', '--cv-folds', '5',
    ])
    self.assertEqual(str(args.clip_img_dir), 'clip')
    self.assertEqual(str(args.clip_text_path), 'text.npz')
    self.assertEqual(str(args.complexity_csv), 'ranking.csv')
    self.assertEqual(str(args.out), 'run')
    self.assertEqual(args.folds, 5)
    self.assertEqual((args.arch, args.features, args.noise, args.seed), ('small', 'all', 0., 42))

  def test_extraction_dispatch_preserves_batch_size_and_model_helpers(self) -> None:
    with patch.object(cli.shared_features, 'extract_and_save_dino') as dino, \
         patch.object(cli.shared_features, 'extract_and_save_clip') as clip:
      self.assertEqual(cli.main(['extract-features', '--nsd-root', 'nsd', '--dino-dir', 'dino',
        '--clip-image-dir', 'clip', '--clip-text-file', 'text.npz', '--device', 'cpu']), 0)
      self.assertEqual(dino.call_args.kwargs, {'device': 'cpu', 'batch_size': 64})
      self.assertEqual(clip.call_args.kwargs, {'device': 'cpu', 'batch_size': 64})


if __name__ == '__main__':
  unittest.main()
