"""Regression checks for scientific data selection, identities, and exports."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from complexity.evaluation.compute_variance import decompose
from complexity.evaluation.compare_paper_tables import compare
from complexity.exports.export_rankings import export_ranking
from complexity.generation.generate_scanpaths import partition_images
from complexity.preprocessing.select_coco_trials import select_trials


class PipelineTests(unittest.TestCase):
  def test_paper_comparison_distinguishes_missing_and_numerical_differences(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / 'suite').mkdir()
      (root / 'suite/loo.csv').write_text('model,elpd_loo\na,-10.49\nb,-10.51\nc,nan\n')
      reference = root / 'reference.csv'
      reference.write_text(
        'table,suite,file,model,metric,published,decimals\n'
        + ''.join(f'1,suite,loo.csv,{name},elpd_loo,-10,0\n' for name in ['a', 'b', 'c', 'd'])
      )
      self.assertEqual(
        [row['status'] for row in compare(root, reference)],
        ['matches_printed_value', 'differs', 'nonfinite', 'missing'],
      )

  def test_generation_preserves_contiguous_four_worker_schedule(self) -> None:
    images = [Path(str(index)) for index in range(7)]
    self.assertEqual(partition_images(images, 4), [images[:2], images[2:4], images[4:6], images[6:]])
    self.assertEqual(partition_images(images[:2], 4), [[images[0]], [images[1]], [], []])

  def test_detectable_image_retains_failed_observers(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / 'input.csv'
      pd.DataFrame({
        'image': ['a', 'a', 'b', 'c'], 'task': ['bottle', 'bowl', 'bottle', 'bottle'],
        'condition': ['present', 'present', 'present', 'absent'],
        'first_target_fix_idx': [0, np.nan, np.nan, 1], 'correct': [0, 0, 1, 1],
      }).to_csv(source, index=False)
      report = select_trials(source, root / 'selected.csv')
      self.assertEqual(report['retained_trials'], 2)
      self.assertEqual(pd.read_csv(root / 'selected.csv')['task'].tolist(), ['bottle', 'bowl'])

  def test_repeated_scene_target_scores_are_averaged(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'ranking.csv'
      pd.DataFrame({
        'image': ['train-1_nsd-1_a.png', 'train-2_nsd-1_a.png', 'train-1_nsd-1_b.png', 'train-3_nsd-2_a.png'],
        'task': ['a', 'a', 'b', 'a'], 'score': [0, 2, 3, 100],
      }).to_csv(path, index=False)
      result = decompose(path)
      self.assertEqual(result['pairs'], 2)
      self.assertEqual(result['images'], 1)
      self.assertAlmostEqual(result['within'], 2)
      self.assertAlmostEqual(result['total'], result['within'] + result['between'])

  def test_export_uses_posterior_means_and_historical_identity(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      posterior = root / 'model.nc'
      xr.Dataset(
        {'img_re': (('chain', 'draw', 'image'), np.array([[[1., 4.], [3., 6.]]]))},
        coords={'image': ['a', 'b']},
      ).to_netcdf(posterior, group='posterior', engine='h5netcdf')
      trials = root / 'trials.csv'
      pd.DataFrame({'image': ['a', 'a', 'b'], 'task': ['bottle', 'bowl', 'chair']}).to_csv(trials, index=False)
      result = pd.read_csv(export_ranking(posterior, trials, root / 'export', plot=False))
      self.assertEqual(result.columns.tolist(), ['image', 'score', 'task'])
      self.assertEqual(result['image'].tolist(), ['b', 'a'])
      self.assertEqual(result['score'].tolist(), [5., 2.])
      self.assertEqual(result['task'].tolist(), ['chair', 'bowl'])


if __name__ == '__main__':
  unittest.main()
