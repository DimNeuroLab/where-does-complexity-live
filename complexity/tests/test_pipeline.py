"""Regression checks for scientific data selection, identities, and exports."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from complexity.evaluation.compare_paper_tables import compare
from complexity.evaluation.compute_variance import decompose
from complexity.exports.export_rankings import export_ranking
from complexity.generation.generate_scanpaths import partition_images
from complexity.models import fit_response_time
from complexity.models.runner import prepare_loo_likelihood, registry
from complexity.preprocessing.select_coco_trials import select_trials


class PipelineTests(unittest.TestCase):
  def test_joint_cv_matches_historical_mixture_in_all_likelihood_families(self) -> None:
    posterior = {
      name: np.array([[value, value]]) for name, value in {
        'b0_N': 0., 'b_trial_N': 0., 'b_C_N': 0., 'alpha_N': 2.,
        'b_trial_RT': 0., 'b_xN': 0., 'b_C_RT': 0.,
      }.items()
    }
    posterior.update({
      'b0_RT': np.array([[0., 10.]]), 'sigma_RT': np.array([[1., 10.]]),
      'sigma': np.array([[1., 10.]]), 'nu': np.array([[3., 7.]]), 'tau': np.array([[0.1, 0.2]]),
      **{name: np.zeros((1, 2, 1)) for name in ['C_image', 'subj_re_N', 'subj_re_RT']},
    })
    idata = az.from_dict(posterior=posterior)
    data, meta = fit_response_time.prepare_df(pd.DataFrame({
      'RT': [np.e] * 3, 'N': [1] * 3, 'subject': [1] * 3, 'image': ['a'] * 3, 'trial': [1, 2, 3],
    }))
    fold = fit_response_time.Fold(0, np.array([0, 1]), np.array([2]))
    names = ['M5a_joint_normal_logN1p', 'M2J_joint_studentT_logN1p',
             'M3J_joint_shiftedLN_logN1p', 'M4J_joint_exGaussian_logN1p', 'M5b_joint_normal_N']
    scorers = [
      lambda mu, sigma, shape: fit_response_time.score_logrt_mixture_normal(np.array([1.]), mu, sigma),
      lambda mu, sigma, shape: fit_response_time.score_logrt_mixture_studentt(np.array([1.]), mu, sigma, shape),
      lambda mu, sigma, shape: fit_response_time.score_logrt_mixture_shifted_lognormal(
        np.array([1.]), np.array([np.e]), mu, sigma, shape,
      ),
      lambda mu, sigma, shape: fit_response_time.score_logrt_mixture_exgaussian(
        np.array([1.]), np.array([np.e]), mu, sigma, shape,
      ),
      lambda mu, sigma, shape: fit_response_time.score_logrt_mixture_normal(np.array([1.]), mu, sigma),
    ]
    for index, (name, scorer) in enumerate(zip(names, scorers)):
      with self.subTest(model=name):
        shape = [0.1, 0.2] if index == 2 else [3., 7.]
        # The recovered computation mixes these four mean/scale combinations.
        expected = scorer(np.array([[0.], [10.], [0.], [10.]]), np.array([1., 1., 10., 10.]),
                          np.array([shape[0], shape[0], shape[1], shape[1]]))[1]
        with patch('complexity.models.sampling.sample_fold', return_value=idata), \
             patch.object(fit_response_time, 'make_image_stratified_cell_folds', return_value=[fold]), \
             patch.object(fit_response_time, 'ranking_stability_from_fold_samples', return_value={}):
          actual = fit_response_time.cellheldout_cv(
            data, meta, {name: registry(fit_response_time, 'rt')[name]}, mcN=2,
          )
        self.assertAlmostEqual(actual[name]['elpd_logrt_total_mean'], expected, places=12)

  def test_verification_reports_changed_labels_despite_matching_shared_scores(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for name, images in [('actual', ['a', 'b', 'new']), ('reference', ['a', 'b', 'old'])]:
        pd.DataFrame({'image': images, 'task': ['bottle'] * 3, 'score': [1., 2., 3.]}).to_csv(
          root / f'{name}.csv', index=False,
        )
      (root / 'diagnostics.json').write_text(json.dumps({
        'divergences': 0, 'max_rhat': 1., 'min_ess_bulk': 500, 'min_ess_tail': 500,
      }))
      result = subprocess.run([
        sys.executable, '-m', 'complexity.evaluation.verify_ranking', '--ranking', str(root / 'actual.csv'),
        '--reference', str(root / 'reference.csv'), '--output', str(root / 'report.json'), '--mode', 'refit',
        '--diagnostics', str(root / 'diagnostics.json'),
      ], capture_output=True, text=True)
      self.assertEqual(result.returncode, 1, result.stderr)
      report = json.loads((root / 'report.json').read_text())
      self.assertEqual(report['added_labels'], ['new'])
      self.assertEqual(report['missing_labels'], ['old'])
      self.assertEqual(report['shared_rows'], 2)
      self.assertTrue(report['numerical_agreement'])
      self.assertFalse(report['passed'])

  def test_joint_loo_marginalizes_count_before_scoring(self) -> None:
    shape = (2, 50)
    posterior = xr.Dataset(
      {
        **{name: (('chain', 'draw'), np.full(shape, value)) for name, value in {
          'b0_RT': 0., 'b_trial_RT': 0., 'b_xN': 1., 'b_C_RT': 1., 'alpha_N': 2., 'sigma_RT': 0.5,
        }.items()},
        'subj_re_RT': (('chain', 'draw', 'subject'), np.zeros((*shape, 1))),
        'C_image': (('chain', 'draw', 'image'), np.zeros((*shape, 1))),
        'mu_N': (('chain', 'draw', 'obs'), np.full((*shape, 4), 2.)),
      },
      coords={'chain': range(2), 'draw': range(50), 'subject': [0], 'image': ['a'], 'obs': range(4)},
    )
    data = pd.DataFrame({
      'log_trial_c': [0.] * 4, 'subj_idx': [0] * 4, 'img_idx': [0] * 4,
      'log_rt': [0.2, 0.4, 0.6, 0.8], 'RT': np.exp([0.2, 0.4, 0.6, 0.8]),
    })
    idata = az.InferenceData(posterior=posterior, log_likelihood=xr.Dataset({
      'logrt_like': (('chain', 'draw', 'obs'), np.full((*shape, 4), 20.)),
    }))
    spec = registry(fit_response_time, 'rt')['M5a_joint_normal_logN1p']
    meta = {'logN1p_mean': 0.}
    expected = fit_response_time.compute_joint_marginal_loglik(
      idata, data, meta, rt_family='normal_logrt', xN_mode='logN1p', mcN=20, seed=42,
    )
    prepare_loo_likelihood(fit_response_time, 'rt', spec, idata, data, meta, seed=42, mc_n=2)
    np.testing.assert_array_equal(idata.log_likelihood['logrt_marginal'], expected)
    loo, selected = fit_response_time.loo_on_common_logrt('joint', None, idata, data)
    self.assertEqual(selected, 'logrt_marginal')
    self.assertLess(float(loo.elpd_loo), 0.)

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
