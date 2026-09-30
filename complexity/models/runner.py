"""Run the preserved measurement models with explicit, recorded settings."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from types import ModuleType
from typing import Any

import arviz as az
import numpy as np
import pandas as pd


def registry(module: ModuleType, family: str) -> dict[str, tuple[Any, ...]]:
  """Return the original builder and CV arguments.

  ``Any`` is confined to the legacy PyMC model interface.
  """
  if family == 'count':
    return {
      'M1_n_lognormal': (module.build_M1_n_lognormal, {}, 'normal_logn', 'img_re'),
      'M2_n_studentT': (module.build_M2_n_studentt, {}, 'studentt_logn', 'img_re'),
      'M3_n_shifted_lognormal': (module.build_M3_n_shifted_lognormal, {}, 'shifted_lognormal_n', 'img_re'),
      'M4_n_exGaussian': (module.build_M4_n_exgaussian, {}, 'exgaussian_n', 'img_re'),
    }
  specs = {
    'M1_logrt_normal': (module.build_M1_rt_lognormal, {}, 'rt_only', 'img_re', 'normal_logrt', None),
    'M2_logrt_studentT': (module.build_M2_rt_studentt, {}, 'rt_only', 'img_re', 'studentt_logrt', None),
    'M3_shifted_lognormal': (module.build_M3_rt_shifted_lognormal, {}, 'rt_only', 'img_re', 'shifted_lognormal_rt', None),
    'M4_rt_exGaussian': (module.build_M4_rt_exgaussian, {}, 'rt_only', 'img_re', 'exgaussian_rt', None),
  }
  for name, rt_family, mode in [
    ('M5a_joint_normal_logN1p', 'normal_logrt', 'logN1p'),
    ('M5b_joint_normal_N', 'normal_logrt', 'N'),
    ('M2J_joint_studentT_logN1p', 'studentt_logrt', 'logN1p'),
    ('M3J_joint_shiftedLN_logN1p', 'shifted_lognormal_rt', 'logN1p'),
    ('M4J_joint_exGaussian_logN1p', 'exgaussian_rt', 'logN1p'),
  ]:
    specs[name] = (
      module.build_joint_model, {'rt_family': rt_family, 'xN_mode': mode}, 'joint', 'C_image', rt_family, mode
    )
  return specs


def prepare_loo_likelihood(
  module: ModuleType, family: str, spec: tuple[Any, ...], idata: az.InferenceData,
  data: pd.DataFrame, meta: dict[str, Any], seed: int, mc_n: int,
) -> None:
  """Restore the original joint-model marginalization before RT LOO scoring.

  Legacy model specifications and metadata retain their original library types.
  Failures propagate instead of silently comparing conditional and marginal RT.
  """
  if family != 'rt' or spec[2] != 'joint':
    return
  if 'log_likelihood' not in idata.groups():
    raise ValueError('Joint LOO requires a log_likelihood group.')
  if 'logrt_marginal' not in idata.log_likelihood:
    idata.log_likelihood['logrt_marginal'] = module.compute_joint_marginal_loglik(
      idata, data, meta, rt_family=spec[4], xN_mode=spec[5], mcN=max(20, mc_n), seed=seed,
    )


def run_suite(module: ModuleType, family: str) -> None:
  """Fit selected models, save diagnostics, and optionally run LOO and CV."""
  specs = registry(module, family)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--csv', type=Path, required=True)
  parser.add_argument('--results-dir', '--results_dir', dest='results_dir', type=Path, required=True)
  parser.add_argument('--models', nargs='+', choices=list(specs), default=list(specs))
  parser.add_argument('--resume', action='store_true', help='Reuse only checkpoints with matching recorded settings.')
  parser.add_argument('--draws', type=int, default=2000)
  parser.add_argument('--tune', type=int, default=2000)
  parser.add_argument('--chains', type=int, default=4)
  parser.add_argument('--target-accept', '--target_accept', dest='target_accept', type=float, default=0.95)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--skip-cv', '--skip_cv', dest='skip_cv', action='store_true')
  parser.add_argument('--skip-loo', action='store_true')
  parser.add_argument('--cv-splits', type=int, default=5)
  parser.add_argument('--cv-draws', type=int, default=600)
  parser.add_argument('--cv-tune', type=int, default=600)
  parser.add_argument('--cv-chains', type=int, default=2)
  parser.add_argument('--cv-mcn', type=int, default=2)
  parser.add_argument('--topk', type=int, default=20)
  args = parser.parse_args()
  if min(args.draws, args.tune, args.chains, args.cv_draws, args.cv_tune, args.cv_chains) < 1:
    parser.error('Sampling counts must be positive.')
  if not 0 < args.target_accept < 1 or args.cv_splits < 2:
    parser.error('Require 0 < target acceptance < 1 and at least two CV folds.')

  settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
  settings.pop('resume')
  settings['input_sha256'] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
  settings['source_sha256'] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
  settings['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  settings['python'] = platform.python_version()
  settings['packages'] = {
    package: importlib.metadata.version(package)
    for package in ['pymc', 'arviz', 'pytensor', 'numpy', 'pandas', 'scipy', 'xarray', 'h5netcdf']
  }
  args.results_dir.mkdir(parents=True, exist_ok=True)
  settings_path = args.results_dir / 'settings.json'
  if any(args.results_dir.iterdir()):
    if not args.resume or not settings_path.exists() or json.loads(settings_path.read_text()) != settings:
      raise ValueError('Use a fresh results directory, or --resume with exactly matching settings.')
  settings_path.write_text(json.dumps(settings, indent=2) + '\n')
  data, meta = module.prepare_df(pd.read_csv(args.csv))
  if data.empty:
    raise ValueError('No valid observations remain.')
  selected = {name: specs[name] for name in args.models}
  if family == 'rt' and not meta['has_N'] and any(spec[2] == 'joint' for spec in selected.values()):
    raise ValueError('Joint models require N. Select only RT models for inputs without counts.')
  manifest_path = args.results_dir / 'manifest.json'
  manifest: dict[str, Any] = {'data': {'rows': len(data), 'images': len(meta['images'])}, 'models': {}}
  fitted: dict[str, tuple[Any, Any]] = {}
  for name, spec in selected.items():
    model = spec[0](data, meta, **spec[1])
    checkpoint = args.results_dir / f'{name}.nc'
    manifest['models'][name] = 'running'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    try:
      if args.resume and checkpoint.exists():
        idata = module.load_idata(checkpoint)
      else:
        idata = module.fit(
          model, draws=args.draws, tune=args.tune, chains=args.chains,
          target_accept=args.target_accept, seed=args.seed
        )
        module.save_idata(idata, checkpoint)
      effect = 'img_re' if 'img_re' in idata.posterior else 'C_image'
      summary = az.summary(idata, var_names=[effect], kind='diagnostics')
      diagnostics = {
        'divergences': int(idata.sample_stats['diverging'].sum()),
        'max_rhat': float(az.rhat(idata, var_names=[effect])[effect].max()) if args.chains > 1 else None,
        'min_ess_bulk': float(summary['ess_bulk'].min()),
        'min_ess_tail': float(summary['ess_tail'].min()),
        'images': int(idata.posterior.sizes['image']),
      }
      (args.results_dir / f'{name}_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
      fitted[name] = (model, idata)
      manifest['models'][name] = 'complete'
    except Exception as error:
      manifest['models'][name] = {'status': 'failed', 'error': str(error)}
      raise
    finally:
      manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

  if not args.skip_loo and len(fitted) >= 2:
    for name, (_, idata) in fitted.items():
      prepare_loo_likelihood(module, family, selected[name], idata, data, meta, args.seed, args.cv_mcn)
    comparison = module.compare_loo(fitted, data)
    comparison.to_csv(args.results_dir / 'loo_table.csv')
  if not args.skip_cv:
    kwargs = {'mcN': args.cv_mcn} if family == 'rt' else {}
    cv = module.cellheldout_cv(
      data, meta, selected, n_splits=args.cv_splits, seed=args.seed,
      draws=args.cv_draws, tune=args.cv_tune, chains=args.cv_chains,
      target_accept=args.target_accept, topk=args.topk, **kwargs
    )
    (args.results_dir / 'cv_results.json').write_text(json.dumps(cv, indent=2) + '\n')
    module.build_model_selection_table(cv).to_csv(args.results_dir / 'model_selection.csv', index=False)
  manifest['status'] = 'complete'
  manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
