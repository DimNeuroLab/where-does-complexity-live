"""Save resumable CV posteriors and assess sampling independently of model scores."""

import hashlib
import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import rankdata


def diagnostics(idata: az.InferenceData) -> dict[str, Any]:
  """Check all non-observation parameters, including the latent image effects.

  ``Any`` contains heterogeneous ArviZ diagnostics and JSON-compatible values.
  :returns: Unrounded diagnostics and an explicit sampling acceptance flag.
  """
  names = [name for name, value in idata.posterior.items() if 'obs' not in value.dims]
  posterior = idata.posterior[names]
  rhat = az.rhat(posterior)
  bulk = az.ess(posterior, method='bulk')
  tail = az.ess(posterior, method='tail')
  parameters = {
    name: {
      'max_rhat': float(rhat[name].max()), 'min_ess_bulk': float(bulk[name].min()),
      'min_ess_tail': float(tail[name].min()),
      'finite': bool(np.isfinite(rhat[name]).all() and np.isfinite(bulk[name]).all()
                     and np.isfinite(tail[name]).all() and np.isfinite(posterior[name]).all()),
    } for name in names
  }
  report: dict[str, Any] = {
    'parameters': parameters, 'divergences': int(idata.sample_stats['diverging'].sum()),
    'max_rhat': max(row['max_rhat'] for row in parameters.values()),
    'min_ess_bulk': min(row['min_ess_bulk'] for row in parameters.values()),
    'min_ess_tail': min(row['min_ess_tail'] for row in parameters.values()),
    'bfmi': az.bfmi(idata).tolist(),
  }
  report['passed'] = bool(
    all(row['finite'] for row in parameters.values()) and report['divergences'] == 0
    and report['max_rhat'] <= 1.01 and report['min_ess_bulk'] >= 400 and report['min_ess_tail'] >= 400
    and np.isfinite(report['bfmi']).all() and min(report['bfmi']) >= 0.3
  )
  if 'reached_max_treedepth' in idata.sample_stats:
    report['max_treedepth_hits'] = int(idata.sample_stats['reached_max_treedepth'].sum())
    report['passed'] = report['passed'] and report['max_treedepth_hits'] == 0
  if 'C_image' in posterior:
    chain_means = posterior['C_image'].mean('draw').transpose('chain', 'image').values
    report['chain_ranking_spearman'] = np.corrcoef(rankdata(chain_means, axis=1)).tolist()
    report['chain_loading_means'] = {
      name: posterior[name].mean('draw').values.tolist() for name in ['b_C_N', 'b_C_RT', 'b_xN']
    }
  return report


def sample_fold(
  model: pm.Model, train: pd.DataFrame, test: pd.DataFrame, checkpoint: Path | None, **kwargs: Any,
) -> az.InferenceData:
  """Sample or resume a fold, preserving draw order and excluding unused observation arrays.

  Sampling keyword arguments cross the PyMC library boundary. The caller must
  guard the parent run directory with a source, input, and environment manifest.
  :param checkpoint: Fold directory, or ``None`` to retain ordinary sampling.
  """
  if checkpoint is None:
    with model:
      return pm.sample(**kwargs)
  settings = {
    'sampling': kwargs,
    'train_sha256': hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
    'test_sha256': hashlib.sha256(test.to_csv(index=False).encode()).hexdigest(),
    'variables': sorted(model.named_vars),
  }
  checkpoint.mkdir(parents=True, exist_ok=True)
  settings_path = checkpoint / 'settings.json'
  if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
    raise ValueError(f'Fold checkpoint settings differ: {checkpoint}')
  settings_path.write_text(json.dumps(settings, indent=2) + '\n')
  path = checkpoint / 'posterior.nc'
  if path.exists():
    idata = az.from_netcdf(path, engine='h5netcdf')
  else:
    with model:
      sampled = pm.sample(**kwargs)
    idata = az.InferenceData(
      posterior=sampled.posterior.drop_vars(
        [name for name, value in sampled.posterior.items() if 'obs' in value.dims]
      ), sample_stats=sampled.sample_stats,
    )
    temporary = checkpoint / 'posterior.tmp.nc'
    idata.to_netcdf(temporary, engine='h5netcdf')
    temporary.replace(path)
  (checkpoint / 'diagnostics.json').write_text(json.dumps(diagnostics(idata), indent=2) + '\n')
  return idata
