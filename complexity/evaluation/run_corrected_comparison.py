"""Run the revised COCO comparison with resumable fits and explicit convergence gates."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from complexity.models import fit_movement_count, fit_response_time
from complexity.models.runner import prepare_loo_likelihood, registry
from complexity.models.sampling import diagnostics


def digest(path: Path) -> str:
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value: Any) -> None:
  """Atomically write heterogeneous scientific results at the JSON boundary."""
  temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
  temporary.write_text(json.dumps(value, indent=2) + '\n')
  temporary.replace(path)


def scientific_functions(path: Path) -> dict[str, str]:
  return {
    node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in ['main', 'cellheldout_cv']
  }


def run_model(config_path: Path, dataset: str, family: str, name: str) -> None:
  """Fit and evaluate one candidate without changing historical artifacts."""
  config = json.loads(config_path.read_text())
  root = Path(config['output'])
  output = root / dataset / family / name
  output.mkdir(parents=True, exist_ok=True)
  csv = Path(config['datasets'][dataset])
  module = fit_response_time if family == 'rt' else fit_movement_count
  source_root = Path(__file__).resolve().parents[1]
  settings = {
    'config_sha256': digest(config_path), 'input_sha256': digest(csv), 'dataset': dataset, 'family': family,
    'model': name, 'python': platform.python_version(),
    'source_sha256': {str(path.relative_to(source_root)): digest(path) for path in sorted(source_root.rglob('*.py'))},
    'packages': {package: importlib.metadata.version(package) for package in
                 ['pymc', 'arviz', 'pytensor', 'numpy', 'pandas', 'scipy', 'xarray', 'h5netcdf']},
  }
  path = output / 'settings.json'
  if path.exists() and json.loads(path.read_text()) != settings:
    raise ValueError(f'Input, source, settings or environment changed: {output}')
  write_json(path, settings)
  status_path = output / 'status.json'
  if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete':
    return
  status: dict[str, Any] = {'status': 'running', 'stage': 'full_fit', 'started': time.time()}
  write_json(status_path, status)
  try:
    data, meta = module.prepare_df(pd.read_csv(csv))
    spec = registry(module, family)[name]
    model = spec[0](data, meta, **spec[1])
    posterior = output / 'posterior.nc'
    donor_record = output / 'reused_posterior.json'
    idata = None
    if posterior.exists():
      idata = module.load_idata(posterior)
    elif donor_record.exists():
      donor = json.loads(donor_record.read_text())
      if digest(Path(donor['path'])) != donor['sha256']:
        raise ValueError('Reused posterior changed.')
      idata = module.load_idata(Path(donor['path']))
    elif dataset == 'synthetic':
      historical = Path(config['historical_root'])
      donor_dir = historical / f'coco_predicted_{family}'
      donor = donor_dir / f'{name}.nc'
      previous = json.loads((donor_dir / 'settings.json').read_text())
      old_source = historical.parent / 'source_snapshot/complexity/models' / Path(module.__file__).name
      if (previous['input_sha256'] == settings['input_sha256'] and donor.exists()
          and scientific_functions(old_source) == scientific_functions(Path(module.__file__))):
        candidate = module.load_idata(donor)
        donor_diagnostics = diagnostics(candidate)
        write_json(output / 'historical_diagnostics.json', donor_diagnostics)
        if donor_diagnostics['passed']:
          idata = candidate
          write_json(donor_record, {'path': str(donor), 'sha256': digest(donor), 'settings': previous})
        else:
          candidate.close()
    if idata is None:
      idata = module.fit(model, **config['full_sampling'])
      temporary = output / 'posterior.tmp.nc'
      module.save_idata(idata, temporary)
      temporary.replace(posterior)
    full_diagnostics = diagnostics(idata)
    write_json(output / 'diagnostics.json', full_diagnostics)
    status['stage'] = 'loo'
    write_json(status_path, status)
    loo_path = output / 'loo.json'
    if not loo_path.exists():
      prepare_loo_likelihood(module, family, spec, idata, data, meta, config['seed'], config['cv_mcn'])
      scorer = module.loo_on_common_logrt if family == 'rt' else module.loo_on_common_logn
      loo, variable = scorer(name, model, idata, data)
      write_json(loo_path, {
        'model': name, 'elpd_loo': float(loo.elpd_loo), 'p_loo': float(loo.p_loo), 'se': float(loo.se),
        'll_var': variable, 'warning': bool(loo.warning), 'max_pareto_k': float(loo.pareto_k.max()),
        'pareto_k_above_0_7': int((loo.pareto_k > 0.7).sum()),
        'good_k': float(loo.good_k), 'pareto_k_above_good_k': int((loo.pareto_k > loo.good_k).sum()),
      })
    idata.close()
    del idata, model
    status['stage'] = 'cv'
    write_json(status_path, status)
    extra = {'mcN': config['cv_mcn']} if family == 'rt' else {}
    cv = module.cellheldout_cv(
      data, meta, {name: spec}, seed=config['seed'], checkpoint_dir=output / 'folds',
      **config['cv_sampling'], **extra,
    )
    write_json(output / 'cv_results.json', cv)
    module.build_model_selection_table(cv).to_csv(output / 'model_selection.csv', index=False)
    folds = sorted((output / 'folds' / name).glob('fold_*/diagnostics.json'))
    scored = cv[name]['n_folds_scored']
    passed = (full_diagnostics['passed'] and len(folds) == scored == config['cv_sampling']['n_splits']
              and all(json.loads(path.read_text())['passed'] for path in folds))
    status.update(status='complete', stage='complete', sampling_passed=bool(passed), finished=time.time())
  except Exception as error:
    status.update(status='failed', error=str(error), finished=time.time())
    raise
  finally:
    write_json(status_path, status)


def summarize(config_path: Path) -> None:
  """Separate computed scores from comparisons that satisfy sampling diagnostics."""
  config = json.loads(config_path.read_text())
  root = Path(config['output'])
  rows: list[dict[str, Any]] = []
  for dataset in config['datasets']:
    for family, module in [('rt', fit_response_time), ('count', fit_movement_count)]:
      if family not in config.get('families', ['rt', 'count']):
        continue
      selection = []
      for name in registry(module, family):
        output = root / dataset / family / name
        path = output / 'status.json'
        row: dict[str, Any] = {'dataset': dataset, 'family': family, 'model': name, 'status': 'pending'}
        if path.exists():
          row.update(json.loads(path.read_text()))
        for filename, key in [('diagnostics.json', 'full_diagnostics'), ('loo.json', 'loo')]:
          if (output / filename).exists():
            row[key] = json.loads((output / filename).read_text())
        if (output / 'model_selection.csv').exists():
          metrics = pd.read_csv(output / 'model_selection.csv').iloc[0].to_dict()
          row['cv'] = metrics
          selection.append({
            'model': name, 'sampling_passed': row.get('sampling_passed', False),
            **{key: value for key, value in metrics.items() if not key.startswith('rank_') and key != 'composite_score'},
            **row.get('loo', {}),
          })
        rows.append(row)
      if selection:
        pd.DataFrame(selection).to_csv(root / f'{dataset}_{family}_comparison.csv', index=False)
  complete = sum(row['status'] == 'complete' for row in rows)
  failed = sum(row['status'] == 'failed' for row in rows)
  report = {
    'completed': complete, 'failed': failed, 'total': len(rows),
    'sampling_passed': sum(bool(row.get('sampling_passed')) for row in rows), 'models': rows,
  }
  write_json(root / 'summary.json', report)
  (root / 'STATUS.md').write_text(
    f'# Corrected COCO comparison\n\nCompleted: {complete}/{len(rows)}. Failed jobs: {failed}. '
    f'Sampling accepted: {report["sampling_passed"]}.\n\n'
    'Completion means the computations finished, not that sampling or PSIS-LOO passed.\n\n'
    '| Dataset | Family | Model | Status | Sampling accepted |\n| --- | --- | --- | --- | --- |\n'
    + ''.join(f'| {row["dataset"]} | {row["family"]} | {row["model"]} | {row["status"]} | '
              f'{row.get("sampling_passed", "pending")} |\n' for row in rows)
  )


def run_all(config_path: Path, workers: int) -> None:
  """Schedule independent candidates; keep failures visible and continue other jobs."""
  config = json.loads(config_path.read_text())
  root = Path(config['output'])
  logs = root / 'logs'
  logs.mkdir(parents=True, exist_ok=True)
  tasks = [(dataset, family, name) for family, module in [('rt', fit_response_time), ('count', fit_movement_count)]
           if family in config.get('families', ['rt', 'count'])
           for name in registry(module, family) for dataset in config['datasets']]
  # Inspect the problematic synthetic joint fit early, while other jobs proceed.
  tasks.sort(key=lambda task: (task[2] != 'M2J_joint_studentT_logN1p', task[1] != 'rt'))

  def launch(task: tuple[str, str, str]) -> int:
    dataset, family, name = task
    command = [sys.executable, '-m', 'complexity.evaluation.run_corrected_comparison', '--config', str(config_path),
               '--dataset', dataset, '--family', family, '--model', name]
    log_path = logs / f'{dataset}_{family}_{name}.log'
    env = os.environ.copy()
    cache = root / 'pytensor_cache' / f'{dataset}_{family}_{name}'
    cache.mkdir(parents=True, exist_ok=True)
    env['PYTENSOR_FLAGS'] = f'base_compiledir={cache}'
    with log_path.open('a') as log:
      print(f'Starting {dataset}/{family}/{name}', flush=True)
      result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
      print(f'Finished {dataset}/{family}/{name}: exit {result.returncode}', flush=True)
    return result.returncode

  summarize(config_path)
  with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(launch, task) for task in tasks]
    for future in as_completed(futures):
      future.result()
      summarize(config_path)
  summarize(config_path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', required=True, type=Path)
  parser.add_argument('--workers', type=int, default=3)
  parser.add_argument('--dataset', choices=['human', 'synthetic'])
  parser.add_argument('--family', choices=['rt', 'count'])
  parser.add_argument('--model')
  parser.add_argument('--summarize', action='store_true')
  args = parser.parse_args()
  if args.summarize:
    summarize(args.config)
  elif args.dataset and args.family and args.model:
    run_model(args.config, args.dataset, args.family, args.model)
  elif any([args.dataset, args.family, args.model]) or args.workers < 1:
    parser.error('Specify all candidate fields, or a positive worker count for the complete comparison.')
  else:
    run_all(args.config, args.workers)


if __name__ == '__main__':
  main()
