"""Paired checkpoint replay and reports for original-behavior reproduction."""

from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from route_b.constants import NSD_SUBJECTS, SEED
from route_b.data.nsd import build_nsd_id_index
from route_b.runtime import atomic_json, check_budget
from route_b.types import Array, Metrics


REFERENCE_HEADS = {
  1.0: 'multisubj_best.pt', 0.0: 'routeb_fmri_only_pretrain_aux0_frozen.pt',
  0.25: 'routeb_mixed_v0p25_f0p75_aux0_frozen.pt', 0.5: 'routeb_mixed_v0p5_f0p5_aux0_frozen.pt',
  0.75: 'routeb_mixed_v0p75_f0p25_aux0_frozen.pt',
}


def predict_checkpoint(checkpoint: Path, directory: Path, original: bool = False) -> dict[str, Array]:
  """Execute the original evaluation helpers or their port on exactly the same records.

  :param original: Import the preserved source snapshot for independent comparison.
  """
  prefix = 'src' if original else 'route_b'
  backend: ModuleType = importlib.import_module(prefix + ('.analysis.evaluate' if original else '.evaluation.metrics'))
  data: ModuleType = importlib.import_module(prefix + '.data.datasets')
  state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
  means = state['category_means']
  records = data.load_complexity_records()
  indices = {id(record): index for index, record in enumerate(records)}
  folds = data.complexity_stratified_group_kfold(records, 5, SEED)
  if len(state['fold_state_dicts']) != 5:
    raise ValueError('Five fold states are required for paper replay')
  directory.mkdir(parents=True, exist_ok=True)
  collected: dict[str, list[Array]] = {}
  rows = {subject: build_nsd_id_index(subject) for subject in NSD_SUBJECTS}
  for fold, (train, validation) in enumerate(folds, 1):
    destination = directory / f'fold_{fold}.npz'
    if not destination.exists():
      check_budget()
      loader, dimension = backend._build_val_loader(train, validation, means)
      model = backend._build_eval_model(dimension, 'cuda:0')
      model.load_state_dict(state['fold_state_dicts'][fold - 1])
      prediction, log_variance, truth, category, subject = backend._run_inference(model, loader, 'cuda:0')
      ordered = [record for name in NSD_SUBJECTS for record in validation if record['nsd_id'] in rows[name]]
      if len(ordered) != len(prediction):
        raise ValueError('Prediction observation ordering differs from the data inventory')
      offset = np.asarray([means[record['task']] for record in ordered], dtype=float)
      values = {'residual_prediction': prediction, 'log_variance': log_variance, 'residual_truth': truth,
                'category': category, 'subject': subject, 'category_mean': offset,
                'prediction': prediction + offset, 'truth': truth + offset,
                'record_index': np.asarray([indices[id(record)] for record in ordered]),
                'nsd_id': np.asarray([record['nsd_id'] for record in ordered]),
                'fold': np.full(len(prediction), fold, dtype=np.int64)}
      with destination.with_suffix('.tmp').open('wb') as stream:
        np.savez(stream, **values)
      os.replace(destination.with_suffix('.tmp'), destination)
      del model, loader
      torch.cuda.empty_cache()
      print(f'{"original" if original else "ported"} {checkpoint.name}: completed fold {fold}', flush=True)
    with np.load(destination) as saved:
      for key in saved.files:
        collected.setdefault(key, []).append(saved[key])
  result = {key: np.concatenate(values) for key, values in collected.items()}
  if len(result['truth']) != 18736:
    raise ValueError('Expected 18,736 subject-record observations')
  atomic_json(directory / 'metrics.json', summarize(result))
  write_predictions(directory / 'predictions.csv', result)
  return result


def compare(reference: dict[str, Array], ported: dict[str, Array]) -> dict[str, float]:
  """Enforce fixed tolerances without adjusting them after observing results."""
  differences = {}
  for key in reference:
    if key in ('category', 'subject', 'record_index', 'nsd_id', 'fold'):
      np.testing.assert_array_equal(reference[key], ported[key], err_msg=key)
    else:
      np.testing.assert_allclose(reference[key], ported[key], rtol=1e-6, atol=1e-6, err_msg=key)
      differences[key] = float(np.max(np.abs(reference[key] - ported[key])))
  for key, value in metrics(reference).items():
    np.testing.assert_allclose(value, metrics(ported)[key], rtol=1e-6, atol=1e-6, err_msg=key)
  return differences


def metrics(values: dict[str, Array]) -> dict[str, float]:
  """Compute the paper's full-score and category-residual metrics."""
  prediction, truth = values['prediction'], values['truth']
  return {'pearson': float(pearsonr(prediction, truth).statistic),
          'spearman': float(spearmanr(prediction, truth).statistic),
          'pearson_residual': float(pearsonr(values['residual_prediction'], values['residual_truth']).statistic),
          'spearman_residual': float(spearmanr(values['residual_prediction'], values['residual_truth']).statistic),
          'mae': float(np.mean(np.abs(prediction - truth))), 'r2': float(r2_score(truth, prediction)),
          'mean_sigma': float(np.mean(np.exp(.5 * values['log_variance']))), 'n': len(truth)}


def summarize(values: dict[str, Array]) -> Metrics:
  """Include pooled, per-subject, and per-category comparisons."""
  result: Metrics = {'pooled': metrics(values)}
  for field, names in [('subject', NSD_SUBJECTS), ('category', list(importlib.import_module(
      'route_b.data.datasets').CATEGORY_TO_IDX))]:
    result['per_' + field] = {name: metrics({key: value[values[field] == index] for key, value in values.items()})
                              for index, name in enumerate(names)}
  return result


def write_predictions(path: Path, values: dict[str, Array]) -> None:
  """Export complete observations with explicit physical-image and fold identities."""
  temporary = path.with_suffix('.tmp')
  keys = list(values)
  with temporary.open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(keys)
    writer.writerows(zip(*(values[key] for key in keys), strict=True))
  os.replace(temporary, path)


def report(root: Path, run_label: str = 'Fresh training') -> None:
  """Compare checkpoint replay and fresh training without hiding failed reproduction."""
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  results: dict[str, Metrics] = {}
  lines = ['# Original Route B reproduction', '',
           (
             'Checkpoint replay tests port equivalence. Fresh training tests reproduction'
             ' under recorded settings.'
           ), '',
           '| Visual weight | Run | Pearson | Spearman | Residual Pearson | MAE |',
           '| --- | --- | --- | --- | --- | --- |']
  for weight in (0., .25, .5, .75, 1.):
    for kind, subdirectory in [('Reference', f'replay/visual_{weight:g}/original'),
                                ('Port replay', f'replay/visual_{weight:g}/ported'),
                                (run_label, f'results/visual_{weight:g}')]:
      path = root / subdirectory / 'metrics.json'
      if not path.exists():
        continue
      result = json.loads(path.read_text())
      results[f'{weight:g}/{kind}'] = result
      pooled = result['pooled']
      lines.append(f'| {weight:g} | {kind} | {pooled["pearson"]:.6f} | {pooled["spearman"]:.6f} | '
                   f'{pooled["pearson_residual"]:.6f} | {pooled["mae"]:.6f} |')
  variance_path = root / 'results/variance.json'
  if variance_path.exists():
    results['variance'] = json.loads(variance_path.read_text())
  atomic_json(root / 'comparison.json', results)
  (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
  figure, axes = plt.subplots(1, 2, figsize=(11, 4))
  for label in ('Reference', run_label):
    weights = [weight for weight in (0., .25, .5, .75, 1.) if f'{weight:g}/{label}' in results]
    for axis, metric in zip(axes, ('pearson', 'pearson_residual'), strict=True):
      axis.plot(weights, [results[f'{weight:g}/{label}']['pooled'][metric] for weight in weights], 'o-', label=label)
      axis.set(xlabel='Visual pretraining weight', ylabel=metric.replace('_', ' '))
      axis.legend()
  figure.tight_layout()
  figure.savefig(root / 'comparison.pdf')
  figure.savefig(root / 'comparison.png', dpi=200)
  plt.close(figure)
