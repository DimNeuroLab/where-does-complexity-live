"""Check label identity and numerical agreement with the preserved Route B ranking."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
  parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
  parser.add_argument('--ranking-file', dest='ranking', type=Path, required=True)
  parser.add_argument('--reference-file', dest='reference', type=Path, required=True)
  parser.add_argument('--output-file', dest='output', type=Path, required=True)
  parser.add_argument('--mode', choices=['export', 'refit'], required=True)
  parser.add_argument('--diagnostics-file', dest='diagnostics', type=Path, help='Required for checking a fresh fit.')
  args = parser.parse_args()
  actual, reference = (pd.read_csv(path, float_precision='round_trip') for path in [args.ranking, args.reference])
  for frame in [actual, reference]:
    if frame['image'].duplicated().any() or not np.isfinite(frame['score']).all():
      raise ValueError('Ranking contains duplicate identifiers or nonfinite scores.')
  same_labels = set(actual['image']) == set(reference['image'])
  aligned = actual.merge(reference, on='image', suffixes=('_actual', '_reference'), validate='one_to_one')
  difference = aligned['score_actual'] - aligned['score_reference']
  same_tasks = bool((aligned['task_actual'] == aligned['task_reference']).all())
  rho = float(spearmanr(aligned['score_actual'], aligned['score_reference']).statistic)
  rmse = float(np.sqrt(np.mean(difference ** 2)))
  maximum = float(abs(difference).max())
  same_order = actual['image'].tolist() == reference['image'].tolist()
  numerical = maximum <= 1e-12 and same_order if args.mode == 'export' else rho >= 0.99 and rmse <= 0.02
  diagnostics = None
  converged = True
  if args.mode == 'refit':
    if args.diagnostics is None:
      parser.error('--diagnostics-file is required for refit verification.')
    diagnostics = json.loads(args.diagnostics.read_text())
    converged = (
      diagnostics['divergences'] == 0 and diagnostics['max_rhat'] is not None
      and diagnostics['max_rhat'] <= 1.01
      and diagnostics['min_ess_bulk'] >= 400 and diagnostics['min_ess_tail'] >= 400
    )
  report = {
    'mode': args.mode, 'rows': len(actual), 'reference_rows': len(reference), 'shared_rows': len(aligned),
    'added_labels': sorted(set(actual['image']) - set(reference['image'])),
    'missing_labels': sorted(set(reference['image']) - set(actual['image'])),
    'same_labels': same_labels, 'same_tasks': same_tasks,
    'same_order': same_order, 'spearman': rho, 'rmse': rmse, 'max_absolute_difference': maximum,
    'byte_identical': args.ranking.read_bytes() == args.reference.read_bytes(),
    'criteria': {
      'export_max_absolute_difference': 1e-12, 'refit_min_spearman': 0.99, 'refit_max_rmse': 0.02,
      'refit_max_rhat': 1.01, 'refit_min_ess': 400, 'refit_max_divergences': 0,
    },
    'diagnostics': diagnostics, 'converged': converged, 'numerical_agreement': bool(numerical),
    'passed': bool(same_labels and same_tasks and numerical and converged),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))
  if not report['passed']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
