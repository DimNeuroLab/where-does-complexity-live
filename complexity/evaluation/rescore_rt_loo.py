"""Recompute RT LOO from completed posteriors without refitting or replacing them."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from complexity.models import fit_response_time as models
from complexity.models.runner import prepare_loo_likelihood, registry


def digest(path: Path) -> str:
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
  parser.add_argument('--results-dir', type=Path, required=True)
  parser.add_argument('--trials-file', dest='csv', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()
  if args.output_dir.resolve() == args.results_dir.resolve():
    parser.error('Use a separate output directory to preserve the original scores.')
  if args.output_dir.exists() and any(args.output_dir.iterdir()):
    parser.error('Use a fresh output directory.')
  settings = json.loads((args.results_dir / 'settings.json').read_text())
  manifest = json.loads((args.results_dir / 'manifest.json').read_text())
  if settings['input_sha256'] != digest(args.csv):
    parser.error('The trial table does not match the recorded fit input.')
  if settings['source_sha256'] != digest(Path(models.__file__)):
    parser.error('The scientific model source differs from the fitted source.')
  if any(manifest['models'].get(name) != 'complete' for name in settings['models']):
    parser.error('All requested model posteriors must be complete.')
  args.output_dir.mkdir(parents=True, exist_ok=True)
  specs = registry(models, 'rt')
  data, meta = models.prepare_df(pd.read_csv(args.csv))
  provenance = {
    'original_settings': settings, 'results_dir': str(args.results_dir.resolve()),
    'scorer_sha256': digest(Path(__file__)),
    'runner_sha256': digest(Path(__file__).resolve().parents[1] / 'models/runner.py'),
    'mc_n': max(20, settings['cv_mcn']), 'posterior_sha256': {}, 'status': 'running',
  }
  tables = []
  for name in settings['models']:
    print(f'Scoring {name}', flush=True)
    posterior = args.results_dir / f'{name}.nc'
    idata = models.load_idata(posterior)
    try:
      prepare_loo_likelihood(models, 'rt', specs[name], idata, data, meta, settings['seed'], settings['cv_mcn'])
      table = models.compare_loo({name: (None, idata)}, data)
      table.to_csv(args.output_dir / f'{name}_loo.csv')
      tables.append(table)
    finally:
      idata.close()
    provenance['posterior_sha256'][name] = digest(posterior)
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
  pd.concat(tables).sort_values('elpd_loo', ascending=False).to_csv(args.output_dir / 'loo_table.csv')
  provenance['status'] = 'complete'
  (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
  main()
