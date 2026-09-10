"""Run the documented complexity flow from saved scanpaths or prepared images."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def execute(module: str, arguments: list[str], log: Path, python: str = sys.executable) -> None:
  """Run a stage and retain its complete output; propagate failures."""
  command = [python, '-m', module, *arguments]
  print('Running:', ' '.join(command), flush=True)
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open('w') as stream:
    stream.write(json.dumps(command) + '\n')
    stream.flush()
    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--profile', choices=['nsd', 'coco', 'all'], default='nsd')
  parser.add_argument('--stages', nargs='+', choices=['generate', 'prepare', 'fit', 'export', 'compare', 'variance', 'verify'])
  parser.add_argument('--posterior', type=Path, help='Export and verify an existing NSD posterior instead of a fresh fit.')
  parser.add_argument('--resume', action='store_true')
  args = parser.parse_args()
  config: dict[str, Any] = json.loads(os.path.expandvars(args.config.read_text()))
  required_paths = [config['output_dir'], *config['inputs'].values()]
  if any(re.search(r'\$\{[A-Za-z_]\w*\}', value) for value in required_paths):
    parser.error('Set the data/output environment variables used in the config, or replace them with paths.')
  output = Path(config['output_dir']).expanduser().resolve()
  inputs = {key: Path(value).expanduser().resolve() for key, value in config['inputs'].items()}
  output.mkdir(parents=True, exist_ok=True)
  profiles = ['coco', 'nsd'] if args.profile == 'all' else [args.profile]
  stages = args.stages or ['prepare', 'fit', 'export', 'compare', 'variance', 'verify']
  trials = output / 'trials'
  trials.mkdir(exist_ok=True)
  sampling = config.get('sampling', {})
  sampling_args = []
  for key in ['draws', 'tune', 'chains', 'seed', 'target_accept', 'cv_splits', 'cv_draws', 'cv_tune', 'cv_chains']:
    if key in sampling:
      sampling_args += ['--' + key.replace('_', '-'), str(sampling[key])]
  if args.resume:
    sampling_args += ['--resume']
  record = {'config': config, 'profile': args.profile, 'stages': stages, 'python': sys.executable}
  (output / 'last_run.json').write_text(json.dumps(record, indent=2) + '\n')

  def run(module: str, argv: list[str], name: str, python: str = sys.executable) -> None:
    execute('complexity.' + module, argv, output / 'logs' / f'{name}.log', python)

  if 'generate' in stages:
    for profile in profiles:
      generation = {**config['generation'], **config['generation'].get('profiles', {}).get(profile, {})}
      destination = output / 'scanpaths' / profile
      argv = [
        '--scandiff-root', generation['checkout'], '--images', str(inputs[f'{profile}_images']),
        '--bboxes', str(inputs[f'{profile}_bboxes']), '--output', str(destination),
        '--device', generation.get('device', 'cuda:0'), '--seed', str(generation.get('seed', 1000)),
        '--precision', generation.get('precision', 'amp'),
      ]
      if generation.get('devices'):
        argv += ['--devices', *generation['devices']]
      if args.resume:
        argv.append('--resume')
      run('generation.generate_scanpaths', argv, f'generate_{profile}', generation['python'])
  for profile in profiles:
    generated = output / 'scanpaths' / profile / 'all_scanpaths.json'
    scanpaths = generated if 'generate' in stages else inputs[f'{profile}_scanpaths']
    if 'prepare' in stages:
      if profile == 'coco':
        run('preprocessing.convert_scanpaths', [
          '--in_json', str(inputs['human_scanpaths']), '--out_csv', str(trials / 'human_all.csv'),
          '--movement', 'saccades', '--only_condition', 'present',
        ], 'convert_human')
        run('preprocessing.select_coco_trials', [
          '--csv', str(trials / 'human_all.csv'), '--output', str(trials / 'human.csv'),
        ], 'select_human')
      run('preprocessing.convert_scanpaths', [
        '--in_json', str(scanpaths), '--out_csv', str(trials / f'{profile}_predicted.csv'),
        '--movement', 'saccades', '--only_condition', 'present', '--only_correct', '1',
      ], f'convert_{profile}')
      if profile == 'nsd':
        run('preprocessing.correct_nsd_targets', [
          '--csv', str(trials / 'nsd_predicted.csv'), '--scanpaths', str(scanpaths),
          '--metadata', str(inputs['nsd_metadata']), '--output', str(trials / 'nsd_corrected.csv'),
        ], 'correct_nsd')
    if 'fit' in stages:
      if profile == 'nsd':
        if args.posterior:
          parser.error('Do not combine --posterior with the fit stage.')
        run('models.fit_movement_count', [
          '--csv', str(trials / 'nsd_corrected.csv'), '--results-dir', str(output / 'nsd_fit'),
          '--models', 'M2_n_studentT', '--skip-cv', *sampling_args,
        ], 'fit_nsd')
      else:
        for source, table in [('human', 'human.csv'), ('predicted', 'coco_predicted.csv')]:
          for family, script in [('rt', 'fit_response_time'), ('count', 'fit_movement_count')]:
            argv = ['--csv', str(trials / table), '--results-dir', str(output / f'coco_{source}_{family}'), *sampling_args]
            selected = config.get('coco_models', {}).get(family)
            if selected:
              argv += ['--models', *selected]
            run(f'models.{script}', argv, f'fit_coco_{source}_{family}')
        run('models.mean_count_baseline', [
          '--csv', str(trials / 'coco_predicted.csv'), '--output_dir', str(output / 'rankings' / 'predicted_mean'),
          '--cv_splits', str(sampling.get('cv_splits', 5)), '--seed', str(sampling.get('seed', 42)),
        ], 'predicted_mean')
    if 'export' in stages:
      if profile == 'nsd':
        posterior = args.posterior or output / 'nsd_fit' / 'M2_n_studentT.nc'
        run('exports.export_rankings', [
          '--posterior', str(posterior), '--trials', str(trials / 'nsd_corrected.csv'),
          '--output-dir', str(output / 'rankings' / 'nsd'),
        ], 'export_nsd')
      else:
        for source, table in [('human', 'human.csv'), ('predicted', 'coco_predicted.csv')]:
          for family in ['rt', 'count']:
            run('exports.export_rankings', [
              '--results-dir', str(output / f'coco_{source}_{family}'), '--trials', str(trials / table),
              '--output-dir', str(output / 'rankings' / f'{source}_{family}'),
            ], f'export_{source}_{family}')
  if 'compare' in stages and 'coco' in profiles:
    rankings = output / 'rankings'
    for name, left, right in [
      ('fig3a', 'human_count/full_ranking_M2_n_studentT.csv', 'human_rt/full_ranking_M2_logrt_studentT.csv'),
      ('fig3b', 'predicted_mean/full_ranking_mean_N.csv', 'human_rt/full_ranking_M2_logrt_studentT.csv'),
      ('fig4a', 'predicted_count/full_ranking_M2_n_studentT.csv', 'human_count/full_ranking_M2_n_studentT.csv'),
      ('fig4b', 'predicted_rt/full_ranking_M2_logrt_studentT.csv', 'human_rt/full_ranking_M2_logrt_studentT.csv'),
    ]:
      run('evaluation.compare_ranking_pair', [
        '--file1', str(rankings / left), '--file2', str(rankings / right),
        '--out-dir', str(output / 'comparisons' / name),
      ], name)
  if 'variance' in stages and 'nsd' in profiles:
    run('evaluation.compute_variance', [str(output / 'rankings/nsd/full_ranking_M2_n_studentT.csv')], 'variance')
  if 'verify' in stages and 'nsd' in profiles:
    diagnostic_args = [] if args.posterior else [
      '--diagnostics', str(output / 'nsd_fit/M2_n_studentT_diagnostics.json')
    ]
    run('evaluation.verify_ranking', [
      '--ranking', str(output / 'rankings/nsd/full_ranking_M2_n_studentT.csv'),
      '--reference', str(ROOT / 'complexity/reference/nsd_m2_ranking.csv'),
      '--output', str(output / 'verification.json'),
      '--mode', 'export' if args.posterior else 'refit',
      *diagnostic_args,
    ], 'verify')
  print(f'Completed requested stages. Outputs: {output}', flush=True)


if __name__ == '__main__':
  main()
