"""Run and resume the original Route B paper reproduction from explicit configuration."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from route_b.config import configure, paths


STAGES = ('features', 'prepare', 'pretrain', 'train', 'evaluate', 'sweep', 'variance')


def checkpoint_stem(weight: float) -> str:
  """Match the original trainers' filename sanitization for fractional weights."""
  return f'visual_{weight:g}'.replace('.', '_')


def status(state: str, **details: object) -> None:
  from route_b.runtime import atomic_json
  atomic_json(paths().output / 'status.json', {'state': state, 'time': time.time(), 'pid': os.getpid(), **details})
  print(f'{state}: {details}', flush=True)


def worker(args: argparse.Namespace) -> None:
  """Run one recoverable stage in its own process to isolate memory and RNG state."""
  config = configure(args.config, reference=args.worker.startswith('replay'))
  from route_b.development.provenance import verify
  from route_b.runtime import Budget, BudgetExhausted, atomic_json, digest
  from route_b import runtime
  verify(args.config, config)
  root = paths().output
  if args.worker.startswith('replay'):
    sys.path.insert(0, str(root / 'source_snapshot/original'))
  import torch
  torch.set_num_threads(config['cpu_threads'])
  torch.set_num_interop_threads(1)
  budget = None
  if args.worker != 'prepare':
    if not torch.cuda.is_available():
      raise RuntimeError('CUDA is unavailable; preserved checkpoints can be resumed later')
    for device in range(len(config['devices'])):
      while torch.cuda.mem_get_info(device)[0] < 12 * 2**30:
        print(f'Waiting for 12 GiB free on configured GPU {device}', flush=True)
        time.sleep(15)
    budget = Budget(root / 'budget.json', config['gpu_hours'], len(config['devices']))
    runtime.active_budget = budget
    budget.check()
  try:
    execute_task(args)
  except BudgetExhausted as error:
    status('budget_paused', task=args.worker, error=str(error))
    raise SystemExit(75) from error
  finally:
    if budget:
      budget.close()
  selected = []
  label = f'visual_{args.weight:g}'
  if args.worker == 'prepare':
    selected = list(paths().pca.glob(f'{args.subject}_*'))
  elif args.worker in ('pretrain', 'train'):
    folder = 'feature_decoder' if args.worker == 'pretrain' else 'complexity'
    selected = [paths().checkpoints / folder / f'{checkpoint_stem(args.weight)}.pt']
  elif args.worker in ('replay', 'evaluate'):
    folder = root / ('replay' if args.worker == 'replay' else 'results') / label
    selected = [path for path in folder.rglob('*') if path.is_file() and path.suffix != '.tmp']
  elif 'variance' in args.worker:
    selected = [root / ('replay' if args.worker.startswith('replay') else 'results') / 'variance.json']
  elif args.worker == 'replay_features':
    selected = [root / 'replay/features.json']
  elif args.worker == 'features':
    selected = list((root / 'features').rglob('*.np*'))
  artifacts = {str(path): {'sha256': digest(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
               for path in selected}
  atomic_json(root / 'completed' / f'{args.worker}_{args.weight:g}_{args.subject}.json',
              {'finished': time.time(), 'artifacts': artifacts})


def execute_task(args: argparse.Namespace) -> None:
  """Dispatch original algorithms with the fixed, recorded paper settings."""
  import torch
  from route_b.config import read_config
  from route_b.runtime import atomic_json
  config = read_config(args.config)
  root = paths().output
  weight = args.weight
  label = f'visual_{weight:g}'
  if args.worker == 'prepare':
    from route_b.data.roi import prepare_all_subjects_pca
    prepare_all_subjects_pca(subjects=[args.subject])
  elif args.worker == 'pretrain':
    from route_b.training.encoder import train_dnn_decoder
    train_dnn_decoder(**config['encoder'], visual_loss_weight=weight, fmri_recon_weight=1-weight,
                       run_name=label, gpus=list(range(len(config['devices']))), recovery_root=root / 'recovery/encoder')
  elif args.worker == 'train':
    from route_b.training.complexity import train_complexity_model
    train_complexity_model(**config['head'], run_name=label,
      pretrained_path=paths().checkpoints / 'feature_decoder' / f'{checkpoint_stem(weight)}.pt',
      gpus=list(range(len(config['devices']))), recovery_root=root / 'recovery/head')
  elif args.worker == 'evaluate':
    from route_b.evaluation.reproduction import predict_checkpoint
    predict_checkpoint(paths().checkpoints / 'complexity' / f'{checkpoint_stem(weight)}.pt', root / 'results' / label)
  elif args.worker == 'replay':
    import numpy as np
    from route_b.evaluation.reproduction import REFERENCE_HEADS, compare, metrics, predict_checkpoint
    checkpoint = paths().checkpoints / 'complexity' / REFERENCE_HEADS[weight]
    folder = root / 'replay' / label
    original = predict_checkpoint(checkpoint, folder / 'original', original=True)
    ported = predict_checkpoint(checkpoint, folder / 'ported')
    differences = compare(original, ported)
    paper = next(item for item in json.loads((root / 'reference/routeb_phase3_sweep_metrics.json').read_text())
                 if item['visual_weight'] == weight)
    observed = metrics(ported)
    for key, reference_key in [('pearson', 'full_r'), ('spearman', 'full_rho'),
                                ('pearson_residual', 'residual_r'), ('mae', 'full_mae')]:
      np.testing.assert_allclose(observed[key], paper[reference_key], rtol=1e-6, atol=1e-6, err_msg=key)
    if weight == 1:
      baseline = json.loads((root / 'reference/analysis_metrics.json').read_text())
      for key in ('pearson', 'spearman', 'pearson_residual', 'spearman_residual', 'mae', 'r2', 'mean_sigma'):
        np.testing.assert_allclose(observed[key], baseline[key], rtol=1e-6, atol=1e-6, err_msg=key)
    atomic_json(folder / 'equivalence.json', {'passed': True, 'matches_paper_metrics': True,
                                           'max_absolute_differences': differences, 'rtol': 1e-6, 'atol': 1e-6})
  elif args.worker in ('variance', 'replay_variance'):
    from complexity.evaluation.compute_variance import decompose
    from route_b.evaluation.variance import decompose_variance, predict_oof_pair_scores
    reference = args.worker == 'replay_variance'
    checkpoint = paths().checkpoints / 'complexity' / ('multisubj_best.pt' if reference else 'visual_1.pt')
    scores = predict_oof_pair_scores(checkpoint, torch.device('cuda:0'), 256)
    total, within, between, count = decompose_variance(scores, 2)
    result = {'predictions': {'total': total, 'within': within, 'between': between, 'images': count,
                              'within_percent': 100 * within / total}, 'ground_truth': decompose(paths().ranking)}
    if count != 1173 or result['ground_truth']['images'] != 1205:
      raise ValueError('Variance image population changed')
    if reference:
      import importlib.util
      import numpy as np
      spec = importlib.util.spec_from_file_location('original_variance',
        root / 'source_snapshot/original/scripts/13_compute_predicted_complexity_variance.py')
      original = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(original)
      original_scores = original.predict_oof_pair_scores(checkpoint, torch.device('cuda:0'), 256)
      for image, tasks in original_scores.items():
        if set(tasks) != set(scores[image]):
          raise ValueError('Variance target population changed')
        for task, value in tasks.items():
          np.testing.assert_allclose(value, scores[image][task], rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose([total, within, between], [.025626, .007898, .017728], rtol=1e-6, atol=1e-6)
    atomic_json(root / ('replay' if reference else 'results') / 'variance.json', result)
  elif args.worker == 'replay_features':
    from route_b.tests.feature_smoke import check_features
    check_features(root)
  elif args.worker == 'features':
    from dataclasses import replace
    from route_b import config as configuration
    configuration._paths = replace(paths(), dino=root / 'features/dino', clip_image=root / 'features/clip_image',
                                   clip_text=root / 'features/clip_text')
    for directory in (paths().dino, paths().clip_image, paths().clip_text):
      directory.mkdir(parents=True, exist_ok=True)
    from route_b.features.dino import extract_and_save_dino
    from route_b.features.clip import extract_and_save_clip
    extract_and_save_dino()
    extract_and_save_clip()
  else:
    raise ValueError(args.worker)


def launch(args: argparse.Namespace, task: str, weight: float = 1., subject: str = '') -> None:
  """Run a stage once, reusing complete artifacts and retaining logs across restarts."""
  root = paths().output
  name = f'{task}_{weight:g}_{subject}'
  marker = root / 'completed' / (name + '.json')
  if marker.exists():
    from route_b.runtime import digest
    for filename, saved in json.loads(marker.read_text())['artifacts'].items():
      path = Path(filename)
      stat = path.stat()
      if stat.st_size != saved['size'] or stat.st_mtime_ns != saved['mtime_ns']:
        if digest(path) != saved['sha256']:
          raise ValueError(f'Completed artifact changed: {filename}')
    return
  status('running', task=task, visual_weight=weight, subject=subject)
  log = root / 'logs' / (name + '.log')
  log.parent.mkdir(parents=True, exist_ok=True)
  environment = dict(os.environ)
  config = json.loads(args.config.read_text())
  environment['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['devices']))
  environment['MPLBACKEND'] = 'Agg'
  environment['HF_HUB_OFFLINE'] = '1'
  command = [sys.executable, '-u', '-m', 'route_b.run', '--config', str(args.config),
             '--verify-original', '--worker', task, '--weight', str(weight), '--subject', subject, '--resume']
  with log.open('a') as stream:
    code = subprocess.call(command, cwd=root / 'source_snapshot/publication', env=environment,
                           stdout=stream, stderr=subprocess.STDOUT)
  if code == 75:
    raise SystemExit(75)
  if code:
    raise RuntimeError(f'{name} failed with exit {code}; inspect {log}')


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--stages', nargs='+', choices=STAGES,
                      default=['prepare', 'pretrain', 'train', 'evaluate', 'sweep', 'variance'])
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--initialize-only', action='store_true')
  parser.add_argument('--worker', help=argparse.SUPPRESS)
  parser.add_argument('--weight', type=float, default=1., help=argparse.SUPPRESS)
  parser.add_argument('--subject', default='', help=argparse.SUPPRESS)
  args = parser.parse_args()
  args.config = args.config.resolve()
  if args.worker:
    worker(args)
    return
  config = configure(args.config)
  root = paths().output
  root.mkdir(parents=True, exist_ok=True)
  with (root / 'controller.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'manifest.json').exists() and not args.resume:
      parser.error('Existing run requires --resume')
    from route_b.development.provenance import freeze
    status('initializing')
    freeze(args.config, config)
    if args.initialize_only:
      status('ready')
      return
    if Path.cwd().resolve() != root / 'source_snapshot/publication':
      raise ValueError('Launch the frozen source snapshot after initialization')
    from route_b.constants import NSD_SUBJECTS
    from route_b.evaluation.reproduction import report
    try:
      launch(args, 'replay_features')
      for weight in (1., 0., .25, .5, .75):
        launch(args, 'replay', weight)
      launch(args, 'replay_variance')
      if 'features' in args.stages:
        launch(args, 'features')
      if 'prepare' in args.stages:
        for subject in NSD_SUBJECTS:
          launch(args, 'prepare', subject=subject)
      for task in ('pretrain', 'train', 'evaluate'):
        if task in args.stages:
          launch(args, task)
      if 'sweep' in args.stages:
        for weight in (0., .25, .5, .75):
          for task in ('pretrain', 'train', 'evaluate'):
            launch(args, task, weight)
      if 'variance' in args.stages:
        launch(args, 'variance')
      report(root)
      status('complete')
    except Exception:
      status('failed', error=traceback.format_exc())
      raise


if __name__ == '__main__':
  main()
