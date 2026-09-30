"""Run standalone Route B extraction, preprocessing, training, and evaluation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from route_b.config import configure, paths

STAGES = ('features', 'text', 'prepare', 'transform', 'pretrain', 'train', 'evaluate', 'sweep', 'variance')


def checkpoint_stem(weight: float) -> str:
  """Match the original trainers' filename sanitization for fractional weights."""
  return f'visual_{weight:g}'.replace('.', '_')


def execute(stage: str, weight: float, config_path: Path) -> list[Path]:
  """Execute one stage; retain numerical algorithms and historical fold definitions."""
  import numpy as np
  import torch
  from route_b.runtime import atomic_json
  config = configure(config_path)
  p = paths()
  write_locations = {
    'features': [p.dino, p.clip_image, p.clip_text], 'text': [p.clip_text],
    'prepare': [p.pca], 'transform': [p.pca], 'pretrain': [p.checkpoints], 'train': [p.checkpoints],
  }
  if any(not location.is_relative_to(p.output) for location in write_locations.get(stage, [])):
    raise ValueError('External input artifacts are read-only; generate new artifacts inside the run output')
  stem = checkpoint_stem(weight)
  device = 'cuda:0' if config['devices'] else 'cpu'
  torch.set_num_threads(config['cpu_threads'])
  if stage in ('features', 'text'):
    from route_b.features.clip import build_clip_model, extract_and_save_clip, extract_clip_text_features
    from route_b.constants import COCO_SEARCH18_CATEGORIES
    for directory in (p.dino, p.clip_image, p.clip_text):
      directory.mkdir(parents=True, exist_ok=True)
    if stage == 'features':
      from route_b.features.dino import extract_and_save_dino
      extract_and_save_dino(device=device)
      extract_and_save_clip(device=device)
      return list(p.dino.glob('*.npy')) + list(p.clip_image.glob('*.npy')) + list(p.clip_text.glob('*.npz'))
    model, _, tokenizer = build_clip_model(device=device)
    values = extract_clip_text_features(COCO_SEARCH18_CATEGORIES, model, tokenizer, device=device)
    output = p.clip_text / 'clip_text_embeddings.npz'
    with output.with_suffix('.tmp').open('wb') as stream:
      np.savez(stream, **values)
    output.with_suffix('.tmp').replace(output)
    return [output]
  if stage in ('prepare', 'transform'):
    from route_b.constants import NSD_SUBJECTS
    from route_b.data.roi import apply_subject_pca, load_pca_models, prepare_all_subjects_pca
    from route_b.data.nsd import load_training_fmri
    p.pca.mkdir(parents=True, exist_ok=True)
    for subject in NSD_SUBJECTS:
      marker = p.output / 'completed' / f'{stage}_{subject}.json'
      if marker.exists():
        from route_b.provenance import verify_files
        verify_files(json.loads(marker.read_text()))
        continue
      if stage == 'prepare':
        prepare_all_subjects_pca(subjects=[subject])
      else:
        hemispheres = load_training_fmri(subject)
        transformed = apply_subject_pca(subject, np.concatenate(hemispheres, axis=1), load_pca_models(subject))
        np.save(p.pca / f'{subject}_pca_fmri.npy', transformed)
      from route_b.provenance import fingerprints
      files = list(p.pca.glob(f'{subject}_*'))
      atomic_json(marker, fingerprints(files))
    return list(p.pca.glob('*'))
  if stage == 'pretrain':
    from route_b.training.encoder import train_dnn_decoder
    train_dnn_decoder(**config['encoder'], visual_loss_weight=weight, fmri_recon_weight=1-weight,
      run_name=stem, gpus=list(range(len(config['devices']))), recovery_root=p.output / 'recovery/encoder')
    return [p.checkpoints / 'feature_decoder' / f'{stem}.pt']
  if stage == 'train':
    from route_b.training.complexity import train_complexity_model
    train_complexity_model(**config['head'], run_name=stem,
      pretrained_path=p.checkpoints / 'feature_decoder' / f'{stem}.pt',
      gpus=list(range(len(config['devices']))), recovery_root=p.output / 'recovery/head')
    return [p.checkpoints / 'complexity' / f'{stem}.pt']
  if stage == 'evaluate':
    from route_b.evaluation.reproduction import predict_checkpoint
    directory = p.output / 'results' / f'visual_{weight:g}'
    predict_checkpoint(p.checkpoints / 'complexity' / f'{stem}.pt', directory)
    return list(directory.glob('*'))
  if stage == 'variance':
    from route_b.evaluation.variance import decompose_variance, predict_oof_pair_scores
    scores = predict_oof_pair_scores(p.checkpoints / 'complexity' / f'{stem}.pt', torch.device(device), 256)
    total, within, between, count = decompose_variance(scores, 2)
    result = {'predictions': {'total': total, 'within': within, 'between': between, 'images': count,
                             'within_percent': 100 * within / total}}
    output = p.output / 'results/variance.json'
    atomic_json(output, result)
    return [output]
  raise ValueError(stage)


def worker(args: argparse.Namespace) -> None:
  from route_b import runtime
  from route_b.provenance import fingerprints, input_files, verify_files
  config = configure(args.config)
  root = paths().output
  name = f'{args.worker}_{args.weight:g}'
  inputs = root / 'inputs' / f'{name}.json'
  if inputs.exists():
    verify_files(json.loads(inputs.read_text()))
  else:
    files = input_files(args.worker)
    if args.worker in ('train', 'evaluate', 'variance'):
      folder = 'feature_decoder' if args.worker == 'train' else 'complexity'
      files.append(paths().checkpoints / folder / f'{checkpoint_stem(args.weight)}.pt')
    runtime.atomic_json(inputs, fingerprints(files))
  budget = None
  try:
    if args.worker not in ('prepare', 'transform'):
      import torch
      if not torch.cuda.is_available():
        raise RuntimeError('Route B training and evaluation require a CUDA GPU; standalone prediction also supports CPU')
      budget = runtime.Budget(root / 'budget.json', config['gpu_hours'], len(config['devices']))
      runtime.active_budget = budget
      budget.check()
    artifacts = execute(args.worker, args.weight, args.config)
    runtime.atomic_json(root / 'completed' / f'{name}.json', fingerprints(artifacts))
  except runtime.BudgetExhausted as error:
    print(str(error), flush=True)
    raise SystemExit(75) from error
  finally:
    if budget:
      budget.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--stages', nargs='+', choices=STAGES,
                      default=['features', 'prepare', 'pretrain', 'train', 'evaluate', 'sweep', 'variance'])
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--initialize-only', action='store_true')
  parser.add_argument('--weight', type=float, choices=[0., .25, .5, .75, 1.], default=1.)
  parser.add_argument('--worker', choices=STAGES, help=argparse.SUPPRESS)
  args = parser.parse_args()
  args.config = args.config.resolve()
  config = configure(args.config)
  if args.worker:
    worker(args)
    return
  from route_b.provenance import freeze, verify_files
  from route_b.runtime import atomic_json
  root = paths().output
  root.mkdir(parents=True, exist_ok=True)
  with (root / 'controller.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'manifest.json').exists() and not args.resume:
      parser.error('Existing run requires --resume')
    freeze(config)
    if args.initialize_only:
      return
    tasks = [(stage, args.weight) for stage in STAGES if stage in args.stages and stage != 'sweep']
    if 'sweep' in args.stages:
      position = next((i for i, (stage, _) in enumerate(tasks) if stage == 'variance'), len(tasks))
      tasks[position:position] = [(stage, weight) for weight in (0., .25, .5, .75)
                                  for stage in ('pretrain', 'train', 'evaluate')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, config['devices'])),
               MPLBACKEND='Agg', PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(config['cpu_threads']),
               OPENBLAS_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads']))
    for stage, weight in tasks:
      marker = root / 'completed' / f'{stage}_{weight:g}.json'
      if marker.exists():
        verify_files(json.loads(marker.read_text()))
        verify_files(json.loads((root / 'inputs' / marker.name).read_text()))
        continue
      atomic_json(root / 'status.json', {'state': 'running', 'stage': stage, 'weight': weight, 'time': time.time()})
      print(f'Running {stage}, visual weight {weight:g}', flush=True)
      logs = root / 'logs'
      logs.mkdir(exist_ok=True)
      with (logs / f'{stage}_{weight:g}.log').open('a') as stream:
        code = subprocess.call([sys.executable, '-u', '-m', 'route_b.run', '--config', str(args.config),
                                '--worker', stage, '--weight', str(weight)], env=env,
                               stdout=stream, stderr=subprocess.STDOUT)
      if code:
        atomic_json(root / 'status.json', {'state': 'budget_paused' if code == 75 else 'failed', 'stage': stage, 'code': code})
        raise SystemExit(code)
    from route_b.evaluation.reproduction import report
    if (root / 'results').exists():
      report(root, run_label='Saved checkpoint' if 'checkpoints' in config['inputs'] else 'Fresh training')
    atomic_json(root / 'status.json', {'state': 'complete', 'time': time.time(), 'stages': args.stages})


if __name__ == '__main__':
  main()
