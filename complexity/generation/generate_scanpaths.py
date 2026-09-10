"""Generate visual-search scanpaths using an external ScanDiff checkout.

This adapter preserves the recovered feature extraction, diffusion, and decoding
steps. It removes the broken plotting call and records explicit inference inputs.
The original sorted, contiguous worker partitions and seeds are recorded.
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def generate_partition(args: argparse.Namespace, images: list[Path]) -> None:
  """Generate one independent original worker stream, with resumable RNG state."""
  root = args.scandiff_root.resolve()
  checkpoint = (args.checkpoint or root / 'checkpoints/scandiff_visualsearch.pth').resolve()
  embeddings = (args.task_embeddings or root / 'data/task_embeddings.npy').resolve()
  settings = {
    'scandiff_root': str(root), 'checkpoint_sha256': digest(checkpoint),
    'embeddings_sha256': digest(embeddings), 'bboxes_sha256': digest(args.bboxes),
    'generator_sha256': digest(Path(__file__)), 'seed': args.seed, 'device': args.device,
    'images': [str(path.resolve()) for path in images], 'viewers': 10, 'precision': args.precision,
  }
  args.output.mkdir(parents=True, exist_ok=True)
  settings_path = args.output / 'settings.json'
  if any(args.output.iterdir()):
    if not args.resume or not settings_path.exists() or json.loads(settings_path.read_text()) != settings:
      raise ValueError('Use a fresh output directory or --resume with matching settings.')
  settings_path.write_text(json.dumps(settings, indent=2) + '\n')
  boxes = json.loads(args.bboxes.read_text())

  # The external checkout owns the model implementation and its Hydra types.
  sys.path.insert(0, str(root))
  import hydra
  import numpy as np
  import PIL.Image
  import timm
  import torch
  from src.utils.create_diffusion import create_diffusion

  device = torch.device(args.device)
  if device.type == 'cuda':
    if not torch.cuda.is_available():
      raise RuntimeError('CUDA is unavailable. Run in a GPU-enabled environment.')
    torch.cuda.set_device(device)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  with hydra.initialize_config_dir(version_base='1.3', config_dir=str(root / 'configs')):
    cfg: Any = hydra.compose(config_name='demo')
  dino = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0).eval().to(device)
  transforms = timm.data.create_transform(**timm.data.resolve_model_data_config(dino), is_training=False)
  model: Any = hydra.utils.instantiate(cfg.model).eval().to(device)
  model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
  diffusion: Any = create_diffusion(
    cfg, timestep_respacing='', diffusion_steps=cfg.diffusion.num_timesteps,
    noise_schedule=cfg.diffusion.noise_schedule, predict_xstart=cfg.diffusion.predict_xstart
  )
  task_embeddings = np.load(embeddings, allow_pickle=True).item()
  all_scanpaths: list[dict[str, Any]] = []
  state_path = args.output / 'rng_state.pt'
  completed = 0
  if args.resume and state_path.exists():
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    completed = state['completed']
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
      torch.cuda.set_rng_state(state['cuda'], device=device)

  def save_progress(completed_count: int) -> None:
    """Persist the worker's random stream and completed input prefix."""
    state = {
      'completed': completed_count, 'python': random.getstate(), 'numpy': np.random.get_state(),
      'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
    }
    state_tmp = state_path.with_suffix('.tmp')
    torch.save(state, state_tmp)
    state_tmp.replace(state_path)
    progress = args.output / 'progress.json'
    progress_tmp = progress.with_suffix('.tmp')
    progress_tmp.write_text(json.dumps({'completed': completed_count, 'total': len(images)}) + '\n')
    progress_tmp.replace(progress)

  for index, path in enumerate(images):
    task = path.parent.name
    destination = args.output / task / f'{path.stem}.json'
    if index < completed:
      all_scanpaths.extend(json.loads(destination.read_text()))
      continue
    bbox = boxes.get(path.name, {}).get('bbox')
    if task not in task_embeddings:
      raise ValueError(f'No task embedding for {task!r}.')
    try:
      with PIL.Image.open(path) as image:
        original_size = image.size
        tensor = transforms(image.convert('RGB').resize((518, 518))).unsqueeze(0).to(device)
    except (PIL.UnidentifiedImageError, OSError) as error:
      # Original workers skip unreadable images before drawing diffusion noise.
      # Keep their positions in the partition so the other worker streams match.
      skipped_path = args.output / 'skipped_images.json'
      skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
      skipped[str(path)] = {'type': type(error).__name__, 'message': str(error)}
      skipped_tmp = skipped_path.with_suffix('.tmp')
      skipped_tmp.write_text(json.dumps(skipped, indent=2) + '\n')
      skipped_tmp.replace(skipped_path)
      destination.parent.mkdir(parents=True, exist_ok=True)
      destination.write_text('[]')
      save_progress(index + 1)
      print(f'{index + 1}/{len(images)}: skipped unreadable {path}: {error}', flush=True)
      continue
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda' and args.precision == 'amp'):
      features = dino.forward_features(tensor)[:, 5:, :].squeeze().repeat(10, 1, 1)
      embedding = torch.from_numpy(task_embeddings[task]).float().to(device).repeat(10, 1)
      noise = torch.randn(10, cfg.data.max_len, model.scanpath_emb_size, device=device)
      samples = diffusion.p_sample_loop(
        model, noise.shape, noise, clip_denoised=False,
        model_kwargs={'y': features, 'task_embedding': embedding}, progress=False, device=device
      )
      coordinates = model.get_coords_and_time(samples).cpu()
      validity = torch.softmax(model.token_validity_predictor(samples), dim=-1).argmax(dim=-1)
      lengths = torch.cumprod(validity, dim=-1).sum(-1).cpu().long()
    records = []
    for viewer in range(10):
      length = max(1, min(cfg.data.max_len, lengths[viewer].item()))
      scanpath = coordinates[viewer, :length]
      x = (scanpath[:, 0] * original_size[0]).tolist()
      y = (scanpath[:, 1] * original_size[1]).tolist()
      durations = (scanpath[:, 2] * 1000).tolist()
      correct = None if bbox is None else int(any(
        bbox[0] <= fx < bbox[0] + bbox[2] and bbox[1] <= fy < bbox[1] + bbox[3] for fx, fy in zip(x, y)
      ))
      records.append({
        'name': path.name, 'subject': viewer + 1, 'task': task, 'condition': 'present', 'bbox': bbox,
        'X': x, 'Y': y, 'T': durations, 'length': length, 'correct': correct, 'RT': sum(durations),
      })
    if destination.exists() and json.loads(destination.read_text()) != records:
      raise ValueError(f'Recomputed records differ from the saved resume checkpoint: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(records, indent=2))
    temporary.replace(destination)
    save_progress(index + 1)
    all_scanpaths.extend(records)
    print(f'{index + 1}/{len(images)}: {task}/{path.name}', flush=True)
  temporary = args.output / 'all_scanpaths.tmp'
  temporary.write_text(json.dumps(all_scanpaths, indent=2))
  temporary.replace(args.output / 'all_scanpaths.json')


def partition_images(images: list[Path], workers: int) -> list[list[Path]]:
  """Match the original ceil-sized contiguous chunks, including empty workers."""
  if workers < 1:
    raise ValueError('At least one worker is required.')
  size = max(1, math.ceil(len(images) / workers))
  return [images[index * size:(index + 1) * size] for index in range(workers)]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--scandiff-root', type=Path, required=True)
  parser.add_argument('--images', type=Path, required=True)
  parser.add_argument('--bboxes', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--task-embeddings', type=Path)
  parser.add_argument('--device', default='cuda:0', help='Device for single-worker inference.')
  parser.add_argument('--devices', nargs='+', help='One device per logical worker; repeated devices are allowed.')
  parser.add_argument('--precision', choices=['amp', 'float32'], default='amp')
  parser.add_argument('--seed', type=int, default=1000, help='Base seed; worker k uses seed + k.')
  parser.add_argument('--limit', type=int, help='Limit input images before partitioning, for testing.')
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--worker-input', type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_input:
    generate_partition(args, [Path(path) for path in json.loads(args.worker_input.read_text())])
    return
  images = sorted(
    path for path in args.images.glob('*/*')
    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
  )
  if args.limit is not None:
    if args.limit < 1:
      parser.error('--limit must be positive.')
    images = images[:args.limit]
  if not images:
    parser.error('No images found under target subdirectories.')
  if not args.devices:
    generate_partition(args, images)
    return
  partitions = partition_images(images, len(args.devices))
  plan = {
    'generator_sha256': digest(Path(__file__)), 'devices': args.devices, 'base_seed': args.seed, 'precision': args.precision,
    'partitions': [[str(path.resolve()) for path in chunk] for chunk in partitions],
    'scandiff_root': str(args.scandiff_root.resolve()), 'bboxes_sha256': digest(args.bboxes),
    'checkpoint_sha256': digest(args.checkpoint or args.scandiff_root / 'checkpoints/scandiff_visualsearch.pth'),
    'embeddings_sha256': digest(args.task_embeddings or args.scandiff_root / 'data/task_embeddings.npy'),
  }
  args.output.mkdir(parents=True, exist_ok=True)
  plan_path = args.output / 'partition_plan.json'
  if any(args.output.iterdir()):
    if not args.resume or not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
      raise ValueError('Use a fresh directory or resume the same generation plan.')
  plan_path.write_text(json.dumps(plan, indent=2) + '\n')
  processes = []
  streams = []
  try:
    for index, (device, chunk) in enumerate(zip(args.devices, partitions)):
      input_path = args.output / f'worker_{index}_inputs.json'
      input_path.write_text(json.dumps([str(path) for path in chunk]) + '\n')
      worker_output = args.output / 'workers' / str(index)
      if not chunk:
        worker_output.mkdir(parents=True, exist_ok=True)
        (worker_output / 'all_scanpaths.json').write_text('[]')
        continue
      command = [
        sys.executable, str(Path(__file__).resolve()), '--scandiff-root', str(args.scandiff_root),
        '--images', str(args.images), '--bboxes', str(args.bboxes), '--output', str(worker_output),
        '--device', device, '--seed', str(args.seed + index), '--worker-input', str(input_path),
        '--precision', args.precision,
      ]
      for option, value in [('--checkpoint', args.checkpoint), ('--task-embeddings', args.task_embeddings)]:
        if value:
          command += [option, str(value)]
      if args.resume:
        command.append('--resume')
      stream = (args.output / f'worker_{index}.log').open('a' if args.resume else 'w')
      streams.append(stream)
      processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    failures = [process.wait() for process in processes]
    if any(failures):
      raise RuntimeError(f'ScanDiff workers failed with exit codes {failures}. See worker logs.')
  finally:
    for process in processes:
      if process.poll() is None:
        process.terminate()
        process.wait()
    for stream in streams:
      stream.close()
  records = []
  for index in range(len(partitions)):
    records.extend(json.loads((args.output / 'workers' / str(index) / 'all_scanpaths.json').read_text()))
  temporary = args.output / 'all_scanpaths.tmp'
  temporary.write_text(json.dumps(records, indent=2))
  temporary.replace(args.output / 'all_scanpaths.json')
  print(f'Completed {len(images)} inputs across {len(partitions)} workers: {len(records)} scanpaths.', flush=True)


if __name__ == '__main__':
  main()
