"""Check original and ported feature extraction on real images before reproduction."""

from __future__ import annotations

import gc
import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from route_b.data.nsd import list_training_images
from route_b.runtime import atomic_json, check_budget, digest


def check_features(root: Path) -> None:
  """Compare both extraction implementations with the same cached backbone weights."""
  original_dino = importlib.import_module('src.features.extract_dino')
  original_clip = importlib.import_module('src.features.extract_clip')
  from route_b.features import clip, dino
  images = list_training_images('subj01')[:4]
  if len(images) != 4:
    raise ValueError('Four reference images are required for the feature test')
  repository = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
  if not repository.is_dir():
    raise FileNotFoundError('The pinned local DINOv2 checkout is required for offline feature validation')
  original_load = torch.hub.load
  load = lambda _repository, name: original_load(str(repository), name, source='local')
  results = {'images': {str(image): digest(image) for image in images}, 'rtol': 1e-6, 'atol': 1e-6}
  with patch.object(torch.hub, 'load', side_effect=load):
    values = []
    for module in (original_dino, dino):
      check_budget()
      model, transform = module.build_dino_model(device='cuda:0')
      values.append(module.extract_dino_all_layers(images, model, transform, batch_size=4, device='cuda:0', num_workers=0))
      del model
      gc.collect()
      torch.cuda.empty_cache()
    np.testing.assert_allclose(values[0], values[1], rtol=1e-6, atol=1e-6)
    results['dino_max_absolute_difference'] = float(np.max(np.abs(values[0] - values[1])))
  values, texts = [], []
  for module in (original_clip, clip):
    check_budget()
    model, transform, tokenizer = module.build_clip_model(device='cuda:0')
    values.append(module.extract_clip_image_features(images, model, transform, batch_size=4, device='cuda:0', num_workers=0))
    texts.append(module.extract_clip_text_features(module.COCO_SEARCH18_CATEGORIES, model, tokenizer, device='cuda:0'))
    del model
    gc.collect()
    torch.cuda.empty_cache()
  np.testing.assert_allclose(values[0], values[1], rtol=1e-6, atol=1e-6)
  results['clip_image_max_absolute_difference'] = float(np.max(np.abs(values[0] - values[1])))
  for task in texts[0]:
    np.testing.assert_allclose(texts[0][task], texts[1][task], rtol=1e-6, atol=1e-6)
  results['clip_text_max_absolute_difference'] = max(float(np.max(np.abs(texts[0][task] - texts[1][task])))
                                                   for task in texts[0])
  results['passed'] = True
  atomic_json(root / 'replay/features.json', results)
