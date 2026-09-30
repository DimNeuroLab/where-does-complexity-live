"""Extract DINOv2 ViT-L/14 features - all 24 intermediate CLS tokens per image."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from route_b.config import paths as get_paths
from route_b.constants import DINO_MODEL_NAME, DINO_NUM_LAYERS, NSD_SUBJECTS
from route_b.data.nsd import list_training_images
from route_b.features.protocols import DinoModel


class ImagePathDataset(Dataset):

  def __init__(self, paths: list[Path], transform: Callable[[Image.Image], torch.Tensor]) -> None:
    self.paths = paths
    self.transform = transform

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
    try:
      img = Image.open(self.paths[idx]).convert('RGB')
    except Exception:
      img = Image.new('RGB', (224, 224), (128, 128, 128))
    return (self.transform(img), idx)


def build_dino_model(
  model_name: str = DINO_MODEL_NAME, device: str = 'cuda',
) -> tuple[DinoModel, Callable[[Image.Image], torch.Tensor]]:
  """Load DINOv2 ViT-L/14 and return (model, transform)."""
  repository = get_paths().dino_repository
  model = torch.hub.load(repository, model_name, source='local') if repository else torch.hub.load(
    'facebookresearch/dinov2:7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c', model_name)
  model = model.to(device).eval()
  transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return (model, transform)


@torch.no_grad()
def extract_dino_all_layers(
  image_paths: list[Path],
  model: DinoModel,
  transform: Callable[[Image.Image], torch.Tensor],
  batch_size: int = 64,
  device: str = 'cuda',
  num_workers: int = 8
) -> np.ndarray:
  """Extract CLS tokens from ALL transformer blocks."""
  dataset = ImagePathDataset(image_paths, transform)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  n_layers = len(model.blocks)
  all_layer_cls = []
  for (imgs, _) in tqdm(loader, desc='DINO multi-layer'):
    imgs = imgs.to(device, non_blocking=True)
    x = model.prepare_tokens_with_masks(imgs)
    layer_cls = []
    for blk in model.blocks:
      x = blk(x)
      cls_tok = x[:, 0, :]
      layer_cls.append(cls_tok.cpu())
    stacked = torch.stack(layer_cls, dim=1).numpy()
    all_layer_cls.append(stacked)
  return np.concatenate(all_layer_cls, axis=0)


def extract_and_save_dino(subjects: list[str] | None = None, device: str = 'cuda', batch_size: int = 64) -> None:
  """Extract multi-layer DINO features for each subject."""
  subjects = subjects or NSD_SUBJECTS
  (model, transform) = build_dino_model(device=device)
  for subj in subjects:
    out_path = get_paths().dino / f'{subj}_dino_layers.npy'
    if out_path.exists():
      print(f'[DINO] {subj}: already exists → skipping')
      continue
    paths = list_training_images(subj)
    print(f'[DINO] {subj}: extracting {DINO_NUM_LAYERS} layers from {len(paths)} images …')
    layer_feats = extract_dino_all_layers(paths, model, transform, batch_size=batch_size, device=device)
    np.save(out_path, layer_feats)
    print(f'  → {out_path}  shape={layer_feats.shape}')
  print('[DINO] Done.')
if __name__ == '__main__':
  extract_and_save_dino()
