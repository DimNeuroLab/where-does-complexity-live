"""DINOv2 and CLIP ViT-L/14 embedding extraction for NSD training images.

Reused by Route A (embedding-conditioned stimulus readout) and Route B
(neural readout auxiliary targets and text conditioning).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from shared.constants import CLIP_MODEL_NAME, CLIP_PRETRAINED, DINO_MODEL_NAME
from shared.nsd_utils import list_training_images


class _ImagePathDataset(Dataset):
  """Loads images from disk, substituting a gray placeholder on read errors."""

  def __init__(self, paths: Sequence[Path], transform) -> None:
    self.paths = list(paths)
    self.transform = transform

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index: int):
    try:
      image = Image.open(self.paths[index]).convert('RGB')
    except Exception:
      image = Image.new('RGB', (224, 224), (128, 128, 128))
    return self.transform(image), index


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

def build_dino_model(model_name: str = DINO_MODEL_NAME, device: str = 'cuda'):
  """Load DINOv2 ViT-L/14 and its input transform.

  Respects ``TORCH_HOME`` for the torch.hub weights cache; see
  ``docs/data_setup.md`` for the recommended setup.
  """
  model = torch.hub.load('facebookresearch/dinov2', model_name)
  model = model.to(device).eval()
  transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return model, transform


@torch.no_grad()
def extract_dino_all_layers(
  image_paths: Sequence[Path],
  model,
  transform,
  batch_size: int = 64,
  device: str = 'cuda',
  num_workers: int = 8,
) -> np.ndarray:
  """Extract the CLS token from every transformer block.

  :param image_paths: Images to extract features from, in order.
  :param model: A DINOv2 model from :func:`build_dino_model`.
  :param transform: The matching input transform.
  :returns: Array of shape ``(len(image_paths), n_layers, embed_dim)``.
  """
  dataset = _ImagePathDataset(image_paths, transform)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

  all_layer_cls: list[np.ndarray] = []
  for images, _ in tqdm(loader, desc='DINO multi-layer'):
    images = images.to(device, non_blocking=True)
    hidden = model.prepare_tokens_with_masks(images)
    layer_cls = []
    for block in model.blocks:
      hidden = block(hidden)
      layer_cls.append(hidden[:, 0, :].cpu())
    all_layer_cls.append(torch.stack(layer_cls, dim=1).numpy())

  return np.concatenate(all_layer_cls, axis=0)


def extract_and_save_dino(
  nsd_root: Path,
  out_dir: Path,
  subjects: Sequence[str],
  device: str = 'cuda',
  batch_size: int = 64,
) -> None:
  """Extract and cache multi-layer DINO features for each subject.

  Writes ``{subject}_dino_layers.npy`` (shape ``(N, 24, 1024)``) to
  ``out_dir``, skipping subjects whose output file already exists.

  :param nsd_root: Root of the NSD/Algonauts-2023 data tree.
  :param out_dir: Directory to write ``{subject}_dino_layers.npy`` files to.
  :param subjects: Subject IDs to process, for example ``['subj01', ...]``.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  model, transform = build_dino_model(device=device)

  for subject in subjects:
    out_path = out_dir / f'{subject}_dino_layers.npy'
    if out_path.exists():
      print(f'[DINO] {subject}: already exists, skipping')
      continue
    paths = list_training_images(nsd_root, subject)
    print(f'[DINO] {subject}: extracting from {len(paths)} images')
    features = extract_dino_all_layers(paths, model, transform, batch_size=batch_size, device=device)
    np.save(out_path, features)
    print(f'  -> {out_path}  shape={features.shape}')

  print('[DINO] Done.')


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: list[str] = [
  'a photo of a {}.',
  'a photograph of a {}.',
  'the target object is a {}.',
  'a {} in a scene.',
  'a photo containing a {}.',
]


def build_clip_model(model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED, device: str = 'cuda'):
  """Load OpenCLIP and its preprocessing transform and tokenizer.

  Respects ``HF_HOME`` and OpenCLIP's own cache env vars for the weights
  cache; see ``docs/data_setup.md``.
  """
  import open_clip

  model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
  model = model.to(device).eval()
  tokenizer = open_clip.get_tokenizer(model_name)
  return model, preprocess, tokenizer


@torch.no_grad()
def extract_clip_image_features(
  image_paths: Sequence[Path],
  model,
  preprocess,
  batch_size: int = 128,
  device: str = 'cuda',
  num_workers: int = 8,
) -> np.ndarray:
  """Return L2-normalised CLIP image embeddings, shape ``(N, embed_dim)``."""
  dataset = _ImagePathDataset(image_paths, preprocess)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

  all_features: list[np.ndarray] = []
  for images, _ in tqdm(loader, desc='CLIP-img'):
    images = images.to(device, non_blocking=True)
    features = model.encode_image(images)
    features = features / features.norm(dim=-1, keepdim=True)
    all_features.append(features.cpu().numpy())

  return np.concatenate(all_features, axis=0)


@torch.no_grad()
def extract_clip_text_features(
  categories: Sequence[str],
  model,
  tokenizer,
  device: str = 'cuda',
) -> dict[str, np.ndarray]:
  """Prompt-ensembled, L2-normalised CLIP text embeddings, one per category."""
  embeddings: dict[str, np.ndarray] = {}
  for category in categories:
    prompts = [template.format(category) for template in PROMPT_TEMPLATES]
    tokens = tokenizer(prompts).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    ensemble = features.mean(dim=0)
    ensemble = ensemble / ensemble.norm()
    embeddings[category] = ensemble.cpu().numpy()
  return embeddings


def extract_and_save_clip(
  nsd_root: Path,
  image_out_dir: Path,
  text_out_path: Path,
  categories: Sequence[str],
  subjects: Sequence[str],
  device: str = 'cuda',
  batch_size: int = 128,
) -> None:
  """Extract and cache CLIP image features per subject and one shared,
  prompt-ensembled CLIP text embedding per category.

  :param nsd_root: Root of the NSD/Algonauts-2023 data tree.
  :param image_out_dir: Directory to write ``{subject}_clip_img.npy`` files to.
  :param text_out_path: File to write the prompt-ensembled text embeddings to,
    for example ``.../clip_text_embeddings.npz``.
  :param categories: Target categories to build text embeddings for.
  :param subjects: Subject IDs to process.
  """
  image_out_dir.mkdir(parents=True, exist_ok=True)
  text_out_path.parent.mkdir(parents=True, exist_ok=True)
  model, preprocess, tokenizer = build_clip_model(device=device)

  if text_out_path.exists():
    print('[CLIP-text] Already exists, skipping')
  else:
    print('[CLIP-text] Extracting prompt-ensembled text embeddings')
    text_features = extract_clip_text_features(categories, model, tokenizer, device=device)
    np.savez(text_out_path, **text_features)
    print(f'  -> saved {text_out_path}  ({len(text_features)} categories)')

  for subject in subjects:
    out_path = image_out_dir / f'{subject}_clip_img.npy'
    if out_path.exists():
      print(f'[CLIP-img] {subject}: already exists, skipping')
      continue
    paths = list_training_images(nsd_root, subject)
    print(f'[CLIP-img] {subject}: extracting from {len(paths)} images')
    features = extract_clip_image_features(paths, model, preprocess, batch_size=batch_size, device=device)
    np.save(out_path, features)
    print(f'  -> saved {out_path}  shape={features.shape}')

  print('[CLIP] Done.')
