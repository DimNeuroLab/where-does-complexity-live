"""Extract CLIP ViT-L/14 image and text embeddings for NSD images."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from route_b.config import paths as get_paths
from route_b.constants import CLIP_MODEL_NAME, CLIP_PRETRAINED, COCO_SEARCH18_CATEGORIES, NSD_SUBJECTS
from route_b.data.nsd import list_training_images
from route_b.features.protocols import ClipModel

CLIP_REPOSITORY = 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
CLIP_REVISION = '1627032197142fbe2a7cfec626f4ced3ae60d07a'


class ImagePathDataset(Dataset):

  def __init__(self, paths: list[Path], preprocess: Callable[[Image.Image], torch.Tensor]) -> None:
    self.paths = paths
    self.preprocess = preprocess

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
    try:
      img = Image.open(self.paths[idx]).convert('RGB')
    except Exception:
      img = Image.new('RGB', (224, 224), (128, 128, 128))
    return (self.preprocess(img), idx)


def build_clip_model(
  model_name: str = CLIP_MODEL_NAME,
  pretrained: str = CLIP_PRETRAINED,
  device: str = 'cuda'
) -> tuple[ClipModel, Callable[[Image.Image], torch.Tensor], Callable[[list[str]], torch.Tensor]]:
  if model_name == CLIP_MODEL_NAME and pretrained == CLIP_PRETRAINED:
    weights = hf_hub_download(CLIP_REPOSITORY, 'open_clip_pytorch_model.bin', revision=CLIP_REVISION)
    settings = open_clip.get_pretrained_cfg(model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=weights,
      image_mean=settings['mean'], image_std=settings['std'],
      image_interpolation=settings['interpolation'], image_resize_mode=settings['resize_mode'])
  else:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
  model = model.to(device).eval()
  tokenizer = open_clip.get_tokenizer(model_name)
  return (model, preprocess, tokenizer)
PROMPT_TEMPLATES = [
  'a photo of a {}.',
  'a photograph of a {}.',
  'the target object is a {}.',
  'a {} in a scene.',
  'a photo containing a {}.'
]


@torch.no_grad()
def extract_clip_image_features(
  image_paths: list[Path],
  model: ClipModel,
  preprocess: Callable[[Image.Image], torch.Tensor],
  batch_size: int = 128,
  device: str = 'cuda',
  num_workers: int = 8
) -> np.ndarray:
  """Return (N, embed_dim) L2-normalised CLIP image embeddings."""
  dataset = ImagePathDataset(image_paths, preprocess)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  all_feats = []
  for (imgs, _) in tqdm(loader, desc='CLIP-img'):
    imgs = imgs.to(device, non_blocking=True)
    feats = model.encode_image(imgs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    all_feats.append(feats.cpu().numpy())
  return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def extract_clip_text_features(
  categories: list[str],
  model: ClipModel,
  tokenizer: Callable[[list[str]], torch.Tensor],
  device: str = 'cuda'
) -> dict[str, np.ndarray]:
  """Prompt-ensembled CLIP text embeddings. Returns {cat: (D,)} L2-normed."""
  cat_embeddings = {}
  for cat in categories:
    prompts = [t.format(cat) for t in PROMPT_TEMPLATES]
    tokens = tokenizer(prompts).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    ensemble = feats.mean(dim=0)
    ensemble = ensemble / ensemble.norm()
    cat_embeddings[cat] = ensemble.cpu().numpy()
  return cat_embeddings


def extract_and_save_clip(subjects: list[str] | None = None, device: str = 'cuda', batch_size: int = 128) -> None:
  subjects = subjects or NSD_SUBJECTS
  (model, preprocess, tokenizer) = build_clip_model(device=device)
  txt_out = get_paths().clip_text / 'clip_text_embeddings.npz'
  if txt_out.exists():
    print('[CLIP-text] Already exists → skipping')
  else:
    print('[CLIP-text] Extracting prompt-ensembled text embeddings …')
    text_feats = extract_clip_text_features(COCO_SEARCH18_CATEGORIES, model, tokenizer, device=device)
    np.savez(txt_out, **text_feats)
    print(f'  → saved {txt_out}  ({len(text_feats)} categories)')
  for subj in subjects:
    out_path = get_paths().clip_image / f'{subj}_clip_img.npy'
    if out_path.exists():
      print(f'[CLIP-img] {subj}: already exists → skipping')
      continue
    paths = list_training_images(subj)
    print(f'[CLIP-img] {subj}: extracting from {len(paths)} images …')
    feats = extract_clip_image_features(paths, model, preprocess, batch_size=batch_size, device=device)
    np.save(out_path, feats)
    print(f'  → saved {out_path}  shape={feats.shape}')
  print('[CLIP] Done.')
if __name__ == '__main__':
  extract_and_save_clip()
