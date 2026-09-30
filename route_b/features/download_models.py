"""Download the pinned upstream weights into the user's selected model caches."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from route_b.features.clip import CLIP_REPOSITORY, CLIP_REVISION
from route_b.runtime import atomic_json, digest


def main() -> None:
  directory = Path(torch.hub.get_dir()) / 'checkpoints'
  directory.mkdir(parents=True, exist_ok=True)
  destination = directory / 'dinov2_vitl14_pretrain.pth'
  url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
  if not destination.exists():
    temporary = destination.with_suffix('.download')
    torch.hub.download_url_to_file(url, str(temporary))
    temporary.replace(destination)
  clip = Path(hf_hub_download(CLIP_REPOSITORY, 'open_clip_pytorch_model.bin', revision=CLIP_REVISION))
  atomic_json(directory / 'route_b_models.json', {
    'dino': {'url': url, 'sha256': digest(destination)},
    'clip': {'repository': CLIP_REPOSITORY, 'revision': CLIP_REVISION, 'sha256': digest(clip)},
  })
  print('Pinned Route B weights are available; hashes saved in', directory / 'route_b_models.json')


if __name__ == '__main__':
  main()
