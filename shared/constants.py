"""Non-path constants shared across components.

Every filesystem location (NSD images, extracted embeddings, complexity
labels, output directories) is supplied explicitly by the caller, as a
function argument or a CLI flag, rather than hardcoded here. See each
component's README for the expected data layout.
"""

from __future__ import annotations

NSD_SUBJECTS: list[str] = [f'subj{i:02d}' for i in range(1, 9)]

#: 16 of the 18 COCO-Search18 target categories, after filtering out
#: ``bowl`` (insufficient data) and ``knife`` (inconsistent data).
COCO_SEARCH18_CATEGORIES: list[str] = [
  'bottle', 'car', 'chair', 'clock', 'cup', 'fork',
  'keyboard', 'laptop', 'microwave', 'mouse', 'oven',
  'potted plant', 'sink', 'stop sign', 'toilet', 'tv',
]

DINO_MODEL_NAME = 'dinov2_vitl14'
DINO_NUM_LAYERS = 24

CLIP_MODEL_NAME = 'ViT-L-14'
CLIP_PRETRAINED = 'laion2b_s32b_b82k'
CLIP_EMBED_DIM = 768

#: The 24 DINO transformer blocks, grouped into 3 tiers mirroring the visual
#: cortex hierarchy. Each tier's representation is the mean across its
#: 8 layers.
DINO_TIER_LAYERS: dict[str, list[int]] = {
  'early': list(range(0, 8)),
  'mid': list(range(8, 16)),
  'late': list(range(16, 24)),
}

DEFAULT_SEED = 42
