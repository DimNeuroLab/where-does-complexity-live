"""Task-conditioned image features used by the engineered Route A model."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image

from route_a.engineered.schema import SUPPORTED_TARGETS, validate_pairs


DETECTOR_FEATURE_NAMES: tuple[str, ...] = (
  'target_present_lo', 'target_present_hi', 'target_count_lo', 'target_count_hi',
  'target_max_score', 'target_sum_scores', 'target_best_area_frac',
  'target_best_x1', 'target_best_y1', 'target_best_x2', 'target_best_y2',
  'target_best_w', 'target_best_h', 'target_best_cx', 'target_best_cy',
  'target_best_aspect', 'n_det_lo', 'n_det_hi', 'sum_scores_all',
  'max_score_all', 'n_unique_classes_lo', 'n_non_target_lo',
  'n_non_target_hi', 'max_non_target_score', 'sum_scores_non_target',
  'top1_is_target',
)

# IDs in the COCO category list attached to the pinned Faster R-CNN V2 COCO_V1 weights.
# Keeping these here lets a fully cached prediction avoid importing torchvision.
_COCO_V1_TARGET_IDS: dict[str, int] = {
  'bottle': 44, 'car': 3, 'chair': 62, 'clock': 85,
  'cup': 47, 'fork': 48, 'keyboard': 76, 'laptop': 73,
  'microwave': 78, 'mouse': 74, 'oven': 79, 'potted plant': 64,
  'sink': 81, 'stop sign': 13, 'toilet': 70, 'tv': 72,
}


def _feature_prefix(value: str) -> str:
  return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


@dataclass(frozen=True)
class FeatureConfig:
  """Settings that determine the exact raw feature schema and model weights."""

  device: str = 'auto'
  dino_model_name: str = 'vit_small_patch14_dinov2.lvd142m'
  dino_dim: int = 384
  openclip_model_name: str = 'ViT-L-14'
  openclip_pretrained: str = 'laion2b_s32b_b82k'
  openclip_dim: int = 768
  openclip_normalize: bool = True
  detector_weights: Literal['COCO_V1'] = 'COCO_V1'
  det_topk: int = 50
  det_thr_lo: float = 0.5
  det_thr_hi: float = 0.7

  def __post_init__(self) -> None:
    if not self.device or not (self.device == 'auto' or self.device == 'cpu' or self.device.startswith('cuda')):
      raise ValueError("device must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'.")
    if not self.dino_model_name or not self.openclip_model_name or not self.openclip_pretrained:
      raise ValueError('Embedding model and pretrained weight identifiers must be nonempty.')
    if self.dino_dim <= 0 or self.openclip_dim <= 0:
      raise ValueError('Embedding dimensions must be positive.')
    if self.detector_weights != 'COCO_V1':
      raise ValueError("detector_weights must be the explicit Faster R-CNN V2 enum member 'COCO_V1'.")
    if self.det_topk <= 0:
      raise ValueError('det_topk must be positive.')
    if not (0.0 <= self.det_thr_lo <= self.det_thr_hi <= 1.0):
      raise ValueError('Detection thresholds must satisfy 0 <= det_thr_lo <= det_thr_hi <= 1.')


def expected_feature_columns(config: FeatureConfig) -> list[str]:
  """Return the exact raw feature names produced by an extractor config.

  :param config: Feature model settings, including embedding dimensions.
  :returns: Task, DINO, OpenCLIP, and detector names in training order.
  """
  task = [f'task_{target.replace(" ", "_")}' for target in SUPPORTED_TARGETS]
  dino = [f'dino_{index:03d}' for index in range(config.dino_dim)]
  clip_prefix = _feature_prefix(config.openclip_model_name)
  clip = [f'openclip_{clip_prefix}_{index:03d}' for index in range(config.openclip_dim)]
  detector = [f'det_{name}' for name in DETECTOR_FEATURE_NAMES]
  return task + dino + clip + detector


@dataclass(frozen=True)
class _ImageFeatures:
  dino: np.ndarray
  openclip: np.ndarray
  scores: np.ndarray
  labels: np.ndarray
  boxes: np.ndarray
  width: float
  height: float


def feature_columns(frame: pd.DataFrame) -> list[str]:
  """Return feature columns in the source model's fixed group order.

  :raises ValueError: If any required feature group is missing or incomplete.
  """
  columns = list(frame.columns)
  if len(columns) != len(set(columns)):
    raise ValueError('Feature table has duplicate column names.')
  task = [f'task_{target.replace(" ", "_")}' for target in SUPPORTED_TARGETS]
  detector = [f'det_{name}' for name in DETECTOR_FEATURE_NAMES]
  missing = [name for name in task + detector if name not in frame.columns]
  if missing:
    raise ValueError(f'Feature table is missing required columns: {", ".join(missing)}.')

  dino = [name for name in columns if name.startswith('dino_')]
  clip = [name for name in columns if name.startswith('openclip_')]
  for group_name, group in (('DINO', dino), ('OpenCLIP', clip)):
    if not group:
      raise ValueError(f'Feature table has no {group_name} columns.')
    matches = [re.fullmatch(r'(.+)_(\d+)', name) for name in group]
    if any(match is None for match in matches):
      raise ValueError(f'{group_name} columns must end in numeric indices.')
    prefixes = {match.group(1) for match in matches if match is not None}
    if len(prefixes) != 1:
      raise ValueError(f'{group_name} columns have inconsistent prefixes.')
    indices = sorted(int(match.group(2)) for match in matches if match is not None)
    if indices != list(range(len(group))):
      raise ValueError(f'{group_name} feature indices must be contiguous from zero.')
  dino.sort(key=lambda name: int(name.rsplit('_', 1)[1]))
  clip.sort(key=lambda name: int(name.rsplit('_', 1)[1]))
  return task + dino + clip + detector


class ImageTargetFeatureExtractor:
  """Extract raw image, target, and detector features with optional disk caching.

  The heavy model imports and weight loads occur on the first uncached image.
  A cache entry contains image-only embeddings and detector output. Its key
  includes the image path, size, modification time, and all feature settings.
  """

  def __init__(self, config: FeatureConfig | None = None, cache_dir: Path | str | None = None) -> None:
    self.config = config or FeatureConfig()
    self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    self._image_cache: dict[str, _ImageFeatures] = {}
    self._device_name: str | None = None
    self._dino_model: Any = None
    self._dino_preprocess: Any = None
    self._openclip_model: Any = None
    self._openclip_preprocess: Any = None
    self._detector_model: Any = None
    self._category_ids: dict[str, int] | None = None

  def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
    """Append ordered numeric features to validated image-target pairs.

    :param pairs: Rows with ``image_path`` and ``target``; metadata is retained.
    :returns: Canonical metadata followed by ``task_*``, ``dino_*``,
      ``openclip_*``, and ``det_*`` columns.
    """
    canonical = validate_pairs(pairs)
    names = self._feature_names()
    collisions = sorted(set(names).intersection(canonical.columns))
    if collisions:
      raise ValueError(f'Input table already has feature columns: {", ".join(collisions)}.')

    unique_paths = canonical['image_path'].drop_duplicates().tolist()
    image_features = {path: self._load_or_compute(path) for path in unique_paths}
    target_indices = {target: index for index, target in enumerate(SUPPORTED_TARGETS)}
    rows: list[np.ndarray] = []
    for image_path, target in zip(canonical['image_path'], canonical['target']):
      stored = image_features[image_path]
      task = np.zeros(len(SUPPORTED_TARGETS), dtype=np.float32)
      task[target_indices[target]] = 1.0
      target_id = self._detector_category_ids()[target]
      detector = self._detector_row_features(stored, target_id)
      rows.append(np.concatenate((task, stored.dino, stored.openclip, detector)).astype(np.float32))
    features = pd.DataFrame(np.stack(rows), columns=names)
    return pd.concat((canonical.reset_index(drop=True), features), axis=1)

  def _feature_names(self) -> list[str]:
    return expected_feature_columns(self.config)

  def _cache_key(self, image_path: str) -> str:
    path = Path(image_path)
    try:
      stat = path.stat()
    except OSError as exc:
      raise FileNotFoundError(f'Cannot stat image file {path}: {exc}') from exc
    payload = {
      'version': 1,
      'path': str(path),
      'size': stat.st_size,
      'mtime_ns': stat.st_mtime_ns,
      'config': asdict(self.config),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

  def _load_or_compute(self, image_path: str) -> _ImageFeatures:
    key = self._cache_key(image_path)
    if key in self._image_cache:
      return self._image_cache[key]
    cache_file = self.cache_dir / key[:2] / f'{key}.npz' if self.cache_dir is not None else None
    if cache_file is not None and cache_file.is_file():
      try:
        with np.load(cache_file, allow_pickle=False) as content:
          stored = _ImageFeatures(
            dino=content['dino'],
            openclip=content['openclip'],
            scores=content['scores'],
            labels=content['labels'],
            boxes=content['boxes'],
            width=float(content['width_height'][0]),
            height=float(content['width_height'][1]),
          )
        self._validate_image_features(stored)
      except Exception as exc:
        raise ValueError(f'Invalid feature cache {cache_file}; remove it and retry: {exc}') from exc
    else:
      stored = self._compute_image_features(image_path)
      self._validate_image_features(stored)
      if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
          with tempfile.NamedTemporaryFile(dir=cache_file.parent, suffix='.npz', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
              temporary,
              dino=stored.dino,
              openclip=stored.openclip,
              scores=stored.scores,
              labels=stored.labels,
              boxes=stored.boxes,
              width_height=np.asarray((stored.width, stored.height), dtype=np.float32),
            )
          os.replace(temporary_path, cache_file)
        finally:
          if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    self._image_cache[key] = stored
    return stored

  def _validate_image_features(self, value: _ImageFeatures) -> None:
    if value.dino.shape != (self.config.dino_dim,) or value.openclip.shape != (self.config.openclip_dim,):
      raise ValueError('Embedding dimensions do not match FeatureConfig.')
    if value.scores.ndim != 1 or value.labels.shape != value.scores.shape:
      raise ValueError('Detector scores and labels have inconsistent shapes.')
    if value.boxes.shape != (len(value.scores), 4):
      raise ValueError('Detector boxes have an inconsistent shape.')
    if len(value.scores) > self.config.det_topk:
      raise ValueError('Detector cache has more boxes than det_topk.')
    if not np.issubdtype(value.labels.dtype, np.integer):
      raise ValueError('Detector labels must be integer category IDs.')
    arrays = (value.dino, value.openclip, value.scores, value.boxes)
    if not all(np.isfinite(array).all() for array in arrays):
      raise ValueError('Image features contain non-finite values.')
    if not (np.isfinite(value.width) and np.isfinite(value.height) and value.width > 0 and value.height > 0):
      raise ValueError('Image dimensions must be positive and finite.')

  def _resolve_device(self) -> str:
    if self._device_name is not None:
      return self._device_name
    try:
      import torch
    except ImportError as exc:
      raise RuntimeError('PyTorch is required to extract uncached image features.') from exc
    requested = self.config.device
    if requested == 'auto':
      requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda'):
      if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device {requested!r} was requested but CUDA is unavailable.')
      try:
        selected = torch.device(requested)
      except RuntimeError as exc:
        raise ValueError(f'Invalid CUDA device {requested!r}.') from exc
      if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise RuntimeError(f'CUDA device {requested!r} is unavailable.')
    self._device_name = requested
    return requested

  def _load_dino(self) -> None:
    if self._dino_model is not None:
      return
    try:
      import timm
    except ImportError as exc:
      raise RuntimeError('timm is required to extract DINOv2 features.') from exc
    try:
      model = timm.create_model(self.config.dino_model_name, pretrained=True, num_classes=0)
      model = model.to(self._resolve_device()).eval()
      model.requires_grad_(False)
      data_config = timm.data.resolve_model_data_config(model)
      preprocess = timm.data.create_transform(**data_config, is_training=False)
    except Exception as exc:
      raise RuntimeError(f'Could not load DINOv2 model {self.config.dino_model_name!r}: {exc}') from exc
    self._dino_model = model
    self._dino_preprocess = preprocess

  def _load_openclip(self) -> None:
    if self._openclip_model is not None:
      return
    try:
      import open_clip
    except ImportError as exc:
      raise RuntimeError('open_clip_torch is required to extract OpenCLIP features.') from exc
    try:
      model, _, preprocess = open_clip.create_model_and_transforms(
        self.config.openclip_model_name,
        pretrained=self.config.openclip_pretrained,
        device=self._resolve_device(),
      )
      model.eval().requires_grad_(False)
    except Exception as exc:
      raise RuntimeError(
        f'Could not load OpenCLIP {self.config.openclip_model_name!r} '
        f'with weights {self.config.openclip_pretrained!r}: {exc}'
      ) from exc
    self._openclip_model = model
    self._openclip_preprocess = preprocess

  def _load_detector(self) -> None:
    if self._detector_model is not None:
      return
    try:
      from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    except ImportError as exc:
      raise RuntimeError('torchvision detection models are required to extract detector features.') from exc
    try:
      from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
      weights = FasterRCNN_ResNet50_FPN_V2_Weights[self.config.detector_weights]
      categories = weights.meta['categories']
      if any(categories[index] != target for target, index in _COCO_V1_TARGET_IDS.items()):
        raise ValueError('Installed detector category IDs differ from the supported COCO_V1 mapping.')
      model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self._resolve_device()).eval()
      model.requires_grad_(False)
    except Exception as exc:
      raise RuntimeError(f'Could not load Faster R-CNN weights {self.config.detector_weights!r}: {exc}') from exc
    self._detector_model = model

  def _detector_category_ids(self) -> dict[str, int]:
    if self._category_ids is None:
      self._category_ids = _COCO_V1_TARGET_IDS.copy()
    assert self._category_ids is not None
    return self._category_ids

  def _compute_image_features(self, image_path: str) -> _ImageFeatures:
    try:
      import torch
      from torchvision.transforms.functional import to_tensor
    except ImportError as exc:
      raise RuntimeError('PyTorch and torchvision are required to extract image features.') from exc
    try:
      with Image.open(image_path) as image:
        rgb = image.convert('RGB')
        rgb.load()
    except Exception as exc:
      raise RuntimeError(f'Could not decode image {image_path}: {exc}') from exc
    width, height = rgb.size

    self._load_dino()
    self._load_openclip()
    self._load_detector()
    try:
      with torch.no_grad():
        dino_output = self._dino_model(self._dino_preprocess(rgb).unsqueeze(0).to(self._resolve_device()))
        if isinstance(dino_output, (tuple, list)):
          dino_output = dino_output[0]
        dino = dino_output.detach().cpu().numpy().astype(np.float32).reshape(-1)

        clip_output = self._openclip_model.encode_image(
          self._openclip_preprocess(rgb).unsqueeze(0).to(self._resolve_device())
        )
        if self.config.openclip_normalize:
          clip_output = clip_output / clip_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        openclip = clip_output.detach().cpu().numpy().astype(np.float32).reshape(-1)

        detections = self._detector_model([to_tensor(rgb).to(self._resolve_device())])[0]
        scores = detections['scores'].detach().cpu().numpy().astype(np.float32)[:self.config.det_topk]
        labels = detections['labels'].detach().cpu().numpy().astype(np.int64)[:self.config.det_topk]
        boxes = detections['boxes'].detach().cpu().numpy().astype(np.float32)[:self.config.det_topk]
    except Exception as exc:
      raise RuntimeError(f'Feature extraction failed for image {image_path}: {exc}') from exc
    return _ImageFeatures(dino, openclip, scores, labels, boxes, float(width), float(height))

  def _detector_row_features(self, image: _ImageFeatures, target_id: int) -> np.ndarray:
    scores, labels, boxes = image.scores, image.labels, image.boxes
    threshold_lo, threshold_hi = self.config.det_thr_lo, self.config.det_thr_hi
    above_lo, above_hi = scores >= threshold_lo, scores >= threshold_hi
    is_target = labels == target_id
    target_scores = scores[is_target]
    non_target_scores = scores[~is_target]
    target_lo = float(np.sum(target_scores >= threshold_lo))
    target_hi = float(np.sum(target_scores >= threshold_hi))

    geometry = [0.0] * 10
    if len(target_scores):
      target_indices = np.flatnonzero(is_target)
      best_index = int(target_indices[np.argmax(scores[target_indices])])
      x1, y1, x2, y2 = (float(value) for value in boxes[best_index])
      width = max(0.0, x2 - x1)
      height = max(0.0, y2 - y1)
      x1n, y1n = x1 / image.width, y1 / image.height
      x2n, y2n = x2 / image.width, y2 / image.height
      widthn, heightn = width / image.width, height / image.height
      geometry = [
        width * height / (image.width * image.height + 1e-12),
        x1n, y1n, x2n, y2n, widthn, heightn,
        (x1n + x2n) / 2.0, (y1n + y2n) / 2.0, widthn / (heightn + 1e-12),
      ]

    return np.asarray([
      float(target_lo > 0),
      float(target_hi > 0),
      target_lo,
      target_hi,
      float(np.max(target_scores)) if len(target_scores) else 0.0,
      float(np.sum(target_scores[target_scores >= threshold_lo])),
      *geometry,
      float(np.sum(above_lo)),
      float(np.sum(above_hi)),
      float(np.sum(scores[above_lo])),
      float(np.max(scores)) if len(scores) else 0.0,
      float(len(set(labels[above_lo].tolist()))),
      float(np.sum(non_target_scores >= threshold_lo)),
      float(np.sum(non_target_scores >= threshold_hi)),
      float(np.max(non_target_scores)) if len(non_target_scores) else 0.0,
      float(np.sum(non_target_scores[non_target_scores >= threshold_lo])),
      float(labels[int(np.argmax(scores))] == target_id) if len(scores) else 0.0,
    ], dtype=np.float32)
