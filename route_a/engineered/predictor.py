"""Public image-and-target prediction API."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from route_a.engineered.bundle import ModelBundle
from route_a.engineered.features import ImageTargetFeatureExtractor
from route_a.engineered.schema import validate_pairs


class EngineeredPredictor:
  """Apply a complete engineered Route A bundle to new images."""

  def __init__(
    self,
    bundle: ModelBundle,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    extractor: ImageTargetFeatureExtractor | None = None,
  ) -> None:
    self.bundle = bundle
    config = bundle.feature_config if device is None else replace(bundle.feature_config, device=device)
    self.extractor = extractor or ImageTargetFeatureExtractor(config=config, cache_dir=cache_dir)

  @classmethod
  def from_bundle(
    cls,
    model_dir: str | Path,
    cache_dir: str | Path | None = None,
    device: str | None = None,
  ) -> EngineeredPredictor:
    """Load a trusted bundle and prepare an image feature extractor."""
    return cls(ModelBundle.load(model_dir), cache_dir=cache_dir, device=device)

  def predict_batch(self, pairs: pd.DataFrame, image_root: str | Path | None = None) -> pd.DataFrame:
    """Predict complexity for image-target pairs.

    :param pairs: Table with ``image_path`` and ``target`` columns.
    :param image_root: Base directory for relative image paths.
    :returns: Input identity columns and one ``prediction`` per row.
    """
    canonical = validate_pairs(pairs, image_root=image_root, check_images=True)
    extracted = self.extractor.transform(canonical)
    matrix = self.bundle.preprocessor.transform(extracted)
    prediction = self.bundle.model.predict(matrix)
    columns = [name for name in ('image_id', 'image_path', 'subject', 'target') if name in canonical]
    result = canonical.loc[:, columns].copy()
    result['prediction'] = prediction
    return result

  def predict_image(self, image_path: str | Path, target: str) -> float:
    """Predict one complexity score from an image and target name."""
    pairs = pd.DataFrame([{'image_path': str(image_path), 'target': target}])
    return float(self.predict_batch(pairs)['prediction'].iloc[0])
