"""Fold-local preprocessing for engineered Route A features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from route_a.engineered.features import feature_columns
from route_a.engineered.schema import SUPPORTED_TARGETS


@dataclass
class FeaturePreprocessor:
  """Fit the transforms needed by one XGBoost model.

  PCA and imputation are fitted only when ``fit`` is called. Cross-validation
  creates a new instance for each training fold.
  """

  pca_components: int = 64
  seed: int = 42
  columns: dict[str, list[str]] = field(default_factory=dict, init=False)
  imputers: dict[str, SimpleImputer] = field(default_factory=dict, init=False)
  pcas: dict[str, PCA] = field(default_factory=dict, init=False)
  output_columns: list[str] = field(default_factory=list, init=False)

  @staticmethod
  def _discover_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    ordered = feature_columns(frame)
    task_columns = [f'task_{target.replace(" ", "_")}' for target in SUPPORTED_TARGETS]
    groups = {
      'task': task_columns,
      'detector': [column for column in ordered if column.startswith('det_')],
      'openclip': [column for column in ordered if column.startswith('openclip_')],
      'dino': [column for column in ordered if column.startswith('dino_')],
    }
    for name, columns in groups.items():
      if not columns:
        raise ValueError(f'Missing {name} feature columns.')
      missing = set(columns).difference(frame.columns)
      if missing:
        raise ValueError(f'Missing {name} feature columns: {sorted(missing)[:5]}')
    return groups

  @staticmethod
  def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    try:
      matrix = frame.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    except (TypeError, ValueError) as error:
      raise ValueError(f'Feature columns must be numeric: {columns[:5]}') from error
    if np.isinf(matrix).any():
      raise ValueError(f'Feature columns contain infinity: {columns[:5]}')
    if not np.isfinite(matrix).any():
      raise ValueError(f'Feature group has no finite values: {columns[:5]}')
    return matrix

  def fit(self, frame: pd.DataFrame) -> FeaturePreprocessor:
    """Fit imputation and PCA on training rows.

    :param frame: Feature table for training rows only.
    :returns: This fitted preprocessor.
    """
    if len(frame) < 2:
      raise ValueError('At least two training rows are required for PCA.')
    if self.pca_components < 1:
      raise ValueError('pca_components must be positive.')
    self.columns = self._discover_columns(frame)
    self.imputers = {}
    self.pcas = {}
    for name, columns in self.columns.items():
      raw = self._matrix(frame, columns)
      imputer = SimpleImputer(strategy='median', keep_empty_features=True)
      filled = imputer.fit_transform(raw)
      self.imputers[name] = imputer
      if name in {'openclip', 'dino'}:
        n_components = min(self.pca_components, filled.shape[1], filled.shape[0] - 1)
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=self.seed)
        pca.fit(filled)
        self.pcas[name] = pca
    self.output_columns = (
      self.columns['detector']
      + self.columns['task']
      + [f'openclip_pca_{index:03d}' for index in range(self.pcas['openclip'].n_components_)]
      + [f'dino_pca_{index:03d}' for index in range(self.pcas['dino'].n_components_)]
    )
    return self

  def transform(self, frame: pd.DataFrame) -> np.ndarray:
    """Transform rows using previously fitted training transforms.

    :param frame: Rows with the same raw feature columns as the fit table.
    :returns: Numeric model matrix with columns in ``output_columns`` order.
    """
    if not self.columns or not self.imputers:
      raise ValueError('FeaturePreprocessor has not been fitted.')
    pieces: list[np.ndarray] = []
    for name in ('detector', 'task', 'openclip', 'dino'):
      missing = set(self.columns[name]).difference(frame.columns)
      if missing:
        raise ValueError(f'Missing {name} model inputs: {sorted(missing)[:5]}')
      raw = self._matrix(frame, self.columns[name])
      transformed = self.imputers[name].transform(raw)
      if name in self.pcas:
        transformed = self.pcas[name].transform(transformed)
      pieces.append(np.asarray(transformed, dtype=np.float32))
    matrix = np.concatenate(pieces, axis=1)
    if matrix.shape[1] != len(self.output_columns):
      raise RuntimeError('Preprocessor output width does not match fitted feature names.')
    return matrix

  def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
    """Fit on ``frame`` and return its model matrix."""
    return self.fit(frame).transform(frame)
