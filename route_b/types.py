"""Shared record and tensor types; opaque library objects stay at I/O boundaries."""

from __future__ import annotations

from typing import Any, TypeAlias, TypedDict

import torch
from numpy.typing import NDArray


Array: TypeAlias = NDArray[Any]  # NumPy arrays have several persisted dtypes in original artifacts.
TensorMap: TypeAlias = dict[str, torch.Tensor]
Sample: TypeAlias = dict[str, torch.Tensor | int]
Scalar: TypeAlias = float | int | bool | str | None
JSON: TypeAlias = Scalar | list['JSON'] | dict[str, 'JSON']
Metrics: TypeAlias = dict[str, JSON]
Checkpoint: TypeAlias = dict[str, Any]  # torch.load includes optimizer, NumPy, and Python RNG objects.


class Record(TypedDict):
  train_idx: int
  nsd_id: int
  task: str
  score: float
  image: str
  split: str


class FoldResult(TypedDict):
  best_val_loss: float
  best_epoch: int
  metrics: dict[str, float]
  state_dict: TensorMap
