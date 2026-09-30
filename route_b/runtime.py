"""Atomic artifacts, persistent compute accounting, and epoch recovery."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from route_b.types import Checkpoint, JSON


def digest(path: Path) -> str:
  """Hash a file without loading its full contents into memory."""
  result = hashlib.sha256()
  with path.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
      result.update(block)
  return result.hexdigest()


def json_default(value: Any) -> JSON:
  """Convert NumPy scalars at the JSON serialization boundary."""
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: object) -> None:
  """Publish a JSON artifact only after flushing the complete replacement."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  with temporary.open('w') as stream:
    json.dump(value, stream, indent=2, default=json_default, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: Checkpoint) -> None:
  """Keep the previous complete checkpoint until its replacement is flushed."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  with temporary.open('wb') as stream:
    torch.save(value, stream)
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)


class BudgetExhausted(RuntimeError):
  """The experiment must pause until its authorized compute budget is reviewed."""


class Budget:
  """Prepay short GPU intervals so interruption cannot reset the allowance.

  :param devices: Number of devices reserved for this process.
  """

  def __init__(self, path: Path, hours: float, devices: int = 1) -> None:
    self.path, self.limit, self.devices = path, hours * 3600, devices
    self.charged, self.deadline = 0.0, None
    if path.exists():
      state = json.loads(path.read_text())
      if state['limit_seconds'] != self.limit:
        raise ValueError('The run compute cap cannot change during resume')
      self.charged = state['charged_seconds']
    self.write()

  def write(self) -> None:
    atomic_json(self.path, {'limit_seconds': self.limit, 'charged_seconds': self.charged,
                           'updated': time.time(), 'reserved_devices': self.devices})

  def check(self) -> None:
    """Check between GPU operations, allowing an in-flight operation to finish."""
    if self.deadline is not None and self.deadline - time.monotonic() >= 5:
      return
    self.close()
    available = (self.limit - self.charged) / self.devices
    if available < 5:
      raise BudgetExhausted('The total reproduction GPU budget is exhausted')
    credit = min(60, available)
    self.charged += credit * self.devices
    self.write()
    self.deadline = time.monotonic() + credit

  def close(self) -> None:
    """Refund unused credit on clean exit; crashes retain their last prepaid slice."""
    if self.deadline is not None:
      self.charged -= (self.deadline - time.monotonic()) * self.devices
      self.deadline = None
      self.write()


active_budget: Budget | None = None


def check_budget() -> None:
  """Account for compute when invoked by a budgeted worker."""
  if active_budget is not None:
    active_budget.check()


def capture_rng() -> Checkpoint:
  """Capture original global RNGs without changing loader sampling behavior."""
  return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
          'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: Checkpoint) -> None:
  """Restore RNGs after model and optimizer reconstruction."""
  random.setstate(state['python'])
  np.random.set_state(state['numpy'])
  torch.set_rng_state(state['torch'].cpu())
  if state['cuda']:
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class EpochRecovery:
  """Add epoch checkpoints without changing original optimization or selection."""

  def __init__(self, directory: Path, metadata: Checkpoint) -> None:
    self.directory, self.metadata = directory, metadata
    self.path = directory / 'last.pt'

  def restore(
    self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
  ) -> Checkpoint | None:
    if not self.path.exists():
      return None
    state = torch.load(self.path, map_location='cpu', weights_only=False)
    if state['metadata'] != self.metadata:
      raise ValueError('Epoch recovery settings changed')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    print(f'Resume {self.directory}: epoch {state["epoch"] + 1}', flush=True)
    return state

  def save(
    self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, progress: Checkpoint,
  ) -> None:
    check_budget()
    atomic_checkpoint(self.path, {'epoch': epoch, 'model': model.state_dict(),
      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': capture_rng(),
      'metadata': self.metadata, 'progress': progress})
    summary = {key: value for key, value in progress.items() if key != 'best_state'}
    atomic_json(self.directory / 'progress.json', {'epoch': epoch, **summary})
    print(f'{self.directory.name}: saved epoch {epoch}', flush=True)
