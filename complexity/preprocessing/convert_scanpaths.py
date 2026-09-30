#!/usr/bin/env python3
"""
COCO-Search18 JSON -> CSV (model-ready)

Creates a trial-level CSV with REQUIRED columns:
 RT, subject, image, trial, N, T2T, TTFix2R

Input JSON: an array of objects with keys:
 name, subject, task, condition, bbox, X, Y, T, length, correct, RT, split, fixOnTarget

Definitions:
 - image   := obj["name"]
 - subject := obj["subject"]
 - trial   := per-subject trial index (1..), assigned in JSON order
 - RT      := obj["RT"]
 - N       := number of eye movements (default: saccades = fixations - 1; configurable)
 - T2T     := time-to-target fixation onset (sum of fixation durations before first fixation on target bbox)
 - TTFix2R := RT - T2T  (time from first target fixation onset to response)

Notes:
 - Fixation lists include initial center fixation.
 - If bbox is missing/invalid OR no fixation lands in bbox, then T2T and TTFix2R are empty (NaN).
 - For target-absent trials, bbox may be missing; T2T/TTFix2R will be NaN.
 - N can be "saccades" (fixations-1) or "fixations" (fixations).
 - You can optionally filter to TP-only, correct-only, fixOnTarget-only, split=train/valid.

Direct conversion::

  python -m complexity.preprocessing.convert_scanpaths --scanpaths-file scanpaths.json --output-file trials.csv

Use ``--only-condition present`` and optional ``--only-correct 1`` to select trials.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO


def open_maybe_gzip(path: Path, mode: str = 'rt', encoding: str = 'utf-8') -> TextIO:
  if str(path).endswith('.gz'):
    return gzip.open(path, mode=mode, encoding=encoding)
  return open(path, mode=mode, encoding=encoding)


def to_float(x: object) -> float | None:
  try:
    if x is None:
      return None
    v = float(x)
    if math.isnan(v) or math.isinf(v):
      return None
    return v
  except Exception:
    return None


def to_int(x: object) -> int | None:
  try:
    if x is None:
      return None
    return int(x)
  except Exception:
    return None


def bbox_unpack(bbox: object) -> tuple[float | None, float | None, float | None, float | None]:
  if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
    bx, by, bw, bh = (to_float(bbox[0]), to_float(bbox[1]), to_float(bbox[2]), to_float(bbox[3]))
    return bx, by, bw, bh
  return (None, None, None, None)


def inside_bbox(x: float, y: float, bx: float, by: float, bw: float, bh: float) -> bool:
  return (bx <= x <= bx + bw) and (by <= y <= by + bh)


def compute_T2T(
  X: list[object],
  Y: list[object],
  T: list[object],
  bbox: tuple[float | None, float | None, float | None, float | None],
) -> tuple[int | None, float | None]:
  """
  Returns (first_hit_fix_index, T2T_onset).
  T2T_onset = sum of fixation durations before the first fixation that lands inside bbox.
  If bbox invalid or no hit, returns (None, None).
  """
  bx, by, bw, bh = bbox
  if bx is None or by is None or bw is None or bh is None:
    return (None, None)

  n = min(len(X), len(Y), len(T))
  if n == 0:
    return (None, None)

  cum = 0.0
  for k in range(n):
    xk = to_float(X[k])
    yk = to_float(Y[k])
    tk = to_float(T[k])
    if tk is None:
      tk = 0.0

    if xk is not None and yk is not None:
      if inside_bbox(xk, yk, bx, by, bw, bh):
        return (k, cum)

    cum += tk

  return (None, None)


def load_json_array(path: Path) -> list[dict[str, Any]]:
  with open_maybe_gzip(path, 'rt') as f:
    data = json.load(f)
  if not isinstance(data, list):
    raise ValueError('Expected a top-level JSON array (list of objects).')
  return data


def json_to_csv(
  in_json: Path,
  out_csv: Path,
  movement: str = 'saccades',            # "saccades" or "fixations"
  only_split: str | None = None,      # "train" or "valid"
  only_condition: str | None = None,  # e.g., "TP"/"TA" or "present"/"absent"
  only_correct: int | None = None,    # 1 or 0
  only_fixOnTarget: int | None = None # 1 or 0
) -> None:
  data = load_json_array(in_json)

  # Per-subject trial counter (JSON order)
  trial_counter = defaultdict(int)

  # REQUIRED columns first; then a few helpful extras
  fieldnames = [
    'RT', 'subject', 'image', 'trial', 'N', 'T2T', 'TTFix2R',
    # extras (helpful for filtering/debugging; safe to ignore later)
    'task', 'condition', 'split', 'correct', 'fixOnTarget',
    'n_fixations', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'first_target_fix_idx'
  ]

  out_csv.parent.mkdir(parents=True, exist_ok=True)
  with open(out_csv, 'w', newline="", encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

    for obj in data:
      image = obj.get('name', "")
      subject = obj.get('subject', "")
      if not subject:
        subject = obj.get('sample_id', "")
      task = obj.get('task', "")
      condition = obj.get('condition', "")
      split = obj.get('split', "")
      correct = to_int(obj.get('correct', None))
      fixOnTarget_flag = obj.get('fixOnTarget', None)
      fixOnTarget_flag = None if fixOnTarget_flag is None else bool(fixOnTarget_flag)
      RT = to_float(obj.get('RT', None))

      # Filters
      if only_split is not None and str(split) != str(only_split):
        continue
      if only_condition is not None and str(condition) != str(only_condition):
        continue
      if only_correct is not None:
        if correct is None or correct != int(only_correct):
          continue
      if only_fixOnTarget is not None:
        want = bool(int(only_fixOnTarget))
        if fixOnTarget_flag is None or fixOnTarget_flag != want:
          continue

      X = obj.get('X', []) or []
      Y = obj.get('Y', []) or []
      T = obj.get('T', []) or []

      # Fixation count
      n_fix = to_int(obj.get('length', None))

      # Use T to limit n_fix only if T is present
      limit_len = min(len(X), len(Y))
      if T:
        limit_len = min(limit_len, len(T))

      if n_fix is None:
        n_fix = limit_len
      else:
        n_fix = max(0, min(n_fix, limit_len))

      # Eye-movement count N
      if movement == 'fixations':
        N = n_fix
      else:  # "saccades"
        N = max(n_fix - 1, 0)

      # Bbox and target timing
      bbox = bbox_unpack(obj.get('bbox', None))
      bx, by, bw, bh = bbox
      first_idx, T2T = compute_T2T(X[:n_fix], Y[:n_fix], T[:n_fix], bbox)

      # TTFix2R defined only if RT and T2T exist
      TTFix2R = None
      if RT is not None and T2T is not None:
        TTFix2R = RT - T2T

      # Trial index per subject
      trial_counter[str(subject)] += 1
      trial = trial_counter[str(subject)]

      row = {
        # REQUIRED
        'RT': RT,
        'subject': subject,
        'image': image,
        'trial': trial,
        'N': N,
        'T2T': T2T,
        'TTFix2R': TTFix2R,
        # extras
        'task': task,
        'condition': condition,
        'split': split,
        'correct': correct,
        'fixOnTarget': fixOnTarget_flag,
        'n_fixations': n_fix,
        'bbox_x': bx,
        'bbox_y': by,
        'bbox_w': bw,
        'bbox_h': bh,
        'first_target_fix_idx': first_idx,
      }

      w.writerow(row)


def main() -> None:
  ap = argparse.ArgumentParser(allow_abbrev=False, description='Convert COCO-Search18 JSON array to model-ready CSV.')
  ap.add_argument(
    '--scanpaths-file',
    dest='in_json',
    required=True,
    type=Path,
    help='Path to COCO-Search18 JSON (or .json.gz)',
  )
  ap.add_argument('--output-file', dest='out_csv', required=True, type=Path, help='Output CSV path')

  ap.add_argument('--movement', choices=['saccades', 'fixations'], default='saccades',
          help='Define N as number of saccades (=fixations-1) or fixations (=fixations). Default: saccades')

  ap.add_argument('--only-split', dest='only_split', choices=['train', 'valid'], default=None,
          help='Filter rows by split')
  ap.add_argument('--only-condition', dest='only_condition', default=None,
          help='Filter rows by condition (e.g., TP/TA or present/absent). Exact string match.')
  ap.add_argument('--only-correct', dest='only_correct', choices=['0', '1'], default=None,
          help='Filter rows by correctness (1=correct only, 0=incorrect only)')
  ap.add_argument('--only-fix-on-target', dest='only_fixOnTarget', choices=['0', '1'], default=None,
          help='Filter rows by fixOnTarget flag (1=true only, 0=false only)')

  args = ap.parse_args()

  json_to_csv(
    in_json=args.in_json,
    out_csv=args.out_csv,
    movement=args.movement,
    only_split=args.only_split,
    only_condition=args.only_condition,
    only_correct=None if args.only_correct is None else int(args.only_correct),
    only_fixOnTarget=None if args.only_fixOnTarget is None else int(args.only_fixOnTarget),
  )


if __name__ == '__main__':
  main()
