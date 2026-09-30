"""Apply the NSD crop transform, target-hit correction and retained trial filters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Metadata JSON contains heterogeneous source fields; keep this boundary local.
TARGET_SIZE = (425, 425)

def load_metadata(json_path: Path) -> tuple[dict[str, dict[str, Any]] | None, dict[str, int] | None]:
  print(f'Loading metadata from {json_path}...')
  try:
    with open(json_path, 'r') as f:
      data = json.load(f)

    cat_map = {}
    if 'categories' in data and 'instances' in data['categories']:
      for c in data['categories']['instances']:
        cat_map[c['name']] = c['id']

    img_map = {}
    if 'images' in data:
      for img in data['images']:
        fname = os.path.basename(img['path'])
        img_map[fname] = img

    print(f'Loaded metadata: {len(img_map)} images, {len(cat_map)} categories.')
    return img_map, cat_map
  except Exception as e:
    print(f'Error loading metadata: {e}')
    return None, None

def get_transformed_bbox(
  bbox: tuple[float, float, float, float] | list[float],
  meta_img: dict[str, Any],
  target_size: tuple[int, int],
) -> tuple[float, float, float, float] | list[float] | None:
  if not meta_img or 'crop_box' not in meta_img or 'original_size' not in meta_img:
    return bbox

  try:
    orig_w, orig_h = meta_img['original_size']
    crop_top_f, crop_bot_f, crop_left_f, crop_right_f = meta_img['crop_box']

    crop_top_px = crop_top_f * orig_h
    crop_left_px = crop_left_f * orig_w
    crop_w_px = orig_w * (1 - crop_left_f - crop_right_f)
    crop_h_px = orig_h * (1 - crop_top_f - crop_bot_f)

    final_w, final_h = target_size

    scale_x = final_w / crop_w_px if crop_w_px > 0 else 1
    scale_y = final_h / crop_h_px if crop_h_px > 0 else 1

    x, y, w, h = bbox

    new_x = (x - crop_left_px) * scale_x
    new_y = (y - crop_top_px) * scale_y
    new_w = w * scale_x
    new_h = h * scale_y

    # Clip to image boundaries
    x1 = max(0, new_x)
    y1 = max(0, new_y)
    x2 = min(final_w, new_x + new_w)
    y2 = min(final_h, new_y + new_h)
    clipped_w = x2 - x1
    clipped_h = y2 - y1

    if clipped_w <= 0 or clipped_h <= 0:
      return None

    return (x1, y1, clipped_w, clipped_h)

  except Exception as e:
    print(f'Error transforming bbox: {e}')
    return bbox

def is_fixation_inside(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
  bx, by, bw, bh = bbox
  return (bx <= x <= bx + bw) and (by <= y <= by + bh)

def main() -> None:
  parser = argparse.ArgumentParser(
    allow_abbrev=False,
    description='Reproduce the NSD target-box correction and trial filters.',
  )
  parser.add_argument('--trials-file', dest='csv', type=Path, required=True)
  parser.add_argument('--scanpaths-file', dest='scanpaths', type=Path, required=True)
  parser.add_argument('--metadata-file', dest='metadata', type=Path, required=True)
  parser.add_argument('--output-file', dest='output', type=Path, required=True)
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  # 1. Load Data
  print(f'Loading {args.csv}...')
  df = pd.read_csv(args.csv)

  print(f'Loading {args.scanpaths}...')
  with open(args.scanpaths, 'r') as f:
    scanpaths_list = json.load(f)

  # Index scanpaths by (image_name, subject) and (image_name, subject, condition) just in case
  # Actually subject + image should be unique for a trial in NSD?
  # Let's check keys
  path_index = {}
  for entry in scanpaths_list:
    key = (entry.get('name'), entry.get('subject'))
    path_index[key] = entry

  meta_imgs, meta_cats = load_metadata(args.metadata)
  if not meta_imgs:
    raise ValueError('Failed to load NSD metadata.')

  print('Updating correct property...')

  corrections_made_count = 0
  bbox_updates_count = 0

  new_corrects = []
  new_bbox_xs = []
  new_bbox_ys = []
  new_bbox_ws = []
  new_bbox_hs = []

  for idx, row in df.iterrows():
    img_name = row['image']
    subj = row['subject']
    task = row['task']
    condition = row['condition']

    # Default correct status from CSV (or 0 if missing)
    is_correct = row['correct']

    # Default bbox from CSV
    bx = row.get('bbox_x', float('nan'))
    by = row.get('bbox_y', float('nan'))
    bw = row.get('bbox_w', float('nan'))
    bh = row.get('bbox_h', float('nan'))

    # Try to match scanpath
    # The key in scanpaths JSON is name, subject.
    # But CSV has image names that might be the modified ones (e.g. bottle suffix).
    # We need to ensure we use the same key as stored in scanpaths JSON.
    # Assuming scanpaths JSON has the same image names as CSV.
    scanpath_key = (img_name, subj)
    scanpath = path_index.get(scanpath_key)

    # Only check correctness if target is present
    if condition == 'present':
      # 1. Resolve metadata image to get TRUE bbox
      # Try exact match first
      meta_img = meta_imgs.get(img_name)

      # Resolve if not direct match
      if not meta_img and '_' in img_name:
        base, ext = os.path.splitext(img_name)
        # Heuristic: split by _task
        # E.g. test-0004_nsd-00293_bottle -> test-0004_nsd-00293
        parts = base.rsplit('_', 1)
        if len(parts) == 2:
          core_name = parts[0]
          # Check if suffix matches task roughly?
          # Or just assume it's the right one.
          trial_name = core_name + ext
          # Lookup trial name
          meta_img = meta_imgs.get(trial_name)

      # Find bbox for task category
      found_bbox = None
      if meta_img and task:
        cat_id = meta_cats.get(task)
        if cat_id and 'instances' in meta_img:
          for inst in meta_img['instances']:
            if inst.get('category_id') == cat_id:
              if 'bbox' in inst:
                # Transform to display coord
                t_bbox = get_transformed_bbox(inst['bbox'], meta_img, TARGET_SIZE)
                if t_bbox:
                  found_bbox = t_bbox
                  break

      if found_bbox:
        nbx, nby, nbw, nbh = found_bbox

        # Update bbox stats
        if (nbx != bx or nby != by or nbw != bw or nbh != bh):
          bbox_updates_count += 1

        # Update bbox for this row
        bx, by, bw, bh = nbx, nby, nbw, nbh

        if scanpath:
          # Check scanpath fixations against NEW bbox
          xs = scanpath.get('X', [])
          ys = scanpath.get('Y', [])

          hit = False
          for fx, fy in zip(xs, ys):
            if (bx <= fx <= bx + bw) and (by <= fy <= by + bh):
              hit = True
              break

          new_val = 1 if hit else 0
          if new_val != is_correct:
            corrections_made_count += 1

          is_correct = new_val

    new_corrects.append(is_correct)
    new_bbox_xs.append(bx)
    new_bbox_ys.append(by)
    new_bbox_ws.append(bw)
    new_bbox_hs.append(bh)

  # Apply updates
  df['correct'] = new_corrects
  df['bbox_x'] = new_bbox_xs
  df['bbox_y'] = new_bbox_ys
  df['bbox_w'] = new_bbox_ws
  df['bbox_h'] = new_bbox_hs

  print(f'Total rows: {len(df)}')
  print(f'BBox updates: {bbox_updates_count}')
  print(f'Correctness input flips: {corrections_made_count}')

  # Filter: Keep only correct=1
  df = df[df['correct'] == 1]
  print(f'Rows after filtering correct=1: {len(df)}')

  # Filter: Remove knife
  df = df[df['task'] != 'knife']
  print(f'Rows after filtering no knife: {len(df)}')

  # Filter: Remove bad bboxes (too large or invalid)
  # Area check
  img_area = TARGET_SIZE[0] * TARGET_SIZE[1]

  def is_bbox_valid(row: pd.Series) -> bool:
    w = row['bbox_w']
    h = row['bbox_h']
    x = row['bbox_x']
    y = row['bbox_y']

    if pd.isna(w) or pd.isna(h) or w <= 0 or h <= 0:
      return False

    area = w * h
    if area > (img_area * 0.9):
      return False

    # Check boundary containment (already clipped in get_transformed_bbox but good to double check)
    if x < 0 or y < 0 or (x + w) > TARGET_SIZE[0] or (y + h) > TARGET_SIZE[1]:
      # Allow small float errors
      if x < -1 or y < -1: return False
      if (x+w) > (TARGET_SIZE[0]+1) or (y+h) > (TARGET_SIZE[1]+1): return False

    return True

  df = df[df.apply(is_bbox_valid, axis=1)]
  print(f'Rows after filtering invalid/large bboxes: {len(df)}')

  print(f'Saving to {args.output}...')
  df.to_csv(args.output, index=False)
  print('Done.')

if __name__ == '__main__':
  main()
