
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plot_images(ranking: pd.Series, img_map: dict[str, str], model_name: str, images_dir: Path, output_dir: Path) -> None:
  # Select images
  n = len(ranking)
  if n < 15:
    print(f"Not enough images to sample (found {n}). Skipping plot.")
    return

  top_5 = ranking.head(5)

  mid_center = n // 2
  mid_5 = ranking.iloc[mid_center - 2 : mid_center + 3]

  bottom_5 = ranking.tail(5)

  # Group them for plotting
  groups = {
    'Top 5 (Hardest)': top_5,
    'Middle 5 (Median)': mid_5,
    'Bottom 5 (Easiest)': bottom_5
  }

  # Collect images first to determine sizes
  rows_data = []

  max_w = 0
  max_h = 0

  for group_name, series in groups.items():
    row_items = []
    for img_name, score in series.items():
      task = img_map.get(img_name)

      # Try to find the image
      img_path = None
      if task:
        img_path = images_dir / task / img_name

      # If not found or task unknown, try searching
      if not img_path or not img_path.exists():
        found = list(images_dir.glob(f"*/{img_name}"))
        if found:
          img_path = found[0]
        else:
          img_path = images_dir / img_name  # Last attempt: directly in images dir

      img_obj = None
      if img_path and img_path.exists():
        try:
          img_obj = Image.open(img_path)
          max_w = max(max_w, img_obj.width)
          max_h = max(max_h, img_obj.height)
        except Exception:
          pass

      row_items.append({
        'img': img_obj,
        'name': img_name,
        'score': score,
        'task': task
      })
    rows_data.append((group_name, row_items))

  # Calculate optimal figure size
  if max_w == 0: max_w = 500
  if max_h == 0: max_h = 500

  # Desired width per subplot in inches
  subplot_w_inch = 3.0
  # Calculate height based on aspect ratio
  aspect_ratio = max_h / max_w
  subplot_h_inch = subplot_w_inch * aspect_ratio

  # Total figsize
  cols = 5
  rows = 3
  fig_w = cols * subplot_w_inch
  fig_h = rows * subplot_h_inch

  # Add extra space for labels
  fig_w += 1
  fig_h += 1.5

  fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)

  fig.suptitle(f"Difficulty Ranking: {model_name}\n(Posterior mean)", fontsize=16)

  for i, (group_name, row_items) in enumerate(rows_data):
    for j, item in enumerate(row_items):
      ax = axes[i, j]

      # Set row label on first column
      if j == 0:
        ax.set_ylabel(group_name, fontsize=12)

      img = item['img']
      img_name = item['name']
      score = item['score']
      task = item['task']

      if img:
        ax.imshow(img)
        # Clean spines
        for spine in ax.spines.values():
          spine.set_visible(False)
      else:
        ax.text(0.5, 0.5, f"Image not found\n{img_name}", ha='center')
        ax.set_facecolor('#eeeeee')

      task_display = task if task else 'unknown'
      ax.set_title(f"Score: {score:.3f}\nTask: {task_display}\n{img_name}", fontsize=9)
      ax.set_xticks([])
      ax.set_yticks([])

  output_dir.mkdir(exist_ok=True, parents=True)
  out_path = output_dir / f"ranking_{model_name}.png"
  plt.savefig(out_path)
  print(f"Saved plot to {out_path}")
  plt.close(fig)

def main() -> None:
  """Plot ranked examples from an exported CSV and a configurable image tree."""
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--ranking', type=Path, required=True)
  parser.add_argument('--images-dir', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()
  frame = pd.read_csv(args.ranking)
  ranking = frame.set_index('image')['score'].sort_values(ascending=False)
  mapping = frame.set_index('image')['task'].to_dict()
  plot_images(ranking, mapping, args.ranking.stem, args.images_dir, args.output_dir)


if __name__ == '__main__':
  main()
