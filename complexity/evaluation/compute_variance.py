"""Decompose complexity variance after averaging repeated scene-target records."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def decompose(path: Path, min_pairs: int = 2) -> dict[str, float | int]:
  """Return variance components for physical NSD scenes with enough targets."""
  repeats: dict[tuple[str, str], list[float]] = defaultdict(list)
  with path.open() as stream:
    for row in csv.DictReader(stream):
      match = re.search(r'nsd-(\d+)(?:_.+)?\.png$', row['image'])
      if not match:
        raise ValueError(f"Unrecognized NSD image: {row['image']}")
      repeats[(match[1], row['task'])].append(float(row['score']))
  scenes: dict[str, list[float]] = defaultdict(list)
  for (scene, _), scores in repeats.items():
    scenes[scene].append(sum(scores) / len(scores))
  groups = [scores for scores in scenes.values() if len(scores) >= min_pairs]
  values = [value for group in groups for value in group]
  if len(values) < 2:
    raise ValueError('Fewer than two scores remain.')
  mean = sum(values) / len(values)
  denominator = len(values) - 1
  total = sum((value - mean) ** 2 for value in values) / denominator
  within = sum(sum((value - sum(group) / len(group)) ** 2 for value in group) for group in groups) / denominator
  between = sum(len(group) * (sum(group) / len(group) - mean) ** 2 for group in groups) / denominator
  return {
    'total': total, 'within': within, 'between': between,
    'within_percent': 100 * within / total if total else 0.0,
    'images': len(groups), 'pairs': len(values),
  }


def main() -> None:
  parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
  parser.add_argument('--ranking-file', dest='ranking', type=Path, required=True)
  parser.add_argument('--min-pairs', type=int, default=2)
  args = parser.parse_args()
  if args.min_pairs < 2:
    parser.error('--min-pairs must be at least two.')
  import json
  print(json.dumps(decompose(args.ranking, args.min_pairs), indent=2))


if __name__ == '__main__':
  main()
