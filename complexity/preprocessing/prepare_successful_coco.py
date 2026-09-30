"""Prepare explicitly successful COCO trials for the revised measurement comparison."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from complexity.models import fit_movement_count, fit_response_time
from complexity.preprocessing.convert_scanpaths import json_to_csv


def counts(data: pd.DataFrame) -> dict[str, int]:
  return {
    'trials': len(data), 'image_ids': int(data['image'].nunique()),
    'image_target_pairs': len(data[['image', 'task']].drop_duplicates()),
    'subjects_or_executions': int(data['subject'].nunique()),
  }


def prepare(source: Path, output: Path) -> dict[str, Any]:
  """Filter individual target-present trials on ``correct == 1`` before assigning trial order.

  :returns: Source provenance and counts before and after each model's validity checks.
  """
  output.mkdir(parents=True, exist_ok=True)
  csv = output / 'successful.csv'
  if csv.exists():
    raise ValueError(f'Use a fresh preparation directory: {output}')
  json_to_csv(source, csv, movement='saccades', only_condition='present', only_correct=1)
  selected = pd.read_csv(csv)
  rt, _ = fit_response_time.prepare_df(selected)
  count, _ = fit_movement_count.prepare_df(selected)
  raw = json.loads(source.read_text())
  report: dict[str, Any] = {
    'source': str(source.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'csv_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(), 'raw_trials': len(raw),
    'raw_success_flags': {str(flag): sum(row.get('correct') == flag for row in raw) for flag in [1, 0, None]},
    'selection': 'condition == present and correct == 1, applied to individual trials',
    'trial_order': 'Per subject in source order, numbered after success filtering, before model validity checks.',
    'selected': counts(selected), 'rt_models': counts(rt), 'count_models': counts(count),
    'selected_without_bbox_hit': int(selected['first_target_fix_idx'].isna().sum()),
    'selected_with_zero_saccades': int((selected['N'] == 0).sum()),
    'rt_validity_excluded': len(selected) - len(rt), 'count_validity_excluded': len(selected) - len(count),
  }
  (output / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == '__main__':
  main()
