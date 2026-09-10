"""Select detectable images while retaining their unsuccessful trials."""

import argparse
import json
from pathlib import Path

import pandas as pd


def select_trials(source: Path, output: Path) -> dict[str, int]:
  """Keep present trials for images with at least one fixation inside the target box.

  Image identity retains the publication code's filename grouping.
  :returns: Input and retained trial and image counts.
  """
  data = pd.read_csv(source)
  present = data.loc[data['condition'] == 'present'].copy()
  hit = pd.to_numeric(present['first_target_fix_idx'], errors='coerce').notna()
  eligible = present.loc[hit, 'image'].unique()
  selected = present.loc[present['image'].isin(eligible)]
  output.parent.mkdir(parents=True, exist_ok=True)
  selected.to_csv(output, index=False)
  report = {
    'input_trials': len(data), 'present_trials': len(present),
    'retained_trials': len(selected), 'retained_images': selected['image'].nunique(),
  }
  output.with_suffix('.selection.json').write_text(json.dumps(report, indent=2) + '\n')
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--csv', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(select_trials(args.csv, args.output), indent=2))


if __name__ == '__main__':
  main()
