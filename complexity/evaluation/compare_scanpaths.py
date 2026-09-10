"""Compare generated scanpaths with a historical aggregate by trial identity."""

import argparse
import json
from pathlib import Path
from typing import Any


def indexed(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
  """Load unique task, filename, and observer records from external JSON data."""
  records = json.loads(path.read_text())
  result = {(row['task'], row['name'], row['subject']): row for row in records}
  if len(result) != len(records):
    raise ValueError(f'Duplicate trial identities in {path}')
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--generated', type=Path, required=True)
  parser.add_argument('--reference', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  current = indexed(args.generated)
  original = indexed(args.reference)
  common = current.keys() & original.keys()
  exact = 0
  same_length = 0
  same_correct = 0
  max_errors = {'X': 0.0, 'Y': 0.0, 'T': 0.0, 'RT': 0.0}
  for key in common:
    left, right = current[key], original[key]
    exact += left == right
    same_length += left['length'] == right['length']
    same_correct += left['correct'] == right['correct']
    for field in ['X', 'Y', 'T']:
      max_errors[field] = max(
        max_errors[field], max((abs(a - b) for a, b in zip(left[field], right[field])), default=0.0)
      )
    max_errors['RT'] = max(max_errors['RT'], abs(left['RT'] - right['RT']))
  missing = sorted(original.keys() - current.keys())
  extra = sorted(current.keys() - original.keys())
  report = {
    'generated': str(args.generated.resolve()), 'reference': str(args.reference.resolve()),
    'generated_records': len(current), 'reference_records': len(original), 'shared_records': len(common),
    'exact_records': exact, 'same_length_records': same_length, 'same_correct_records': same_correct,
    'missing_records': len(missing), 'extra_records': len(extra),
    'missing_examples': missing[:10], 'extra_examples': extra[:10],
    'max_absolute_errors_on_aligned_prefix': max_errors,
    'exact_match': not missing and not extra and exact == len(original),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
