"""Compare rerun tables against values transcribed from Tables 1, 2, 3, and A1.

Published rounding is checked without treating Monte Carlo differences as proof
that a model is invalid. Missing results and non-finite values remain explicit.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def compare(run: Path, reference: Path, tables: list[str] | None = None) -> list[dict[str, str | float]]:
  """Return one comparison per published table cell.

  :param run: Directory containing the selected COCO model suites.
  :param reference: CSV containing published values and their printed precision.
  :param tables: Paper table identifiers to compare, or all tables when omitted.
  :returns: Comparisons with differences and explicit availability status.
  """
  with reference.open() as stream:
    expected = list(csv.DictReader(stream))
  if tables is not None:
    unknown = set(tables) - {row['table'] for row in expected}
    if unknown:
      raise ValueError('Unknown paper tables: ' + ', '.join(sorted(unknown)))
    expected = [row for row in expected if row['table'] in tables]
  if not expected:
    raise ValueError('No reference cells selected.')
  cache: dict[Path, dict[str, dict[str, str]]] = {}
  results: list[dict[str, str | float]] = []
  for row in expected:
    path = run / row['suite'] / row['file']
    if path not in cache:
      if path.exists():
        with path.open() as stream:
          cache[path] = {record['model']: record for record in csv.DictReader(stream)}
      else:
        cache[path] = {}
    result: dict[str, str | float] = dict(row)
    result.update(observed='', difference='', status='missing')
    value = cache[path].get(row['model'], {}).get(row['metric'])
    if value is not None:
      observed = float(value)
      published = float(row['published'])
      result.update(observed=observed, difference=observed - published, status='nonfinite')
      if math.isfinite(observed):
        precision = int(row['decimals'])
        matches = f'{observed:.{precision}f}' == f'{published:.{precision}f}'
        result['status'] = 'matches_printed_value' if matches else 'differs'
    results.append(result)
  return results


def main() -> None:
  parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
  parser.add_argument('--results-dir', dest='run', type=Path, required=True)
  parser.add_argument(
    '--reference-file',
    dest='reference',
    type=Path,
    default=Path(__file__).resolve().parents[1] / 'paper_tables.csv',
  )
  parser.add_argument('--output-dir', dest='output', type=Path, required=True)
  parser.add_argument('--tables', nargs='+', help='Compare only the named paper tables, such as 1 A1.')
  args = parser.parse_args()
  rows = compare(args.run, args.reference, args.tables)
  args.output.mkdir(parents=True, exist_ok=True)
  with (args.output / 'table_differences.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  counts = dict(Counter(str(row['status']) for row in rows))
  summary = {
    'run': str(args.run.resolve()), 'reference': str(args.reference.resolve()), 'counts': counts,
    'tables': sorted({str(row['table']) for row in rows}),
    'all_cells_available': not counts.get('missing', 0),
    'all_match_printed_values': counts.get('matches_printed_value', 0) == len(rows),
    'criterion': 'Exact agreement at the precision printed in the paper; differences are reported without retuning.',
  }
  (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
