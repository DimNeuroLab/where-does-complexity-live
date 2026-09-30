"""Export posterior-mean complexity rankings with the historical CSV schema."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def export_ranking(posterior: Path, trials: Path, output: Path, plot: bool = True) -> Path:
  """Export ``image,score,task`` in descending posterior-mean order.

  The filename-to-task mapping retains the original last-entry behavior.
  """
  data = pd.read_csv(trials)
  mapping = data[['image', 'task']].dropna().drop_duplicates().set_index('image')['task'].to_dict()
  with xr.open_dataset(posterior, group='posterior', engine='h5netcdf') as dataset:
    variable = 'img_re' if 'img_re' in dataset else 'C_image'
    means = dataset[variable].mean(dim=['chain', 'draw']).load()
  ranking = pd.Series(means.values, index=means.coords['image'].values).sort_values(ascending=False)
  frame = ranking.rename_axis('image').rename('score').reset_index()
  frame['task'] = frame['image'].map(mapping).fillna('unknown')
  output.mkdir(parents=True, exist_ok=True)
  csv_path = output / f'full_ranking_{posterior.stem}.csv'
  frame.to_csv(csv_path, index=False)
  if plot:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(frame['score'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axis.axvline(frame['score'].mean(), color='red', linestyle='dashed', label='Mean')
    axis.set(xlabel='Difficulty score (posterior mean)', ylabel='Count', title=posterior.stem)
    axis.legend()
    figure.savefig(output / f'distribution_{posterior.stem}.png')
    plt.close(figure)
  return csv_path


def main() -> None:
  parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
  sources = parser.add_mutually_exclusive_group(required=True)
  sources.add_argument('--posterior-file', dest='posterior', type=Path)
  sources.add_argument('--results-dir', type=Path)
  parser.add_argument('--trials-file', dest='trials', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--no-plots', action='store_true')
  args = parser.parse_args()
  paths = [args.posterior] if args.posterior else sorted(args.results_dir.glob('*.nc'))
  if not paths:
    parser.error('No posterior files found.')
  for path in paths:
    print(export_ranking(path, args.trials, args.output_dir, not args.no_plots))


if __name__ == '__main__':
  main()
