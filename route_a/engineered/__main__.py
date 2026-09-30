"""Command-line interface for engineered stimulus readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from route_a.engineered.features import FeatureConfig, ImageTargetFeatureExtractor
from route_a.engineered.predictor import EngineeredPredictor
from route_a.engineered.schema import validate_pairs
from route_a.engineered.training import TrainingConfig, train_engineered


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Train or use the engineered Route A complexity predictor.')
  commands = parser.add_subparsers(dest='command', required=True)

  train = commands.add_parser('train', help='Train a complete model bundle from labeled image-target pairs.')
  train.add_argument('--data', type=Path, required=True, help='CSV with image_id,image_path,target,score columns.')
  train.add_argument('--image-root', type=Path, default=None, help='Base directory for relative image paths.')
  train.add_argument('--out', type=Path, required=True, help='New directory for the trained model bundle.')
  train.add_argument('--cache-dir', type=Path, default=None, help='Feature cache directory.')
  train.add_argument('--device', default='auto', help='Feature extraction device: auto, cpu, or cuda[:index].')
  train.add_argument('--n-estimators', type=int, default=300)
  train.add_argument('--pca-components', type=int, default=64)
  train.add_argument('--cv-folds', type=int, default=5, help='Grouped validation folds; use 0 to skip validation.')
  train.add_argument('--seed', type=int, default=42)
  train.add_argument('--overwrite', action='store_true', help='Replace files in an existing output directory.')

  predict = commands.add_parser('predict', help='Predict from one image-target pair or an input CSV.')
  predict.add_argument('--model', type=Path, required=True, help='Directory containing the complete model bundle.')
  source = predict.add_mutually_exclusive_group(required=True)
  source.add_argument('--image', type=Path, help='One input image.')
  source.add_argument('--input', type=Path, help='CSV with image_path,target columns.')
  predict.add_argument('--target', help='Search target for --image.')
  predict.add_argument('--image-root', type=Path, default=None, help='Base directory for relative CSV image paths.')
  predict.add_argument('--out', type=Path, help='Output CSV path for batch predictions.')
  predict.add_argument('--cache-dir', type=Path, default=None, help='Feature cache directory.')
  predict.add_argument('--device', default=None, help='Override bundle device: auto, cpu, or cuda[:index].')
  predict.add_argument('--overwrite', action='store_true', help='Replace an existing batch output CSV.')
  return parser


def _read_table(path: Path) -> pd.DataFrame:
  if not path.is_file():
    raise FileNotFoundError(f'Input CSV does not exist: {path}')
  return pd.read_csv(path, dtype=str)


def _train(args: argparse.Namespace) -> int:
  if args.out.exists() and not args.out.is_dir():
    raise ValueError(f'Output path is not a directory: {args.out}.')
  if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
    raise ValueError(f'Output directory already has files: {args.out}. Use --overwrite to replace them.')
  if args.cv_folds == 1 or args.cv_folds < 0:
    raise ValueError('--cv-folds must be 0 or at least 2.')
  labels = validate_pairs(_read_table(args.data), require_score=True, image_root=args.image_root)
  feature_config = FeatureConfig(device=args.device)
  extractor = ImageTargetFeatureExtractor(config=feature_config, cache_dir=args.cache_dir)
  features = extractor.transform(labels)
  training_config = TrainingConfig(
    pca_components=args.pca_components,
    n_estimators=args.n_estimators,
    cv_folds=args.cv_folds,
    seed=args.seed,
  )
  trained = train_engineered(features, feature_config, training_config)
  output = trained.bundle.save(args.out)
  if not trained.validation_predictions.empty:
    trained.validation_predictions.to_csv(output / 'validation_predictions.csv', index=False)
  else:
    (output / 'validation_predictions.csv').unlink(missing_ok=True)
  print(json.dumps({'model_bundle': str(output), 'training_rows': len(features)}, indent=2))
  return 0


def _predict(args: argparse.Namespace) -> int:
  if args.image is not None:
    if args.target is None:
      raise ValueError('--target is required with --image.')
    if args.out is not None:
      raise ValueError('--out is for batch prediction with --input.')
  elif args.target is not None:
    raise ValueError('--target is only used with --image; batch CSV rows supply their targets.')
  if args.input is not None and args.out is None:
    raise ValueError('--out is required with --input.')
  if args.input is not None and args.out is not None:
    if args.input.resolve() == args.out.resolve():
      raise ValueError('Batch output must differ from the input CSV.')
    if args.out.exists() and not args.overwrite:
      raise ValueError(f'Output CSV already exists: {args.out}. Use --overwrite to replace it.')
  predictor = EngineeredPredictor.from_bundle(args.model, cache_dir=args.cache_dir, device=args.device)
  if args.image is not None:
    value = predictor.predict_image(args.image, args.target)
    print(json.dumps({'image_path': str(args.image), 'target': args.target, 'prediction': value}))
    return 0
  rows = _read_table(args.input)
  predictions = predictor.predict_batch(rows, image_root=args.image_root)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  predictions.to_csv(args.out, index=False)
  print(json.dumps({'output': str(args.out), 'rows': len(predictions)}))
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """Run the Route A command-line interface."""
  parser = _parser()
  args = parser.parse_args(argv)
  try:
    if args.command == 'train':
      return _train(args)
    if args.command == 'predict':
      return _predict(args)
    parser.error(f'Unknown command {args.command!r}.')
  except (ValueError, FileNotFoundError) as error:
    parser.exit(2, f'error: {error}\n')
  return 2


if __name__ == '__main__':
  raise SystemExit(main())
