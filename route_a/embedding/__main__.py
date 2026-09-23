"""Command-line interface for the embedding-conditioned Route A predictor.

Subcommands mirror the pipeline steps in ``README.md``: extract features,
train (main model or an ablation), then evaluate globally and per subject.
Each subcommand delegates to that step's own module, so ``--help`` on a
subcommand shows exactly that module's flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from route_a.embedding import evaluation, evaluation_per_subject, training
from shared import features as shared_features
from shared.constants import COCO_SEARCH18_CATEGORIES, NSD_SUBJECTS

_COMMANDS = ('extract-features', 'train', 'evaluate-global', 'evaluate-per-subject')


def _extract_features_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Extract and cache DINOv2 and CLIP embeddings.')
  parser.add_argument('--nsd-root', type=Path, required=True, help='Root of the NSD/Algonauts-2023 data tree.')
  parser.add_argument('--dino-dir', type=Path, required=True, help='Output directory for DINO features.')
  parser.add_argument('--clip-img-dir', type=Path, required=True, help='Output directory for CLIP image features.')
  parser.add_argument('--clip-text-path', type=Path, required=True, help='Output path for CLIP text embeddings.')
  parser.add_argument('--subjects', nargs='+', default=None, help='Defaults to all 8 NSD subjects.')
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--skip-dino', action='store_true')
  parser.add_argument('--skip-clip', action='store_true')
  parser.add_argument('--device', default='cuda')
  return parser


def _extract_features(argv: Sequence[str]) -> int:
  args = _extract_features_parser().parse_args(argv)
  subjects = args.subjects or NSD_SUBJECTS
  if not args.skip_dino:
    shared_features.extract_and_save_dino(
      args.nsd_root, args.dino_dir, subjects, device=args.device, batch_size=args.batch_size,
    )
  if not args.skip_clip:
    shared_features.extract_and_save_clip(
      args.nsd_root, args.clip_img_dir, args.clip_text_path, COCO_SEARCH18_CATEGORIES, subjects,
      device=args.device, batch_size=args.batch_size,
    )
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """Run the Route A embedding-conditioned command-line interface."""
  argv = list(sys.argv[1:] if argv is None else argv)
  if not argv or argv[0] in ('-h', '--help'):
    print(f'Usage: python -m route_a.embedding {{{",".join(_COMMANDS)}}} ...')
    return 0 if argv else 2

  command, rest = argv[0], argv[1:]
  if command == 'extract-features':
    return _extract_features(rest)
  if command == 'train':
    return training.main(rest)
  if command == 'evaluate-global':
    return evaluation.main(rest)
  if command == 'evaluate-per-subject':
    return evaluation_per_subject.main(rest)

  print(f'Unknown command {command!r}. Choose from: {", ".join(_COMMANDS)}')
  return 2


if __name__ == '__main__':
  raise SystemExit(main())
