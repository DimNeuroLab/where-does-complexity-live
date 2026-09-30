"""Public commands for Complexity."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


COMMANDS = {'run': 'complexity.run'}


def main(argv: Sequence[str] | None = None) -> int:
  """Dispatch to the selected workflow while retaining its own argument parser."""
  arguments = list(sys.argv[1:] if argv is None else argv)
  parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
  parser.add_argument('command', choices=list(COMMANDS))
  if not arguments or arguments[0] in ('-h', '--help'):
    parser.print_help()
    return 0 if arguments else 2
  selected = parser.parse_args(arguments[:1])
  module = importlib.import_module(COMMANDS[selected.command])
  module.main(arguments[1:])
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
