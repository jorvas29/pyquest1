"""Permite ejecutar `python -m predictor ...`."""

import sys

from .cli import main

sys.exit(main())
