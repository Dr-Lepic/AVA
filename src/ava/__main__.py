"""Allow `python -m ava` in addition to the installed `ava` script."""

from ava.cli import main

raise SystemExit(main())
