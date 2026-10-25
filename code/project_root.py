"""Make the repository root importable.

The course grading suite requires ``implementations.py`` to sit at the
repository root, while the rest of the pipeline lives in ``code/``. Importing
this module puts the root on ``sys.path`` so ``from implementations import ...``
resolves no matter which directory the entry point was started from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
