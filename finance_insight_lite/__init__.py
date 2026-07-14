"""Source-checkout import shim for the ``src`` layout package.

This lets ``python -m finance_insight_lite...`` work from the repository root
without requiring an editable install or a manually-set PYTHONPATH.
"""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "finance_insight_lite"

if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

