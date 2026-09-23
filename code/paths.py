"""Where a run writes, created on import.

Neither directory is tracked in the repository, so on a fresh clone they do not
exist yet; making them here is what lets a script run without any setup step.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"

RESULTS.mkdir(exist_ok=True)
MANUSCRIPT.mkdir(exist_ok=True)
