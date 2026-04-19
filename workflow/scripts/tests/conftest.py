import sys
from pathlib import Path

# Make `from workflow.scripts import _common` importable when pytest is run
# from the repo root.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
