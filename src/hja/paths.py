"""Repository-relative paths (the package is installed editable, so
__file__ stays inside the repo)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SCIENCEBASE = DATA / "sciencebase"
HUGGINGFACE = DATA / "huggingface"
RETRAIN = DATA / "retrain"
RESULTS = REPO / "results"
