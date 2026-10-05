from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

RAW = DATA / "raw"
PROCESSED = DATA / "processed"

MODELS = OUTPUTS / "models"
LABELS = OUTPUTS / "labels"
REPORTS = OUTPUTS / "reports"