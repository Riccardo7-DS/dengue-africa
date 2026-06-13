from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"

# Optional machine-local paths (defined in local_paths.py, which is git-ignored).
# Copy src/local_paths.py.example → src/local_paths.py and fill in for your machine.
try:
    from local_paths import SM_DATA_PATH, LC_DATA_PATH  # type: ignore[import]
except ImportError:
    SM_DATA_PATH = None
    LC_DATA_PATH = None
