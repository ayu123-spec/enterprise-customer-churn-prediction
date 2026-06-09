import sys
from pathlib import Path

# Make `src` importable in tests no matter where pytest is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))