import sys
from pathlib import Path

# Repo root must be importable as `categories.*` / `utils.*` / `main` regardless
# of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
