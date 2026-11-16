"""Convenience entry point: `python run_pipeline.py [--quick] [--raw-dir data/raw]`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rossmann.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
