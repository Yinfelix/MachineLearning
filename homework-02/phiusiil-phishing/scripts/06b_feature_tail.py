"""Regenerate fold-local polynomial comparisons and held-out explanations."""
from importlib import import_module
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import_module("06_feature_engineering").main(explanations_only=True)
