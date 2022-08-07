import sys
from pathlib import Path


def ensure_module_available():
    try:
        import cursed_plots
    except ImportError:
        current_location = Path(__file__)
        expected_location = current_location.parents[1] / "src"
        sys.path.append(str(expected_location))
