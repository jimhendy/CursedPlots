import sys
from pathlib import Path


def ensure_module_available() -> None:
    """
    Ensure the cursed_plots module is available for import
    If not, assume we are in the default folder structure and add the expected location to the path
    """
    try:
        import cursed_plots  # pylint: disable=import-outside-toplevel, unused-import
    except ImportError:
        current_location = Path(__file__)
        expected_location = current_location.parents[1] / "src"
        sys.path.append(str(expected_location))
