from pathlib import Path
import pytest

EXCLUDED_DIRECTORIES = []

def test_expected_location():
    assert Path(__file__).parents[1].name == 'tests'

def test_all_inits_available():
    base_loc = Path(__file__).parents[1]
    