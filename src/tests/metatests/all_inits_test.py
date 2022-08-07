from pathlib import Path

EXCLUDED_DIRECTORIES = []


class TestAllTestsAccessable:
    @staticmethod
    def test_expected_location():
        assert Path(__file__).parents[1].name == "tests"

    @staticmethod
    def test_all_inits_available():
        base_loc = Path(__file__).parents[1]
