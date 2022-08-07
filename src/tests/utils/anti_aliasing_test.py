import numpy as np

from cursed_plots.utils import anti_aliased, data_utils


class TestInterpolateRegularly:
    @staticmethod
    def test_simple_integers() -> None:
        array = np.array([0, 5, 10])
        data = data_utils.xy_to_data(x=array, y=array)
        expected_series = np.arange(array.max() + 1)
        expected = data_utils.xy_to_data(x=expected_series, y=expected_series)
        actual = anti_aliased.interpolate_regularly(data)
        np.testing.assert_almost_equal(actual, expected)

    @staticmethod
    def test_simple_floats() -> None:
        array = np.array([0.1, 5.1, 10.1])
        data = data_utils.xy_to_data(x=array, y=array)
        expected_series = np.linspace(
            array.min(), array.max(), num=int(array.max() - array.min()) + 1
        )
        expected = data_utils.xy_to_data(x=expected_series, y=expected_series)
        actual = anti_aliased.interpolate_regularly(data)
        np.testing.assert_almost_equal(actual, expected)

    @staticmethod
    def test_gradient_change_integers() -> None:
        x = np.array([0, 5, 10])
        y = np.array([0, 5, 15])
        data = data_utils.xy_to_data(x=x, y=y)
        expected_series = np.array([0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15])
        expected = data_utils.xy_to_data(x=np.arange(x.max() + 1), y=expected_series)
        actual = anti_aliased.interpolate_regularly(data)
        np.testing.assert_almost_equal(actual, expected)


class TestInverseDistanceToVertex:
    @staticmethod
    def _normalised_inverse_distance(dx, dy):
        distance = (dx**2 + dy**2) ** 0.5
        return (np.sqrt(2) - distance) / np.sqrt(2)

    @staticmethod
    def test_single_point_all_bottom_left() -> None:
        data = data_utils.xy_to_data([0.75], [0.1])
        expected = np.array([[0, 0]]), np.array(
            [TestInverseDistanceToVertex._normalised_inverse_distance(0.75, 0.1)]
        )
        actual = anti_aliased._inverse_distance_to_vertex(
            data=data, increase_x=False, increase_y=False
        )
        np.testing.assert_almost_equal(actual[0], expected[0])
        np.testing.assert_almost_equal(actual[1], expected[1])

    @staticmethod
    def test_single_point_all_bottom_right() -> None:
        data = data_utils.xy_to_data([0.75], [0.1])
        expected = np.array([[1, 0]]), np.array(
            [TestInverseDistanceToVertex._normalised_inverse_distance(0.25, 0.1)]
        )
        actual = anti_aliased._inverse_distance_to_vertex(
            data=data, increase_x=True, increase_y=False
        )
        np.testing.assert_almost_equal(actual[0], expected[0])
        np.testing.assert_almost_equal(actual[1], expected[1])

    @staticmethod
    def test_single_point_all_top_right() -> None:
        data = data_utils.xy_to_data([0.75], [0.1])
        expected = np.array([[1, 1]]), np.array(
            [TestInverseDistanceToVertex._normalised_inverse_distance(0.25, 0.9)]
        )
        actual = anti_aliased._inverse_distance_to_vertex(
            data=data, increase_x=True, increase_y=True
        )
        np.testing.assert_almost_equal(actual[0], expected[0])
        np.testing.assert_almost_equal(actual[1], expected[1])

    @staticmethod
    def test_single_point_all_top_left() -> None:
        data = data_utils.xy_to_data([0.75], [0.1])
        expected = np.array([[0, 1]]), np.array(
            [TestInverseDistanceToVertex._normalised_inverse_distance(0.75, 0.9)]
        )
        actual = anti_aliased._inverse_distance_to_vertex(
            data=data, increase_x=False, increase_y=True
        )
        np.testing.assert_almost_equal(actual[0], expected[0])
        np.testing.assert_almost_equal(actual[1], expected[1])

    @staticmethod
    def test_multiple_points_bottom_left() -> None:
        data = data_utils.xy_to_data([0.5, 1, 1.5], [0.5, 1, 1.5])
        expected_points = np.array([[0, 0], [1, 1], [1, 1]])
        expected_weights = np.array([0.5, 1, 0.5])
        actual = anti_aliased._inverse_distance_to_vertex(
            data=data, increase_x=False, increase_y=False
        )
        np.testing.assert_almost_equal(actual[0], expected_points)
        np.testing.assert_almost_equal(actual[1], expected_weights)


class TestAntiAlisedData:
    @staticmethod
    def test_single_point() -> None:
        data = data_utils.xy_to_data([0.5], [0.5])
        expected = (
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        )
        actual = anti_aliased.anti_aliased_data(data)
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            np.testing.assert_almost_equal(actual[i], expected[i])
