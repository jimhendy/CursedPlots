import curses
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

import _curses
import numpy as np

from ..utils import data_utils


class PlottingException(Exception):
    """Simple exception to localise errors to Plot subclasses"""


class Plot(ABC):  # pylint: disable=too-many-instance-attributes
    """
    Abstract base class for a plot. Implements screen access functionality

    TODO Tidy up:
    grid_maxs:
        2-tuple of ints of the maximum value allowed in the grid.
        The grid is assumed to start at 0 in both dimensions.
        These are domain of the output data.

    x_lims:
        (Optional) 2-tuple of x-axis min, max. If not supplied (`None`) the
        extremes of the data are used.

    y_lims:
        (Optional) As for `x_lims` but for the y-axis.

    buffer:
        Fraction of extra space to add between the data extremes and edge of the grid.
        If `x_lims` or `y_lims` are supplied, buffer is **NOT** used.
        E.g. if `buffer=0.1`, 10% the maximum x value will be at
        `grid_shape[0] - buffer * (max(data[0]) - min(data[0]))`
    """

    CHARACTERS = list(
        r"""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. """
    )[::-1]
    N_CHCRACTERS = len(CHARACTERS)

    def __init__(
        self,
        screen: _curses.window,
        x_lims: Optional[Tuple[int, int]] = None,
        y_lims: Optional[Tuple[int, int]] = None,
        buffer: float = 0.05,
    ) -> None:
        self.screen = screen
        self.x_lims = x_lims  # Limits in coordinate space
        self.y_lims = y_lims
        self._static_x_lims = self.x_lims is not None
        self._static_y_lims = self.y_lims is not None
        self.buffer = buffer
        self.screen_size = self._fetch_screen_size()
        self._init_curses()

    @property
    def grid_maxs(self) -> Tuple[int, ...]:
        """
        The maximum grid row/column we are willing to plot in
        Slightly reduced from total grid to ensure we don't overflow
        """
        return tuple(i - 2 for i in self.screen_size[::-1])

    def _init_curses(self) -> None:
        curses.curs_set(False)
        curses.start_color()
        curses.use_default_colors()
        for i in range(curses.COLORS):
            curses.init_pair(i + 1, i, -1)
        self.screen.clear()

    def _fetch_screen_size(self) -> Tuple[int, int]:
        rows, cols = self.screen.getmaxyx()
        return rows - 1, cols

    @property
    def rows(self) -> int:
        """Number of rows on screen"""
        return self.screen_size[0]

    @property
    def columns(self) -> int:
        """Number of columns on screen"""
        return self.screen_size[1]

    def clear(self) -> None:
        """Clear the current screen by filling all characters with spaces"""
        window_size = self.screen.getmaxyx()  # Use directly for full clean
        empty_string = " " * window_size[0] * (window_size[1] - 1)
        self.screen.addstr(0, 0, empty_string)

    def _character_from_alpha(self, alpha: float) -> str:
        assert 0 <= alpha <= 1
        return Plot.CHARACTERS[int(alpha * (self.N_CHCRACTERS - 0.5))]

    def set_char(
        self, row_num: int, col_num: int, char: str, color_num: int = 0
    ) -> None:
        """
        Add a character to the screen at position `row_num` and `col_num`.
        """
        row_num = self.rows - row_num  # Invert y-axis
        if (0 <= row_num < self.rows) and (0 <= col_num < self.columns):
            self.screen.addch(row_num, col_num, char, curses.color_pair(color_num + 2))

    def refresh(self) -> None:
        """Print the current screen to the terminal"""
        self.screen.refresh()

    def _add_axes(self) -> None:
        """
        Add x- and y-axis to the plot
        """
        show_y = self.y_lims and ((self.y_lims[1] * self.y_lims[0]) < 0)
        show_x = self.x_lims and ((self.x_lims[1] * self.x_lims[0]) < 0)
        if not show_x and not show_y:
            return
        origin = (
            self._translate_data_to_grid(
                data=data_utils.xy_to_data([0], [0]),
            )
            .round(0)
            .astype(int)[0]
        )
        if show_y:
            self._add_axis(
                data=data_utils.xy_to_data(
                    np.arange(self.columns),
                    np.full(fill_value=origin[1], shape=self.columns),
                ).astype(int),
                character="-",
            )
        if show_x:
            self._add_axis(
                data=data_utils.xy_to_data(
                    np.full(fill_value=origin[0], shape=self.rows),
                    np.arange(self.rows),
                ).astype(int),
                character="|",
            )
            if show_y:
                self._add_axis(
                    data=data_utils.xy_to_data(x=[origin[0]], y=[origin[1]]).astype(
                        int
                    ),
                    character="+",
                )

    def _add_axis(self, data: np.ndarray, character: str) -> None:
        for point in data:
            self.set_char(
                row_num=point[1],
                col_num=point[0],
                char=character,
                color_num=6,
            )

    @abstractmethod
    def plot(self) -> Any:
        """Abstract method for plotting the subclass"""

    @staticmethod
    def _get_extreme(
        lims: Optional[Tuple[float, float]],
        extreme_func: Callable,
        series: np.ndarray,
        index: int,
        buffer: float,
    ) -> float:
        """
        Find the max/min of a series or use the supplied limit if available
        """
        if lims:
            assert len(lims) == 2, f"Limits must be of length 2: {lims}"
            assert (
                lims[1] > lims[0]
            ), f"Second limit must be larger than first (min, max): {lims}"
            extreme = lims[index]
        else:
            extreme = extreme_func(series)
            if buffer:
                offset = np.ptp(series) * buffer
                extreme += offset if extreme_func is np.max else -offset
        return extreme

    @staticmethod
    def _get_max(
        lims: Optional[Tuple[float, float]], series: np.ndarray, buffer: float = 0
    ) -> float:
        return Plot._get_extreme(
            lims=lims, series=series, extreme_func=np.max, index=1, buffer=buffer
        )

    @staticmethod
    def _get_min(
        lims: Optional[Tuple[float, float]], series: np.ndarray, buffer: float = 0
    ) -> float:
        return Plot._get_extreme(
            lims=lims, series=series, extreme_func=np.min, index=0, buffer=buffer
        )

    def _translate_data_to_grid(
        self,
        data: np.ndarray[float, float],
    ) -> np.ndarray[float, float]:
        """
        Translate plot data (x,y) to coordinates on a grid of shape `grid_shape`.

        Optionally axis limits can be passed via the args: TODO
        If no limits are passed, the data extremes are used as limits considering the `buffer`.

        Args:
            data:
                2d array of floats. The input data in (x,y) coordinate space.
                `x` = data_utils.data_x(data), `y` = data_utils.data_y(data).
                `len(data.shape) == 2`
                `data.shape[1] == 2`


        """
        # assert shapes are correct
        data_utils.assert_data_shape(data)

        x = data_utils.data_x(data)
        y = data_utils.data_y(data)

        # get realised maxs n mins
        maxs = np.array(
            (
                self._get_max(self.x_lims, x, self.buffer),
                self._get_max(self.y_lims, y, self.buffer),
            )
        )
        mins = np.array(
            (
                self._get_min(self.x_lims, x, self.buffer),
                self._get_min(self.y_lims, y, self.buffer),
            )
        )

        # do translation
        data = np.multiply(
            np.divide(np.subtract(data, mins), np.subtract(maxs, mins)), self.grid_maxs
        )

        return data
