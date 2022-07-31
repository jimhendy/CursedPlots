import curses
from abc import ABC, abstractmethod
from typing import Any, Tuple

import _curses


class PlottingException(Exception):
    """Simple exception to localise errors to Plot subclasses"""


class Plot(ABC):
    """
    Abstract base class for a plot. Implements screen access functionality
    """

    CHARACTERS = list(
        r"""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. """
    )[::-1]

    def __init__(self, screen: _curses.window) -> None:
        self.screen = screen
        self.n_characters = len(Plot.CHARACTERS)
        self.screen_size = self._fetch_screen_size()

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
        return Plot.CHARACTERS[int(alpha * (self.n_characters - 0.5))]

    def set_char(
        self, row_num: int, col_num: int, char: str, color_num: int = 0
    ) -> None:
        """
        Add a character to the screen at position `row_num` and `col_num`.
        """
        # Plotting is done from row=0 at top to row=self.rows at bottom so needs inverting
        row_num = self.rows - row_num
        if row_num >= self.rows:
            raise PlottingException(
                f"Cannot add to row {row_num} as screen is only {self.rows} rows"
            )
        if col_num >= self.columns:
            raise PlottingException(
                f"Cannot add to column {col_num} as screen is only {self.columns} columns"
            )
        self.screen.addch(row_num, col_num, char, curses.color_pair(color_num + 2))

    def refresh(self) -> None:
        """Print the current screen to the terminal"""
        self.screen.refresh()

    @abstractmethod
    def plot(self) -> Any:
        """Abstract method for plotting the subclass"""
