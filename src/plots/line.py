import time
from typing import Callable, List, Optional, Tuple

import _curses
import numpy as np

from utils import anti_aliased

from .base import Plot


class LinePlot(Plot):
    """
    Simple line plot implementing anti-aliasing
    """

    def __init__(self, screen: _curses.window, functions: List[Callable]) -> None:
        super().__init__(screen=screen)
        self.functions = functions

    def _fill_grid(self, data: np.ndarray, color_num: int) -> None:

        grid_maxs: Tuple[int, ...] = tuple(i - 2 for i in self.screen_size[::-1])
        locations, weights = anti_aliased.anti_alias(data, grid_maxs)

        for point, alpha in zip(locations, weights):
            self.set_char(
                row_num=point[1],
                col_num=point[0],
                char=self._character_from_alpha(alpha),
                color_num=color_num,
            )

    def plot(self, iterations: Optional[int] = None) -> None:
        """
        Animate line plots for the supplied `functions`.
        """

        for time_ in range(iterations or int(10e10)):
            self.clear()
            for i, func in enumerate(self.functions):
                data = func(time_ * 0.1)
                self._fill_grid(data=data, color_num=i)

            self.refresh()
            time.sleep(6e-2)

        time.sleep(4)
