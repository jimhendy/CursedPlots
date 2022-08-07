import time
from typing import Any, Callable, List, Optional

import numpy as np

from ..utils import anti_aliased, data_utils
from .base import Plot


class LinePlot(Plot):
    """
    Simple line plot implementing anti-aliasing
    """

    def __init__(
        self, functions: List[Callable], step_delay: float = 6e-2, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.functions = functions
        self.step_delay = step_delay

    def _fill_grid(self, all_data: List[np.ndarray]) -> None:

        self._set_x_y_lims(all_data)

        for color_num, data in enumerate(all_data):
            grid_data = self._translate_data_to_grid(data)
            interp_data = anti_aliased.interpolate_regularly(grid_data)

            for point, alpha in zip(*anti_aliased.anti_aliased_data(interp_data)):
                self.set_char(
                    row_num=point[1],
                    col_num=point[0],
                    char=self._character_from_alpha(alpha),
                    color_num=color_num,
                )

    def _set_x_y_lims(self, all_data: List[np.ndarray]) -> None:

        if self._static_x_lims and self._static_y_lims:
            return

        x_min = data_utils.data_x(all_data[0]).min()
        x_max = data_utils.data_x(all_data[0]).max()
        y_min = data_utils.data_y(all_data[0]).min()
        y_max = data_utils.data_y(all_data[0]).max()
        for data in all_data[1:]:
            x_min = min(x_min, data_utils.data_x(data).min())
            x_max = min(x_max, data_utils.data_x(data).max())
            y_min = min(y_min, data_utils.data_y(data).min())
            y_max = min(y_max, data_utils.data_y(data).max())

        if not self._static_x_lims:
            self.x_lims = (x_min, x_max)
        if not self._static_y_lims:
            self.y_lims = (y_min, y_max)

    def plot(self, iterations: Optional[int] = None) -> None:
        """
        Animate line plots for the supplied `functions`.
        """

        for time_ in range(iterations or int(10e10)):
            self.clear()
            data = [func(time_) for func in self.functions]
            self._fill_grid(all_data=data)

            self._add_axes()
            self.refresh()
            time.sleep(self.step_delay)

        time.sleep(4)
