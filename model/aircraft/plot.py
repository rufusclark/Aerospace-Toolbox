from typing import Literal, TYPE_CHECKING, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.axes import Axes

from model.core.airfoil import Airfoil


class _Plot:
    def __init__(self) -> None:
        self._i = 0
        self.colour_palette = (
            list(plt.colormaps["tab20b"].colors) +  # type: ignore
            list(plt.colormaps["tab20c"].colors)  # type: ignore
        )
        # self.colour_palette = plt.colormaps["tab20"](np.linspace(0, 1, 20))

    def reset(self) -> None:
        self._i = 0

    def get_color(self) -> str:
        color = self.colour_palette[self._i]
        self._i = (self._i + 1) % len(self.colour_palette)
        return color

    def plot_box(self, ax: Axes, center_x: float, center_y: float, width: float, height: float, label: str, color: str | None = None) -> None:
        xmin = center_x - width / 2
        xmax = center_x + width / 2
        ymin = center_y - height / 2
        ymax = center_y + height / 2

        points = [
            (xmin, ymin),  # bottom-left
            (xmax, ymin),  # bottom-right
            (xmax, ymax),  # top-right
            (xmin, ymax),  # top-left
        ]

        self.plot_2d(ax, points=points, label=label, color=color)

    def plot_airfoil(self, ax: Axes, airfoil: Airfoil, chord: float, setting_angle: float, label: str, color: str | None = None, facing: Literal["left", "right"] = "left", leading_edge_x: float = 0, leading_edge_y: float = 0) -> None:
        x, y = airfoil.scale_DAT(
            chord=chord, angle=setting_angle, facing=facing)
        x = leading_edge_x + x
        y = leading_edge_y + y
        points = np.column_stack((x, y))

        self.plot_2d(
            ax, points=points, label=label, color=color  # type: ignore
        )

    def plot_2d(self, ax: Axes, points: Sequence[tuple[float, float]], label: str, color: str | None = None) -> None:
        if color is None:
            color = self.get_color()

        shape = Polygon(
            points,
            closed=True,
            alpha=0.2,
            linewidth=2,
            color=color
        )
        ax.add_patch(shape)
        outline = Polygon(
            points,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=2,
            alpha=1,
            label=label
        )
        ax.add_patch(outline)


_plot = _Plot()
