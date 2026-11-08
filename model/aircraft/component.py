from model.aircraft.base import Component, Position
from model.aircraft.plot import _plot
from model.util.output import Axes
from dataclasses import dataclass, InitVar, field
from typing import Any, Callable, Self
from model.aircraft import *


@dataclass(frozen=True)
class MassComponent(Component):

    dx: float = field(kw_only=True)
    """length in x direction (longitudinal) [m]"""
    dy: float = field(kw_only=True)
    """length in y direction (lateral) [m]"""
    dz: float = field(kw_only=True)
    """length in z direction (vertical) [m]"""

    center: Position = field(init=False, repr=False)  # type: ignore
    fixed_center: Position = field(default_factory=Position, kw_only=True)
    """fixed center position"""
    dynamic_center: Callable[[], Position] | None = field(
        default=None, kw_only=True
    )
    """function that returns the center position when called, allowing the position to be dynamically determined"""

    @property
    def center(self) -> Position:
        if self.dynamic_center is not None:
            return self.fixed_center + self.dynamic_center()
        else:
            return self.fixed_center

    @center.setter
    def center(self, value: Position | Callable[[], Position]) -> None:
        if callable(value) and isinstance(value(), Position):
            self._set("dynamic_center", value)
        elif isinstance(value, Position):
            self._set("fixed_center", value)
        elif isinstance(value, property):
            pass
        else:
            raise TypeError(f"Unsupported type ({type(value)})")

    def copy(self, name: str, offset: Position | Callable[[], Position] = Position()) -> Self:
        return super().copy(name, offset)  # type: ignore

    def move(self, offset: Position | Callable[[], Position]) -> None:
        if callable(offset):
            if not isinstance(offset(), Position):
                raise TypeError(f"Unsupported type ({type(offset)})")

            previous_dynamic = self.dynamic_center
            if previous_dynamic is not None:
                self._set(
                    "dynamic_center",
                    lambda: previous_dynamic() + offset(),
                )
            else:
                self._set("dynamic_center", lambda: offset())

        elif isinstance(offset, Position):
            self._set("fixed_center", self.fixed_center + offset)
            self.mass._move(offset)

        else:
            raise TypeError(f"Unsupported type ({type(offset)})")

    def _draw_planform(self, ax: Axes):
        _plot.plot_box(
            ax,
            center_x=self.center.x,
            center_y=self.center.y,
            width=self.dx,
            height=self.dy,
            label=self.component_name
        )

    def _draw_side_elevation(self, ax: Axes):
        _plot.plot_box(
            ax,
            center_x=self.center.x,
            center_y=self.center.z,
            width=self.dx,
            height=self.dz,
            label=self.component_name
        )
