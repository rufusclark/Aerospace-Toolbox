from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Literal, overload, TypeVar, Generic, Self
from matplotlib.axes import Axes
import pandas as pd
import copy


from model.util.dataclass import FrozenDataclassBase
from model.aircraft.plot import _plot
from model.util.output import *

#
# Global Assumptions
#


@dataclass
class __Assumptions:
    """global class containing all assumptions"""

    mass_factor: float = 1
    """factor to multiply all masses by"""

    reynolds_number: float = 300000
    """reynolds number for use in xFoil airfoil simulations"""

    mach_number: float = 0.1
    """mach number for use in xFoil airfoil simulations"""

    def set_reynolds_number(self, airspeed: float, wing_chord: float, air_density: float = 1.225, dynamic_viscosity: float = 1.81e-5) -> None:
        """set assumed reynolds number

        Args:
            airspeed: airspeed [m/s]
            wing_chord: wing chord [m]
            air_density: air density [kg/m^3]. Defaults to 1.225.
            dynamic_viscosity: dynamic viscosity [Pas]. Defaults to 1.81e-5.
        """
        self.reynolds_number = (
            airspeed * wing_chord * airspeed
        ) / dynamic_viscosity

    def set_mach_number(self, airspeed: float, speed_of_sound: float = 340) -> None:
        """set assumed mach number

        Args:
            airspeed: airspeed [m/s]
            speed_of_sound: speed of sound [m/s]. Defaults to 340.
        """
        self.mach_number = airspeed / speed_of_sound


assumptions = __Assumptions()

#
# Position
#


@dataclass(frozen=True)
class Position:
    """position in the aircraft fixed body frame"""
    x: float = 0
    """positive forwards (longitudinal) [m]"""
    y: float = 0
    """positive starboard (right) (lateral) [m]"""
    z: float = 0
    """positive downwards (vertical) [m]"""

    def __add__(self, other: Position) -> Position:
        if not isinstance(other, Position):
            raise TypeError(
                f"other must be type 'Position', {type(other)} provided")
        return Position(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __radd__(self, other: Position | Literal[0]) -> Position:
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: Position) -> Position:
        if not isinstance(other, Position):
            raise TypeError(
                f"other must be type 'Position', {type(other)} provided")
        return Position(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def __mul__(self, other: float | int):
        if not isinstance(other, (float, int)):
            raise TypeError(
                f"other must be type 'float', {type(other)} provided")
        other = float(other)
        return Position(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
        )

    def __truediv__(self, other: float | int) -> Position:
        if not isinstance(other, (float, int)):
            raise TypeError(
                f"other must be type 'float', {type(other)} provided")
        other = float(other)
        return Position(
            x=self.x / other,
            y=self.y / other,
            z=self.z / other,
        )

#
# Mass
#


@dataclass(frozen=True)
class Mass(FrozenDataclassBase):
    """
    represents a point mass [kg] and position (x, y, z) used to determine system mass and mass distribution.

    this recursively computes mass and cg from the child components of parent.

    components with no children must provide override_mass to set component mass. The cg can either be provided through override_cg otherwise its assumed to be the same as the component center.

    Raises:
        ValueError: invalid inputs
    """

    override_mass: float | None = None
    """mass [kg] of this component (including all child components)"""
    override_cg: Position | None = None
    """cg position of this component (including all child components)"""
    override_mass_factor: float | None = None
    """mass factor multiplied with estimated, overridden or inherited mass to determine working mass. if not provided global assumptions.mass_factor will be used instead. Set to 1 to disable mass factor for this object"""
    parent: Component = field(init=False)
    """parent component object"""

    def __set_parent(self, parent: Component) -> None:
        """do not use.

        internal method only called by `Component`
        """
        self._set("parent", parent)

    def __repr__(self) -> str:
        try:
            return f"Mass(mass={self.mass:.3f}kg cg={self.cg}"
        except:
            return f"Mass(override_made={self.override_mass}kg override_cg={self.override_cg})"

    def _move(self, offset: Position) -> None:
        if self.override_cg is not None:
            self._set("override_cg", self.override_cg + offset)

    @property
    def mass_factor(self) -> float:
        if self.override_mass_factor is not None:
            return self.override_mass_factor
        else:
            return assumptions.mass_factor

    @property
    def mass(self) -> float:
        """component mass [kg] including children"""
        if self.override_mass is not None:
            return self.override_mass * self.mass_factor

        children = self.parent.children
        if len(children) == 0:
            raise ValueError(
                f"{self.parent} has not children and override mass has not been set. Components with no children must set override mass")

        return sum([child.mass.mass for child in children])

    @property
    def cg(self) -> Position:
        """component cg position including children"""
        if self.override_cg is not None:
            return self.override_cg

        children = self.parent.children
        if len(children) == 0:
            return self.parent.center
        else:
            return sum((child.mass.cg * child.mass.mass for child in children), Position()) / self.mass

#
# Base Component
#


@dataclass(frozen=True)
class Component(FrozenDataclassBase):
    category = "aircraft"
    info: ComponentInfo = field(init=False, repr=False)
    plots: ComponentPlots = field(init=False, repr=False)

    component_name: str
    mass: Mass
    center: Position

    def __post_init__(self):
        self._set("info", ComponentInfo(self))
        self._set("plots", ComponentPlots(self))

        self.mass._Mass__set_parent(self)  # type: ignore

    def move(self, offset: Position) -> None:
        """move the component by the position offset"""
        self._set("center", self.center + offset)
        self.mass._move(offset)

    @property
    def children(self) -> list[Component]:
        """get all immediate child components including those stored within a list"""
        out: list[Component] = []

        for field in fields(self):
            if not hasattr(self, field.name):
                continue

            value = getattr(self, field.name)

            if isinstance(value, list):
                for list_value in value:
                    if isinstance(list_value, Component):
                        out.append(list_value)

            elif isinstance(value, dict):
                for list_value in value.values():
                    if isinstance(list_value, Component):
                        out.append(list_value)

            elif isinstance(value, Component):
                out.append(value)

        return out

    @property
    def component_dict(self) -> dict[str, Component]:
        """returns this component and all children keyed by component path"""
        out: dict[str, Component] = {}

        def add(component: Component, path: str) -> None:
            out[path] = component

            for child in component.children:
                add(child, f"{path}.{child.component_name}")

        add(self, self.component_name)

        return out

    @property
    def name(self) -> str:
        return self.component_name

    def _draw_planform(self, ax: Axes):
        """2d projection in xy plane"""
        raise NotImplementedError()

    def _draw_side_elevation(self, ax: Axes):
        """2d projection in xz plane"""
        raise NotImplementedError()

    def copy(self, name: str, offset: Position = Position()) -> Self:
        """return a new copy of this object with a new name and position offset if provided"""
        new = copy.deepcopy(self)
        new._set("component_name", name)
        new.move(offset)
        return new


T = TypeVar("T", bound=Component)


class ComponentPlots(PlotBase[T], Generic[T]):

    @scientific_plot(auto_limits="0.1scale", outside_legend="lower", outside_legend_cols=3)
    def planform(self, ax: Axes):
        """2d projection in xy plane"""
        model = self.model

        # reset colour wheel
        _plot.reset()

        model._draw_planform(ax)

        cg = model.mass.cg
        ax.plot(cg.x, cg.y, "*", markersize=10, markeredgewidth=2, label="CG")

        ax.set_aspect("equal")
        ax.legend()

        # ax.invert_yaxis()

        ax.set_xlabel("positive forward, x [m]")
        ax.set_ylabel("positive starboard, y [m]")

    @scientific_plot(auto_limits="0.1scale", outside_legend="lower", outside_legend_cols=3)
    def side_elevation(self, ax: Axes):
        """2d projection in xz plane"""
        model = self.model

        # reset colour wheel
        _plot.reset()

        model._draw_side_elevation(ax)

        cg = model.mass.cg
        ax.plot(cg.x, cg.z, "*", markersize=10, markeredgewidth=2, label="CG")

        ax.set_aspect("equal")
        ax.legend()

        ax.invert_yaxis()

        ax.set_xlabel("positive forward, x [m]")
        ax.set_ylabel("positive downwards, z [m]")


class ComponentInfo(InfoBase[T], Generic[T]):

    @scientific_table()
    def mass_breakdown(self):
        model = self.model

        out = []

        for name, child in model.component_dict.items():
            mass = child.mass
            cg = mass.cg
            out.append(
                {
                    "Name": name,
                    "Mass [kg]": mass.mass,
                    "Mass Factor": "N/A" if isinstance(child, ParentComponent) else mass.mass_factor,
                    "X Position [m]": cg.x,
                    "Y Position [m]": cg.y,
                    "Z Position [m]": cg.z
                }
            )

        return pd.DataFrame(out)


@dataclass(frozen=True)
class ParentComponent(Component):
    mass: Mass = field(default=None, kw_only=True)  # type: ignore
    center: Position = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self):
        # set mass to Mass() evoking inheritance behaviour
        if self.mass is None:
            self._set("mass", Mass())

        # approximate geometric center position from children
        if self.center is None:
            children = self.children
            if len(children) == 0:
                self._set("center", Position())
            else:
                avg_center = sum(
                    (child.center for child in children), Position()) / len(children)
                self._set("center", avg_center)

        return super().__post_init__()

    def move(self, offset: Position) -> None:
        for child in self.children:
            child.move(offset=offset)

    def _draw_planform(self, ax: Axes):
        for child in self.children:
            child.copy(
                f"{self.component_name} {child.component_name}"
            )._draw_planform(ax)

    def _draw_side_elevation(self, ax: Axes):
        for child in self.children:
            child.copy(
                f"{self.component_name} {child.component_name}"
            )._draw_side_elevation(ax)
