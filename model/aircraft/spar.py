from __future__ import annotations

from matplotlib.axes import Axes

from model.util.output import *
from model.core.airfoil import Airfoil, AirfoilManager
from model.aircraft.material import Material
from model.util.dataclass import FrozenDataclassBase
from model.aircraft.base import Component, Mass, Position
from model.aircraft.plot import _plot

from scipy.integrate import cumulative_trapezoid
from typing import overload, TypeVar, Self
from dataclasses import dataclass, field
from scipy.optimize import minimize
from functools import lru_cache
import numpy as np
import math

from model.util.output import Axes


@dataclass(frozen=True)
class BaseSpar(Component):
    component_name: str = field(init=False, default="Spar")

    material: Material
    """spar material"""

    external_height: float
    """external spar height [m]"""
    external_width: float
    """external spar width [m]"""

    length: float
    """spar length [m]"""

    top_thickness: float
    """top thickness [m]"""
    bottom_thickness: float
    """bottom thickness [m]"""
    left_thickness: float
    """left side thickness [m]"""
    right_thickness: float
    """right side thickness [m]"""

    def __post_init__(self):
        if self.mass.override_mass is None:
            self._set("mass", Mass(override_mass=self.estimated_mass))

        return super().__post_init__()

    @classmethod
    def from_simple_spar(cls, material: Material, external_height: float, external_width: float, length: float, thickness: float, center: Position, *, mass: Mass = Mass()) -> Self:
        """square spar with equal thicknesses"""
        return cls(
            mass=mass,
            center=center,
            material=material,
            external_height=external_height,
            external_width=external_width,
            length=length,
            top_thickness=thickness,
            bottom_thickness=thickness,
            left_thickness=thickness,
            right_thickness=thickness
        )

    @classmethod
    def from_optimisation(
        cls,
        material: Material,
        external_height: float,
        length: float,
        top_thickness: float,
        bottom_thickness: float,
        left_thickness: float,
        right_thickness: float,
        load_distribution: WingLoadingDistribution,
        load: float,
        min_factor_of_safety: float,
        max_deflection: float,
        max_deflection_angle: float,
        *,
        mass: Mass = Mass()
    ) -> Self:

        @lru_cache(maxsize=None)
        def get_spar(width) -> Self:
            return cls(
                mass=mass,
                center=Position(),
                material=material,
                external_height=external_height,
                external_width=width,
                length=length,
                top_thickness=top_thickness,
                bottom_thickness=bottom_thickness,
                left_thickness=left_thickness,
                right_thickness=right_thickness
            )

        def get_factor_of_safety(width) -> float:
            return get_spar(width).run_load_model(
                distribution=load_distribution,
                load=load
            ).root_factor_of_safety

        def constraint_factor_of_safety(params):
            width, = params
            return get_factor_of_safety(width) - min_factor_of_safety

        def get_tip_deflection(width) -> float:
            return get_spar(width).run_load_model(
                distribution=load_distribution,
                load=load
            ).tip_deflection

        def constraint_tip_deflection(params):
            width, = params
            return max_deflection - get_tip_deflection(width)

        def get_tip_deflection_angle(width) -> float:
            return get_spar(width).run_load_model(
                distribution=load_distribution,
                load=load
            ).tip_deflection_angle

        def constraint_tip_deflection_angle(params) -> float:
            width, = params
            return max_deflection_angle - get_tip_deflection_angle(width)

        def obj(params):
            width, = params
            return width

        result = minimize(
            obj,
            x0=[external_height],
            bounds=[(left_thickness + right_thickness, None)],
            constraints=[
                {
                    "type": "ineq",
                    "fun": constraint_factor_of_safety,
                },
                {
                    "type": "ineq",
                    "fun": constraint_tip_deflection
                },
                {
                    "type": "ineq",
                    "fun": constraint_tip_deflection_angle
                }
            ],
            method="SLSQP",
            tol=1e-3,
        )

        # width = result.x[0]
        # print(
        #     f"height: {external_height}m, "
        #     f"width: {width}m, "
        #     f"fos: {get_factor_of_safety(width)}, "
        #     f"deflection: {get_tip_deflection(width)}, "
        #     f"deflection angle: {get_tip_deflection_angle(width)}"
        # )

        return get_spar(result.x[0])

    def update_position(self, center: Position) -> None:
        """update center position"""
        self._set("center", center)

    @property
    def name(self) -> str:
        return f"{self.external_width}x{self.external_height}m"

    @property
    def inner_width(self) -> float:
        """inner width [m]"""
        return self.external_width - 2 * self.left_thickness - self.right_thickness

    @property
    def inner_height(self) -> float:
        """inner height [m]"""
        return self.external_height - self.top_thickness - self.bottom_thickness

    @property
    def median_width(self) -> float:
        """approximately the median width [m]

        Width from center of walls"""
        return (self.external_width + self.inner_width) / 2

    @property
    def median_height(self) -> float:
        """approximately the median height [m]

        Height from center of walls
        """
        return (self.external_height + self.inner_height) / 2

    @property
    def median_area(self) -> float:
        """area within median walls (includes empty space within spar) [m^2]"""
        return self.median_height * self.median_width

    @property
    def structure_cross_sectional_area(self) -> float:
        """cross-sectional area [m^2] of the spar structure.

        Not including empty space"""
        return self.external_height * self.external_width - self.inner_height * self.inner_width

    @property
    def structure_volume(self) -> float:
        """volume [m^3] of the structure of the wing"""
        return self.structure_cross_sectional_area * self.length

    @property
    def estimated_mass(self) -> float:
        """structure mass [kg] not including adhesives or fixings"""
        return self.material.density * self.structure_volume

    @property
    def second_moment_of_area(self) -> float:
        """second moment of area, I [m^4]

        measure of bending resistance along the spar using thin wall approximations"""
        b = self.external_width
        h = self.external_height
        hi = self.inner_height

        tl = self.left_thickness
        tr = self.right_thickness
        tt = self.top_thickness
        tb = self.bottom_thickness

        # areas of the four walls
        A_top = b * tt
        A_bottom = b * tb
        A_left = tl * hi
        A_right = tr * hi

        # centroid locations measured from the bottom
        y_top = h - tt / 2
        y_bottom = tb / 2
        y_left = tb + hi / 2
        y_right = tb + hi / 2

        # composite centroid
        y_bar = (
            A_top * y_top
            + A_bottom * y_bottom
            + A_left * y_left
            + A_right * y_right
        ) / (A_top + A_bottom + A_left + A_right)

        I = (
            b * tt**3 / 12 + A_top * (y_top - y_bar)**2
            + b * tb**3 / 12 + A_bottom * (y_bottom - y_bar)**2
            + tl * hi**3 / 12 + A_left * (y_left - y_bar)**2
            + tr * hi**3 / 12 + A_right * (y_right - y_bar)**2
        )

        return I

    @property
    def polar_moment_of_area(self) -> float:
        """polar moment of area, J [m^4]

        measure of torsional resistance along the spar using thin wall approximations"""
        tt = self.top_thickness
        tb = self.bottom_thickness
        tl = self.left_thickness
        tr = self.right_thickness

        Am = self.median_area
        bm = self.median_width
        hm = self.median_height

        J = (4 * Am**2) / (
            bm / tt
            + bm / tb
            + hm / tl
            + hm / tr
        )

        return J

    @property
    def torsional_compliance(self) -> float:
        """measure of amounts of twist per applied torque [degrees/Nm]"""
        return math.degrees(self.length / (self.material.shear_modulus * self.polar_moment_of_area))

    @overload
    def run_load_model(
        self,
        distribution: WingLoadingDistribution,
        *,
        load: float,
        n: int = 100
    ) -> "SparLoadModel":
        ...

    @overload
    def run_load_model(
        self,
        distribution: WingLoadingDistribution,
        *,
        linear_loading: float,
        n: int = 100
    ) -> "SparLoadModel":
        ...

    def run_load_model(
        self,
        distribution: WingLoadingDistribution,
        *,
        load: float | None = None,
        linear_loading: float | None = None,
        n: int = 100
    ) -> "SparLoadModel":
        """create a load model based on your spar and load or loading

        Args:
            distribution: class that implements `WingLoadingDistribution`, such as `UniformWingLoading` or `EllipticalWingLoading`
            load: load on the entire spar [N]. If the load is distributed over two spars, for example one on either side of a fuselage half your load value. Defaults to None.
            linear_loading: spar loading [N/m]. Please be aware if not using a `UniformLoadingDistribution` the actual load will differ from this load due to load distribution. Defaults to None.
            n: number of spanwise locations to calculate. Defaults to 100.

        Raises:
            ValueError: Neither load or loading provided

        Returns:
            `SparLoadModel` objects with data output and plotting functions
        """
        if linear_loading is not None:
            load = linear_loading * self.length
        elif load is None:
            raise ValueError(f"either load or loading must be provided")

        return SparLoadModel(
            spar=self,
            load=load,
            distribution=distribution,
            n=n
        )


@dataclass(frozen=True)
class LateralSpar(BaseSpar):

    def _draw_planform(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.y,
            height=self.length,
            width=self.external_width,
            label=self.component_name
        )

    def _draw_side_elevation(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.z,
            height=self.external_height,
            width=self.external_width,
            label=self.component_name
        )


@dataclass(frozen=True)
class LongitudinalSpar(BaseSpar):

    def _draw_planform(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.y,
            height=self.external_width,
            width=self.length,
            label=self.component_name
        )

    def _draw_side_elevation(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.z,
            height=self.external_height,
            width=self.length,
            label=self.component_name
        )


@dataclass(frozen=True)
class VerticalSpar(BaseSpar):

    def _draw_planform(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.y,
            height=self.external_height,
            width=self.external_width,
            label=self.component_name
        )

    def _draw_side_elevation(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.z,
            height=self.length,
            width=self.external_height,
            label=self.component_name
        )


T = TypeVar("T", float, np.ndarray)

# TODO: Extend with lifting-line theory to estimate for whole wing?


class WingLoadingDistribution:
    def spanwise_normalised_load(self, xs: T) -> T:
        """normalised spanwise load where x/s is the distance from the wing root

        w/s = 0 is root
        w/s = 1 is tip"""

        # TODO: Support numpy too
        raise NotImplementedError()

    def spanwise_load(self, xs: T, load: float, span: float) -> T:
        """spanwise load where x/s is the distance from the wing root

        w/s = 0 is root
        w/s = 1 is tip"""
        return load / span * self.spanwise_normalised_load(xs)

    @property
    def name(self) -> str:
        return self.__class__.__name__


class UniformWingLoading(WingLoadingDistribution):
    def spanwise_normalised_load(self, xs: T) -> T:

        if isinstance(xs, float):
            return 1.0

        return np.ones_like(xs)


class EllipticalWingLoading(WingLoadingDistribution):
    def spanwise_normalised_load(self, xs: T) -> T:

        w = 4 / np.pi * np.sqrt(1 - xs**2)

        if isinstance(xs, float):
            return float(w)

        return w


@dataclass(frozen=True)
class SparLoadModel(FrozenDataclassBase):
    category = "spar"

    spar: BaseSpar
    load: float
    """total load on this spar [N].
    
    If the load is distributed over two spars, for example one on either side of a fuselage half your load value"""
    distribution: WingLoadingDistribution = UniformWingLoading()
    """wing loading distribution model to use"""
    n: int = 100
    """number of spanwise locations"""

    xs: np.ndarray = field(init=False)
    """x/s position"""
    x: np.ndarray = field(init=False)
    """span position [m]"""
    w: np.ndarray = field(init=False)
    """load [N]"""
    V: np.ndarray = field(init=False)
    """shear force [N]"""
    M: np.ndarray = field(init=False)
    """bending moment [Nm]"""
    y: np.ndarray = field(init=False)
    """deflection [m]"""
    deflection_angle: np.ndarray = field(init=False)
    """deflection angle [deg]"""
    surface_stress: np.ndarray = field(init=False)
    """surface stress [Pa] on top/button of spars"""
    FoS: np.ndarray = field(init=False)
    """factor of safety between surface stress and ultimate stress"""

    plots: "SparLoadPlots" = field(init=False)

    def __post_init__(self):
        self._set("plots", SparLoadPlots(self))

        self._set("xs", np.linspace(0, 1, self.n))
        self._set("x", self.xs * self.spar.length)
        self._set("w", self.distribution.spanwise_load(
            self.xs, self.load, self.spar.length))
        # integrate w wrt x to get V
        self._set(
            "V",
            -cumulative_trapezoid(
                self.w[::-1],
                self.x[::-1],
                initial=0,
            )[::-1]
        )
        # integrate V wrt x to get M
        self._set(
            "M",
            -cumulative_trapezoid(
                self.V[::-1],
                self.x[::-1],
                initial=0,
            )[::-1]
        )

        # deflection using euler-bernoulli beam theory and treating the spar as a cantilever beam - https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory
        # calculate curvature from bending moment and then integrate twice to get deflection
        # valid only for small deflections
        E = self.spar.material.youngs_modulus
        I = self.spar.second_moment_of_area
        M = self.M

        curvature = M / (E * I)
        slope = cumulative_trapezoid(curvature, self.x, initial=0)
        y = cumulative_trapezoid(slope, self.x, initial=0)

        self._set("y", y)

        self._set("deflection_angle", np.rad2deg(slope))

        # estimated surface stress
        self._set(
            "surface_stress",
            self.M * (self.spar.external_height / 2) /
            self.spar.second_moment_of_area
        )

        # factor of safety
        self._set(
            "FoS",
            self.spar.material.ultimate_strength / self.surface_stress
        )

    @property
    def distribution_name(self) -> str:
        return self.distribution.name

    @property
    def name(self) -> str:
        return f"{self.distribution_name.rstrip("WingLoading")}"

    @property
    def root_moment(self) -> float:
        """bending moment at the root [Nm]"""
        return self.M[0]

    @property
    def root_surface_stress(self) -> float:
        """root surface stress [Pa] on top/button of spars"""
        return self.surface_stress[0]

    @property
    def root_factor_of_safety(self) -> float:
        return self.FoS[0]

    @property
    def tip_deflection(self) -> float:
        return self.y[-1]

    @property
    def tip_deflection_angle(self) -> float:
        return self.deflection_angle[-1]


class SparLoadPlots(PlotBase[SparLoadModel]):  # type: ignore

    @scientific_plot(auto_limits="x")
    def load(self, ax: Axes):
        model = self.model

        ax.plot(model.x, model.w)

        ax.set_xlabel("Spanwise distance from wing root [m]")
        ax.set_ylabel("Load [N]")

    @scientific_plot(auto_limits="x")
    def shear_force(self, ax: Axes):
        model = self.model

        ax.plot(model.x, model.V)

        ax.set_xlabel("Spanwise distance from wing root [m]")
        ax.set_ylabel("Shear Force [N]")

    @scientific_plot(auto_limits="x")
    def bending_moment(self, ax: Axes):
        model = self.model

        ax.plot(model.x, model.M)

        ax.set_xlabel("Spanwise distance from wing root [m]")
        ax.set_ylabel("Bending Moment [Nm]")

    @scientific_plot(auto_limits="x")
    def deflection(self, ax: Axes):
        model = self.model

        ax.plot(model.x, model.y)

        ax.set_xlabel("Spanwise distance from wing root [m]")
        ax.set_ylabel("Deflection [m]")

    @scientific_plot(auto_limits="x")
    def deflection_angle(self, ax: Axes):
        model = self.model

        ax.plot(model.x, model.deflection_angle)

        ax.set_xlabel("Spanwise distance from wing root [m]")
        ax.set_ylabel("Deflection angle [deg]")

    @scientific_plot(auto_limits="x")
    def surface_stress(self, ax: Axes):
        model = self.model

        ax.plot(model.x, model.surface_stress / 1e6)

        ax.set_xlabel("Spanwise distance from wing root [m]")
        ax.set_ylabel("Surface Stress [MPa]")

    @scientific_plot(auto_limits="x")
    def factor_of_safety(self, ax: Axes):
        model = self.model

        ax.plot(model.x, model.FoS)

        ax.set_ylim(0, 3)

        ax.set_xlabel("Spanwise distance from wing root [m]")
        ax.set_ylabel("Factor of Safety")

    @scientific_compose()
    def overview(self):
        plots = [
            self.load.with_args(),
            self.shear_force.with_args(),
            self.bending_moment.with_args(),
            self.deflection.with_args(),
            self.deflection_angle.with_args(),
            self.surface_stress.with_args(),
            self.factor_of_safety.with_args()
        ]

        return Composition(
            plots=plots,
            sharex=True,
            figsize=(6, len(plots) * 1.5),
            outside_legend="top-inside"
        )
