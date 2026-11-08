from __future__ import annotations
from typing import Any, overload, Literal, Self
from scipy.optimize import least_squares
from dataclasses import field, dataclass
from functools import lru_cache
import pandas as pd
import numpy as np
import math

from model.aircraft.base import Component, Mass, Position, assumptions, ComponentInfo, ComponentPlots
from model.core.airfoil import Airfoil, AirfoilManager
from model.aircraft.material import Material
from model.aircraft.plot import _plot
from model.aircraft.spar import BaseSpar, LongitudinalSpar, LateralSpar
from model.util.output import *
from model.util.dataclass import FrozenDataclassBase


@dataclass(frozen=True)
class Airfoil3D(Component):
    """Represents half a 3D Wing (with length semi-span)"""
    component_name: str = field(init=False, default="3D Airfoil")
    plots: Airfoil3DPlots = field(init=False, repr=False)
    info: Airfoil3DInfo = field(init=False, repr=False)

    skin_material: Material
    """airfoil skin material"""
    rib_material: Material
    """airfoil rib material"""

    semi_span: float
    """airfoil length (or span) [m]"""
    chord: float
    """chord length [m]"""
    setting_angle: float
    """wing setting angle [deg]"""

    e: float
    """span/oswald efficiency factor"""

    airfoil2d: Airfoil
    """2d airfoil to use"""

    skin_thickness: float | None = None
    """airfoil skin thickness [m] to estimate mass"""
    rib_thickness: float | None = None
    """airfoil rib thickness [m] to estimate mass"""
    rib_spacing: float | None = None
    """spacing between ribs [m] used to estimate mass"""
    overlapping_spar: BaseSpar | None = None
    """option to estimate the mass when the spar overlaps the airfoil skin"""

    default_solver: solvers = "Lifting Line Theory"
    solvers: Solvers = field(init=False, repr=False)

    def __post_init__(self):
        if self.mass.override_mass is None:
            object.__setattr__(self, "mass", Mass(
                override_mass=self.estimated_mass,
                override_cg=self.chordwise_position(0.4)
            ))

        super().__post_init__()
        self._set("plots", Airfoil3DPlots(self))
        self._set("info", Airfoil3DInfo(self))

        self._set("solvers", Solvers(
            airfoil2d=self.airfoil2d,
            semi_span=self.semi_span,
            chord=self.chord,
            e=self.e,
            default=self.default_solver
        ))

    @classmethod
    def from_leading_edge(cls, skin_material: Material, rib_material: Material, length: float, chord: float, setting_angle: float, airfoil2d: Airfoil, oswald_efficiency_factor: float, leading_edge: Position, *, skin_thickness: float | None = None, rib_spacing: float | None = None, overlapping_spar: BaseSpar | None = None, rib_thickness: float | None = None, mass: Mass = Mass()) -> Self:
        center = cls.center_from_leading_edge(
            chord, setting_angle, leading_edge)

        return cls(
            mass=mass,
            center=center,
            skin_material=skin_material,
            rib_material=rib_material,
            semi_span=length,
            airfoil2d=airfoil2d,
            setting_angle=setting_angle,
            chord=chord,
            e=oswald_efficiency_factor,
            skin_thickness=skin_thickness,
            rib_spacing=rib_spacing,
            overlapping_spar=overlapping_spar,
            rib_thickness=rib_thickness,
        )

    @classmethod
    def from_aerodynamic_center(cls, skin_material: Material, rib_material: Material, length: float, chord: float, setting_angle: float, airfoil2d: Airfoil, oswald_efficiency_factor: float, leading_edge: Position, *, skin_thickness: float | None = None, rib_spacing: float | None = None, overlapping_spar: BaseSpar | None = None, rib_thickness: float | None = None, mass: Mass = Mass()) -> Self:
        center = cls.center_from_aerodynamic_center(
            chord, setting_angle, leading_edge)

        return cls(
            mass=mass,
            center=center,
            skin_material=skin_material,
            rib_material=rib_material,
            semi_span=length,
            airfoil2d=airfoil2d,
            setting_angle=setting_angle,
            chord=chord,
            e=oswald_efficiency_factor,
            skin_thickness=skin_thickness,
            rib_spacing=rib_spacing,
            overlapping_spar=overlapping_spar,
            rib_thickness=rib_thickness,
        )

    @classmethod
    def from_NACA(cls, skin_material: Material, rib_material: Material, length: float, chord: float, leading_edge: Position, airfoil_name: str, oswald_efficiency_factor: float, setting_angle: float = 0, *, mass: Mass = Mass(), skin_thickness: float | None = None, rib_spacing: float | None = None, overlapping_spar: BaseSpar | None = None, rib_thickness: float | None = None) -> Airfoil3D:
        airfoil2d = AirfoilManager().get_airfoil(
            airfoil_name=airfoil_name,
            Re=assumptions.reynolds_number,
            Mach=assumptions.mach_number,
            alpha_min=-10,
            alpha_max=90
        )

        return Airfoil3D.from_leading_edge(
            skin_material=skin_material,
            rib_material=rib_material,
            length=length,
            chord=chord,
            oswald_efficiency_factor=oswald_efficiency_factor,
            setting_angle=setting_angle,
            airfoil2d=airfoil2d,
            leading_edge=leading_edge,
            skin_thickness=skin_thickness,
            rib_spacing=rib_spacing,
            mass=mass,
            overlapping_spar=overlapping_spar,
            rib_thickness=rib_thickness,
        )

    @property
    def name(self) -> str:
        return f"{self.airfoil2d.name}-{self.component_name}"

    @classmethod
    def _chordwise_position(cls, x: float, chord: float, setting_angle: float, center: Position) -> Position:
        raise NotImplementedError()

    @classmethod
    def center_from_leading_edge(cls, chord: float, setting_angle: float, leading_edge: Position) -> Position:
        return cls._chordwise_position(x=1, chord=chord, setting_angle=setting_angle, center=leading_edge)

    @classmethod
    def center_from_aerodynamic_center(cls, chord: float, setting_angle: float, aerodynamic_center: Position) -> Position:
        return cls._chordwise_position(x=0.75, chord=chord, setting_angle=setting_angle, center=aerodynamic_center)

    def chordwise_position(self, x: float) -> Position:
        return self._chordwise_position(x=x, chord=self.chord, setting_angle=self.setting_angle, center=self.center)

    @property
    def leading_edge(self) -> Position:
        return self.chordwise_position(0)

    @property
    def trailing_edge(self) -> Position:
        return self.chordwise_position(1)

    @property
    def aerodynamic_center(self) -> Position:
        return self.chordwise_position(0.25)

    @property
    def estimated_mass(self) -> float:
        """estimate the mass of the airfoil [kg] based on skin thickness and rib spacing"""
        if self.rib_spacing is None or self.skin_thickness is None or self.rib_thickness is None:
            raise ValueError(
                f"rib spacing (provided: {self.rib_spacing is not None}), skin thickness (provided: {self.skin_thickness is not None}) and rib thickness (provided: {self.rib_thickness is not None}) are required to estimate mass")

        perimeter = self.airfoil2d.airfoil_perimeter_length(self.chord)
        area = self.airfoil2d.airfoil_cross_sectional_area(self.chord)

        if self.overlapping_spar is not None:
            spar = self.overlapping_spar

            area = area - spar.external_width * self.skin_thickness
            perimeter = perimeter - 2 * spar.external_width

        n_ribs = self.semi_span // self.rib_spacing + 1

        skin_area = perimeter * self.skin_thickness
        rib_area = area - skin_area

        skin_volume = skin_area * self.semi_span
        rib_volume = n_ribs * rib_area * self.rib_thickness

        skin_mass = skin_volume * self.skin_material.density
        rib_mass = rib_volume * self.rib_material.density

        # print(
        #     f"skin mass: {skin_mass*1e3:.1f}g, "
        #     f"rib mass: {rib_mass*1e3:.1f}g"
        # )

        return skin_mass + rib_mass

    @classmethod
    def from_smallest_fitting_spar(
        cls,
        camber: float,
        camber_location: float,
        chord: float,
        setting_angle: float,
        spar: BaseSpar,
        padding: float,
        *,
        tmax: int = 30
    ) -> tuple[Airfoil, float, float]:
        """smallest fitting airfoil finds the smallest NACA 4-series airfoil with the given camber and camber_location that will fit this wing spar. The chord and setting angle and padding around the spar are also considered.

        The spar is fitted within the airfoil in the x, y plane without rotation whilst the airfoil itself may be rotated. As such teh spar with always be normal with the axis.

        This algorithm works by starting with an airfoil with the same thickness as the chord, checking if a square defined by each of its 4 vertices fits within the aerofoil coordinates fits where the first corner is posited along the upper edge of the spar. Successive airfoils are tried in 1% thickness increments until either the maximum thickness is reached or a spar fits.

        This algorithm does not guarantee the best position or the thinnest possible spar but should be pretty close.

        This function depends on xFoil where cached data does not exist.

        Args:
            camber: maximum camber as a fraction of chord (i.e. 0.1 = 10% of chord).
            camber_location: location of maximum camber as a fraction of chord (i.e. 0.5 is the middle of the chord).
            padding: padding around all sides of the spar [m]
            chord: airfoil chord length [m]
            setting_angle: airfoil setting angle [deg]
            tmax: Maximum thickness to try. Defaults to 30.

        Raises:
            ValueError: Minimum thickness from provided geometry is greater than maximum thickness. Consider changing geometry.
            ValueError: No solution found

        Returns:
            airfoil: Airfoil object containing the smallest fitted airfoil
            center_x: x position of the center of the spar within airfoil reference frame (0, 0) is at the leading edge
            center_y: y position of the center of the spar within airfoil reference frame (0, 0) is at the leading edge
        """
        H = spar.external_height + padding * 2
        W = spar.external_width + padding * 2

        # initial thickness guess
        t0 = H/chord
        t0 = math.floor(t0 * 100)
        t0 = max(1, t0)

        if t0 > tmax:
            raise ValueError(
                f"Minimum thickness {t0:.2f}% is greater than {tmax:.2f}%")

        for t in np.arange(t0, tmax + 1, 1):
            t = t / 100

            # get airfoil from xFoil
            airfoil = AirfoilManager().get_airfoil(
                camber=camber,
                camber_location=camber_location,
                thickness=t,
                Re=assumptions.reynolds_number,
                Mach=assumptions.mach_number,
                alpha_min=-10,
                alpha_max=90
            )

            # get coordinates
            xs, ys = airfoil.scale_DAT(chord=chord, angle=setting_angle)

            # split into upper and lower
            origin = np.argmin(xs) + 1
            upper_x = xs[:origin][::-1]
            upper_y = ys[:origin][::-1]
            lower_x = xs[origin:]
            lower_y = ys[origin:]

            # construct search func
            xmin = np.min(xs)
            xmax = np.max(xs)
            ymin = np.min(ys)
            ymax = np.max(ys)

            def within_airfoil(x: float, y: float) -> bool:
                if ymin > y or ymax < y:
                    return False

                if xmin > x or xmax < x:
                    return False

                y_up = np.interp(x, upper_x, upper_y)
                if y_up < y:
                    return False

                y_down = np.interp(x, lower_x, lower_y)
                if y_down > y:
                    return False

                return True

            # get end of search point
            max_idx = np.argmax(upper_y)

            # search starting a top left corner
            fwd_xs = upper_x[:max_idx]
            fwd_ys = upper_y[:max_idx]

            for fwd_x, fwd_y in zip(fwd_xs, fwd_ys):
                # check top right corner
                if not within_airfoil(fwd_x + W, fwd_y):
                    continue

                # check bottom right corner
                if not within_airfoil(fwd_x + W, fwd_y - H):
                    continue

                # check bottom left corner
                if not within_airfoil(fwd_x, fwd_y - H):
                    continue

                center_x = fwd_x + W/2
                center_y = fwd_y - H/2
                return airfoil, center_x, center_y

            # search start at top right corner
            bwd_xs = upper_x[max_idx:]
            bwd_ys = upper_y[max_idx:]

            for bwd_x, bwd_y in zip(bwd_xs, bwd_ys):
                # check bottom right corner
                if not within_airfoil(bwd_x, bwd_y - H):
                    continue

                # check bottom left corner
                if not within_airfoil(bwd_x - W, bwd_y - H):
                    continue

                # check top left corner
                if not within_airfoil(bwd_x - W, bwd_y):
                    continue

                center_x = bwd_x - W/2
                center_y = bwd_y - H/2
                return airfoil, center_x, center_y

        raise ValueError(f"No solution found")

    @property
    def wing_thickness(self) -> float:
        return self.airfoil2d.thickness * self.chord


@dataclass(frozen=True)
class LateralAirfoil3D(Airfoil3D):

    @classmethod
    def _chordwise_position(cls, x: float, chord: float, setting_angle: float, center: Position) -> Position:
        if x < 0 or x > 1:
            raise ValueError(f"x should be in range 0 to 1, {x} provided")
        setting_angle = math.radians(setting_angle)
        return center + Position(
            x=-chord * (x - 0.5) * math.cos(setting_angle),
            y=0,
            z=-chord * (x - 0.5) * math.sin(setting_angle),
        )

    def _draw_planform(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.y,
            width=self.leading_edge.x - self.trailing_edge.x,
            height=self.semi_span,
            label=self.component_name
        )

    def _draw_side_elevation(self, ax: Axes):
        _plot.plot_airfoil(
            ax=ax,
            airfoil=self.airfoil2d,
            chord=self.chord,
            setting_angle=self.setting_angle,
            label=self.component_name,
            leading_edge_x=self.leading_edge.x,
            leading_edge_y=self.leading_edge.z,
            facing="right"
        )


@dataclass(frozen=True)
class VerticalAirfoil3D(Airfoil3D):

    @classmethod
    def _chordwise_position(cls, x: float, chord: float, setting_angle: float, center: Position) -> Position:
        if x < 0 or x > 1:
            raise ValueError(f"x should be in range 0 to 1, {x} provided")
        setting_angle = math.radians(setting_angle)
        return center + Position(
            x=-chord * (x - 0.5) * math.cos(setting_angle),
            y=chord * (x - 0.5) * math.sin(setting_angle),
            z=0,
        )

    def _draw_planform(self, ax: Axes):
        _plot.plot_airfoil(
            ax=ax,
            airfoil=self.airfoil2d,
            chord=self.chord,
            setting_angle=self.setting_angle,
            label=self.component_name,
            leading_edge_x=self.leading_edge.x,
            leading_edge_y=self.leading_edge.y,
            facing="right"
        )

    def _draw_side_elevation(self, ax: Axes):
        _plot.plot_box(
            ax=ax,
            center_x=self.center.x,
            center_y=self.center.z,
            width=self.leading_edge.x - self.trailing_edge.x,
            height=self.semi_span,
            label=self.component_name
        )


class Airfoil3DPlots(ComponentPlots[Airfoil3D]):

    @scientific_plot(auto_limits=None)
    def CL(self, ax: Axes):
        model = self.model

        min_alpha = model.airfoil2d.min_alpha
        max_alpha = model.airfoil2d.max_alpha
        alpha = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(alpha, model.solvers.ideal_finite_wing.CL(
            alpha), label="Ideal Finite Wing")
        ax.plot(alpha, model.solvers.finite_wing.CL(
            alpha), label=f"Finite Wing (e={model.solvers.finite_wing.e})")
        ax.plot(alpha, model.solvers.lifting_line_theory.CL(
            alpha), label="Lifting Line Theory")
        ax.plot(alpha, model.airfoil2d.CL(alpha), label="xFoil (2D)")

        ax.legend()
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Lift Coefficient")

    @scientific_plot(auto_limits=None)
    def CD(self, ax: Axes):
        model = self.model

        min_alpha = model.airfoil2d.min_alpha
        max_alpha = model.airfoil2d.max_alpha
        alpha = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(alpha, model.solvers.ideal_finite_wing.CD(
            alpha), label="Ideal Finite Wing")
        ax.plot(alpha, model.solvers.finite_wing.CD(
            alpha), label=f"Finite Wing (e={model.solvers.finite_wing.e})")
        ax.plot(alpha, model.solvers.lifting_line_theory.CD(
            alpha), label="Lifting Line Theory")
        ax.plot(alpha, model.airfoil2d.CD(alpha), label="xFoil (2D)")

        ax.legend()
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Drag Coefficient")

    @scientific_plot(auto_limits=None)
    def CM(self, ax: Axes):
        model = self.model

        min_alpha = model.airfoil2d.min_alpha
        max_alpha = model.airfoil2d.max_alpha
        alpha = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(alpha, model.solvers.ideal_finite_wing.CM(
            alpha), label="Ideal Finite Wing")
        ax.plot(alpha, model.solvers.finite_wing.CM(
            alpha), label=f"Finite Wing (e={model.solvers.finite_wing.e})")
        ax.plot(alpha, model.solvers.lifting_line_theory.CM(
            alpha), label="Lifting Line Theory")
        ax.plot(alpha, model.airfoil2d.CM(alpha), label="xFoil (2D)")

        ax.legend()
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Moment Coefficient (about x/c=0.25)")

    @scientific_plot(auto_limits=None)
    def LD(self, ax: Axes):
        model = self.model

        min_alpha = model.airfoil2d.min_alpha
        max_alpha = model.airfoil2d.max_alpha
        alpha = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(alpha, model.solvers.ideal_finite_wing.LD(
            alpha), label="Ideal Finite Wing")
        ax.plot(alpha, model.solvers.finite_wing.LD(
            alpha), label=f"Finite Wing (e={model.solvers.finite_wing.e})")
        ax.plot(alpha, model.solvers.lifting_line_theory.LD(
            alpha), label="Lifting Line Theory")
        ax.plot(alpha, model.airfoil2d.LD(alpha), label="xFoil (2D)")

        ax.legend()
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Lift to Drag Ratio")

    @scientific_plot(auto_limits=None)
    def drag_polar(self, ax: Axes):
        model = self.model

        min_alpha = model.airfoil2d.min_alpha
        max_alpha = model.airfoil2d.max_alpha
        alpha = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(
            model.solvers.ideal_finite_wing.CL(alpha),
            model.solvers.ideal_finite_wing.CD(alpha),
            label="Ideal Finite Wing"
        )
        ax.plot(
            model.solvers.finite_wing.CL(alpha),
            model.solvers.finite_wing.CD(alpha),
            label=f"Finite Wing (e={model.solvers.finite_wing.e})"
        )
        ax.plot(
            model.solvers.lifting_line_theory.CL(alpha),
            model.solvers.lifting_line_theory.CD(alpha),
            label="Lifting Line Theory"
        )
        ax.plot(
            model.airfoil2d.CL(alpha),
            model.airfoil2d.CD(alpha),
            label="xFoil (2D)"
        )

        ax.legend()
        ax.set_xlabel(r"Lift Coefficient")
        ax.set_ylabel(r"Drag Coefficient")


class Airfoil3DInfo(ComponentInfo[Airfoil3D]):

    @scientific_table()
    def overview(self):
        model = self.model

        chord = model.chord

        return pd.DataFrame({
            "Name": model.name,
            "Airfoil Name": model.airfoil2d.name,
            "Thickness [m]": model.airfoil2d.thickness * chord,
            "Semi-Span (Length) [m]": model.semi_span,
            "Chord [m]": chord,
            "Aspect Ratio": model.solvers.AR,
            "Setting Angle [deg]": model.setting_angle,
            "Estimated Mass [kg]": model.estimated_mass if model.mass.override_mass is None else "NA",
            "Working mass [kg]": model.mass.mass,
            "spanwise efficiency factor, e": model.solvers.lifting_line_theory.e
        }, index=[0])


solvers = Literal["Finite Wing", "Ideal Finite Wing", "Lifting Line Theory"]


class Solvers:
    def __init__(
        self,
        airfoil2d: Airfoil,
        semi_span: float,
        chord: float,
        e: float,
        *,
        default: solvers = "Lifting Line Theory"
    ) -> None:
        self.default_solver = default

        self.ideal_finite_wing = FiniteWingMethod(
            airfoil2d=airfoil2d,
            semi_span=semi_span,
            chord=chord,
            e=1
        )

        self.finite_wing = FiniteWingMethod(
            airfoil2d=airfoil2d,
            semi_span=semi_span,
            chord=chord,
            e=e
        )

        self.lifting_line_theory = LiftingLineTheorySolver(
            airfoil2d=airfoil2d,
            semi_span=semi_span,
            chord=chord,
        )

    @property
    def solvers(self) -> list[BaseSolver]:
        return [
            self.ideal_finite_wing,
            self.finite_wing,
            self.lifting_line_theory
        ]

    @property
    def solver(self) -> BaseSolver:
        return self.__getattribute__(
            self.default_solver.replace(" ", "_").lower()
        )

    @property
    def span(self) -> float:
        """span [m] of the equivalent whole wing"""
        return self.solver.span

    @property
    def S_semi(self) -> float:
        """Semi/Half-wing area [m^2]"""
        return self.solver.S_semi

    @property
    def S_full(self) -> float:
        """full wing area [m^2]"""
        return self.solver.S_full

    @property
    def AR(self) -> float:
        """aspect ratio"""
        return self.solver.AR

    @overload
    def CL(self, alpha: float) -> float: ...

    @overload
    def CL(self, alpha: np.ndarray) -> np.ndarray: ...

    def CL(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.solver.CL(alpha)

    @overload
    def CD(self, alpha: float) -> float: ...

    @overload
    def CD(self, alpha: np.ndarray) -> np.ndarray: ...

    def CD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.solver.CD(alpha)

    @overload
    def CM(self, alpha: float) -> float: ...

    @overload
    def CM(self, alpha: np.ndarray) -> np.ndarray: ...

    def CM(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.solver.CM(alpha)

    @overload
    def LD(self, alpha: float) -> float: ...

    @overload
    def LD(self, alpha: np.ndarray) -> np.ndarray: ...

    def LD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.solver.LD(alpha)


@dataclass(frozen=True, eq=False)
class BaseSolver(FrozenDataclassBase):
    airfoil2d: Airfoil = field(hash=False, compare=False)
    """airfoil profile (including xFoil geometry and polar data)"""
    semi_span: float
    """semi span (half wing) length [m]"""
    chord: float
    """chord length [m]"""
    e: float
    """oswald/span efficiency factor"""

    @property
    def span(self) -> float:
        """span [m] of the equivalent whole wing"""
        return 2.0 * self.semi_span

    @property
    def S_semi(self) -> float:
        """Semi/Half-wing area [m^2]"""
        return self.semi_span * self.chord

    @property
    def S_full(self) -> float:
        """full wing area [m^2]"""
        return self.S_semi * 2.0

    @property
    def AR(self) -> float:
        """aspect ratio"""
        return self.span / self.chord

    @overload
    def CL(self, alpha: float) -> float: ...

    @overload
    def CL(self, alpha: np.ndarray) -> np.ndarray: ...

    def CL(self, alpha: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError()

    @overload
    def CD(self, alpha: float) -> float: ...

    @overload
    def CD(self, alpha: np.ndarray) -> np.ndarray: ...

    def CD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError()

    @overload
    def CM(self, alpha: float) -> float: ...

    @overload
    def CM(self, alpha: np.ndarray) -> np.ndarray: ...

    def CM(self, alpha: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError()

    @overload
    def LD(self, alpha: float) -> float: ...

    @overload
    def LD(self, alpha: np.ndarray) -> np.ndarray: ...

    def LD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.CL(alpha) / self.CD(alpha)


@dataclass(frozen=True)
class FiniteWingMethod(BaseSolver):

    @property
    def k(self) -> float:
        """induced drag factor, k"""
        return 1 / (math.pi * self.e * self.AR)

    @property
    def a(self) -> float:
        """3D lift curve slope [1/rad]"""
        return self.airfoil2d.a / (1 + (self.airfoil2d.a * self.k))

    @overload
    def CL(self, alpha: float) -> float: ...

    @overload
    def CL(self, alpha: np.ndarray) -> np.ndarray: ...

    def CL(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.a * np.radians(alpha - self.airfoil2d.alpha_L0)

    @overload
    def CD(self, alpha: float) -> float: ...

    @overload
    def CD(self, alpha: np.ndarray) -> np.ndarray: ...

    def CD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.airfoil2d.CD_0 + self.k * self.CL(alpha)**2

    @overload
    def CM(self, alpha: float) -> float: ...

    @overload
    def CM(self, alpha: np.ndarray) -> np.ndarray: ...

    def CM(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """momement coefficient relative to the quarter chord point"""
        return self.airfoil2d.CM(alpha)


@dataclass(frozen=True)
class LiftingLineTheorySolver(BaseSolver):
    """Non-linear lifting-line solver for symmetric rectangular straight untwisted wing

    Solves circulation using the full-wing LLT equations, then integrates aerodynamic equations over one semi-span, normalised by the semi-span area.

    Generates half-wing coefficients (equal full-wing for a symmetric wing)"""
    e: float = field(init=False, hash=False, default=0)
    """oswald/span efficiency factor"""

    def __post_init__(self):
        self._set("e", self.__lifting_line_solver(0)["e"])

    @lru_cache(maxsize=100)
    def __lifting_line_solver(self, alpha_deg, N=25):
        try:
            semi_span = self.semi_span
            b = self.span
            chord = c_bar = self.chord
            S_semi = self.S_semi
            S_full = self.S_full
            AR = self.AR

            alpha = np.deg2rad(alpha_deg)

            # full-wing lifting-line collocation points
            n = np.arange(1, N + 1)
            theta = np.arange(1, N + 1) * np.pi / (N + 1)

            # y increases from left wingtip to right wingtip
            y = -semi_span * np.cos(theta)

            # Rectangular wing: constant chord
            c = np.full_like(y, chord, dtype=float)

            sin_matrix = np.sin(np.outer(theta, n))

            downwash_matrix = (
                sin_matrix * n[None, :]
                / np.sin(theta)[:, None]
            )

            # solve nonlinear LLT equations
            def residual(A):
                # Local sectional lift coefficient from circulation.
                cl_gamma = (
                    4.0 * b / c
                    * (sin_matrix @ A)
                )

                # Induced angle of attack [rad].
                alpha_i = downwash_matrix @ A

                # Local effective angle [deg].
                alpha_eff_deg = np.rad2deg(alpha - alpha_i)

                # XFoil sectional lift polar.
                cl_section = self.airfoil2d.CL(alpha_eff_deg)

                return cl_gamma - cl_section

            A0 = np.zeros(N)

            # Estimate initial first Fourier coefficient from the sectional lift at the requested geometric alpha.
            cl0 = self.airfoil2d.CL(alpha_deg)
            A0[0] = cl0 * chord / (4.0 * b)

            solution = least_squares(
                residual,
                A0,
                max_nfev=3000,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11
            )

            if not solution.success:
                raise RuntimeError("LLT solver failed: " + solution.message)

            A = solution.x

            # Evaluate teh solved circulation on one semispan
            # Include the root (theta=pi/2) and tip (theta=pi)
            # so the numerical integration includes both endpoints.
            theta_half = np.linspace(np.pi / 2.0, np.pi, 1001)

            sin_half = np.sin(np.outer(theta_half, n))

            # Circulation normalized by freestream velocity: Gamma / V = 2*b*sum(A_n*sin(n*theta)).
            gamma_over_V = 2.0 * b * (sin_half @ A)

            # Half-span coordinate, from root to tip.
            y_half = -semi_span * np.cos(theta_half)

            c_half = np.full_like(y_half, chord)

            # Induced-angle calculation, including the finite tip limit.
            sin_theta = np.sin(theta_half)
            downwash_half = np.empty_like(sin_half)

            downwash_half[:-1] = (
                sin_half[:-1] * n[None, :]
                / sin_theta[:-1, None]
            )

            # Limit as theta -> pi: sin(n*theta)/sin(theta) -> (-1)^(n+1)*n.
            downwash_half[-1] = (
                (-1.0)**(n + 1) * n
            )

            alpha_i_half = downwash_half @ A
            alpha_eff_deg = np.rad2deg(alpha - alpha_i_half)

            # local aerodynamic coeffs
            cl_local = 2.0 * gamma_over_V / c_half

            # Check the effective angles against the polar domain.
            if (
                np.min(alpha_eff_deg) < self.airfoil2d.min_alpha
                or np.max(alpha_eff_deg) > self.airfoil2d.max_alpha
            ):
                raise ValueError(
                    f"Effective angle of attack lies outside the XFoil polar range [{self.airfoil2d.min_alpha:.2f}, {self.airfoil2d.max_alpha:.2f}] deg. Extend the polar or restrict the requested alpha."
                )

            cd_local = self.airfoil2d.CD(alpha_eff_deg)
            cm_local = self.airfoil2d.CM(alpha_eff_deg)

            # integrate over one semi-span: CL = 2/(V*S_semi) * integral(Gamma dy)
            CL = (
                2.0 / S_semi
                * np.trapezoid(gamma_over_V, y_half)
            )

            # Profile drag coefficient
            CD_profile = (np.trapezoid(c_half * cd_local, y_half) / S_semi)

            # Induced drag coefficient
            CD_induced = (
                2.0 / S_semi *
                np.trapezoid(gamma_over_V * alpha_i_half, y_half)
            )

            CD = CD_profile + CD_induced

            # Sectional airfoil pitching-moment contribution only.
            # Moment reference follows the input XFoil CM reference.
            CM_airfoil_only = (
                np.trapezoid(
                    c_half**2 * cm_local,
                    y_half
                ) / (S_semi * c_bar)
            )

            # Span efficiency from induced drag.
            e = (CL**2 / (np.pi * AR * CD_induced)
                 if CD_induced > 0.0 else np.nan)

            # results
            return {
                "CL": CL,
                "CD": CD,
                "CD_profile": CD_profile,
                "CD_induced": CD_induced,
                "CM_airfoil_only": CM_airfoil_only,
                "e": e,
                "cl_local": cl_local,
                "cd_local": cd_local,
                "cm_local": cm_local,
                "alpha_eff_deg": alpha_eff_deg,
                "alpha_i_deg": np.rad2deg(alpha_i_half),
                "A": A,
                "converged": solution.success,
                "residual_norm": np.linalg.norm(solution.fun),
            }
        except ValueError as e:
            return {
                "CL": np.nan,
                "CD": np.nan,
                "CD_profile": np.nan,
                "CD_induced": np.nan,
                "CM_airfoil_only": np.nan,
                "e": np.nan,
                "cl_local": np.nan,
                "cd_local": np.nan,
                "cm_local": np.nan,
                "alpha_eff_deg": np.nan,
                "alpha_i_deg": np.nan,
                "A": np.nan,
                "converged": np.nan,
                "residual_norm": np.nan,
            }

    @overload
    def CL(self, alpha: float) -> float: ...

    @overload
    def CL(self, alpha: np.ndarray) -> np.ndarray: ...

    def CL(self, alpha: float | np.ndarray) -> float | np.ndarray:
        if isinstance(alpha, float | int):
            return float(self.__lifting_line_solver(alpha)["CL"])
        elif isinstance(alpha, np.ndarray):
            return np.array(
                [self.CL(a) for a in alpha]
            )
        else:
            raise TypeError(f"Unsupported type ({type(alpha)})")

    @overload
    def CD(self, alpha: float) -> float: ...

    @overload
    def CD(self, alpha: np.ndarray) -> np.ndarray: ...

    def CD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        if isinstance(alpha, float | int):
            return float(self.__lifting_line_solver(alpha)["CD"])
        elif isinstance(alpha, np.ndarray):
            return np.array(
                [self.CD(a) for a in alpha]
            )
        else:
            raise TypeError(f"Unsupported type ({type(alpha)})")

    @overload
    def CD_profile(self, alpha: float) -> float: ...

    @overload
    def CD_profile(self, alpha: np.ndarray) -> np.ndarray: ...

    def CD_profile(self, alpha: float | np.ndarray) -> float | np.ndarray:
        if isinstance(alpha, float | int):
            return float(self.__lifting_line_solver(alpha)["CD_profile"])
        elif isinstance(alpha, np.ndarray):
            return np.array(
                [self.CD_profile(a) for a in alpha]
            )
        else:
            raise TypeError(f"Unsupported type ({type(alpha)})")

    @overload
    def CD_induced(self, alpha: float) -> float: ...

    @overload
    def CD_induced(self, alpha: np.ndarray) -> np.ndarray: ...

    def CD_induced(self, alpha: float | np.ndarray) -> float | np.ndarray:
        if isinstance(alpha, float | int):
            return float(self.__lifting_line_solver(alpha)["CD_induced"])
        elif isinstance(alpha, np.ndarray):
            return np.array(
                [self.CD_induced(a) for a in alpha]
            )
        else:
            raise TypeError(f"Unsupported type ({type(alpha)})")

    @overload
    def CM(self, alpha: float) -> float: ...

    @overload
    def CM(self, alpha: np.ndarray) -> np.ndarray: ...

    def CM(self, alpha: float | np.ndarray) -> float | np.ndarray:
        if isinstance(alpha, float | int):
            return float(self.__lifting_line_solver(alpha)["CM_airfoil_only"])
        elif isinstance(alpha, np.ndarray):
            return np.array(
                [self.CM(a) for a in alpha]
            )
        else:
            raise TypeError(f"Unsupported type ({type(alpha)})")

    def estimated_CL_max(self, stall_correction: float = 1.0) -> float:
        """estimated wing maximum coefficient of lift, CL_max. Approximately just before stall.

        Stall correction factor used to empirically link to other reported CL_max values"""
        max_CL = 0
        for alpha in np.arange(0, 90, 0.1):
            CL = self.CL(alpha)

            if np.isnan(CL):
                return max_CL * stall_correction

            if CL > max_CL:
                max_CL = CL

        return max_CL * stall_correction
