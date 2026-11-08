from __future__ import annotations
from dataclasses import field, dataclass
from model.aircraft.base import ParentComponent, Mass, Position, ComponentPlots, ComponentInfo
from model.aircraft.spar import LateralSpar, WingLoadingDistribution, BaseSpar, VerticalSpar
from model.aircraft.airfoil3D import Airfoil3D, LateralAirfoil3D, VerticalAirfoil3D
from model.aircraft.material import Material
from model.util.output import *
from typing import Self, ClassVar
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class BaseWing(ParentComponent):
    component_name: str = field(init=False, default="LateralWing")
    plots: BaseWingPlots = field(init=False, repr=False)
    info: BaseWingInfo = field(init=False, repr=False)

    spar: BaseSpar
    airfoil3D: Airfoil3D

    spar_type: ClassVar[type[BaseSpar]] = field(
        init=False, default=BaseSpar)
    airfoil3D_type: ClassVar[type[Airfoil3D]] = field(
        init=False, default=Airfoil3D)

    def __post_init__(self):
        super().__post_init__()
        self._set("plots", BaseWingPlots(self))
        self._set("info", BaseWingInfo(self))

    @property
    def downwash_gradient(self) -> float:
        """estimate downwash gradient bsaed on lift curve slope and geometry using the method desribed here, (https://www.sciencedirect.com/science/chapter/monograph/pii/B9780123973085000118 - Eq 11-14).

        valid for initial designs"""
        a = self.airfoil3D.solvers.finite_wing.a
        AR = self.airfoil3D.solvers.AR
        return (2 * a) / (np.pi * AR)

    @classmethod
    def from_optimised_thickness(
        cls,

        length: float,

        wing_oswald_efficiency_factor: float,

        spar_material: Material,
        spar_external_height: float,
        spar_padding: float,
        spar_top_thickness: float,
        spar_bottom_thickness: float,
        spar_left_thickness: float,
        spar_right_thickness: float,
        spar_load_distribution: WingLoadingDistribution,
        spar_load: float,

        spar_min_factor_of_safety: float,
        spar_max_deflection: float,
        spar_max_deflection_angle: float,

        airfoil_setting_angle: float,
        airfoil_camber: float,
        airfoil_camber_location: float,
        airfoil_chord: float,

        airfoil_skin_material: Material,
        airfoil_rib_material: Material,

        airfoil_skin_thickness: float | None = None,
        airfoil_rib_spacing: float | None = None,
        airfoil_rib_thickness: float | None = None,

        spar_mass: Mass = Mass(),
        airfoil_leading_edge: Position = Position(),
        airfoil_mass: Mass = Mass(),
    ) -> Self:

        # determine optimal spar

        spar = cls.spar_type.from_optimisation(
            material=spar_material,
            external_height=spar_external_height,
            length=length,
            top_thickness=spar_top_thickness,
            bottom_thickness=spar_bottom_thickness,
            left_thickness=spar_left_thickness,
            right_thickness=spar_right_thickness,
            load_distribution=spar_load_distribution,
            load=spar_load,
            min_factor_of_safety=spar_min_factor_of_safety,
            max_deflection=spar_max_deflection,
            max_deflection_angle=spar_max_deflection_angle,
            mass=spar_mass
        )

        # print(f"Optimised spar width: {spar.external_width*1e3:.1f}mm")

        # find the thinnest 2d airfoil that will fit the spar
        airfoil2d, x, y = cls.airfoil3D_type.from_smallest_fitting_spar(
            camber=airfoil_camber,
            camber_location=airfoil_camber_location,
            chord=airfoil_chord,
            setting_angle=airfoil_setting_angle,
            spar=spar,
            padding=spar_padding,
        )

        # print(f"Optimised airfoil thickness: {airfoil2d.name}")

        # create the thinnest 3d airfoil from smallest 2d airfoil
        airfoil3d = cls.airfoil3D_type.from_leading_edge(
            skin_material=airfoil_skin_material,
            rib_material=airfoil_rib_material,
            length=length,
            chord=airfoil_chord,
            setting_angle=airfoil_setting_angle,
            airfoil2d=airfoil2d,
            leading_edge=airfoil_leading_edge,
            oswald_efficiency_factor=wing_oswald_efficiency_factor,
            skin_thickness=airfoil_skin_thickness,
            rib_spacing=airfoil_rib_spacing,
            rib_thickness=airfoil_rib_thickness,
            mass=airfoil_mass,
            overlapping_spar=spar
        )

        # create the final fin with new position
        spar = cls.spar_type(
            material=spar_material,
            center=airfoil_leading_edge + Position(x=-x, z=y),
            length=length,
            external_width=spar.external_width,
            external_height=spar.external_height,
            bottom_thickness=spar.bottom_thickness,
            top_thickness=spar.top_thickness,
            left_thickness=spar.left_thickness,
            right_thickness=spar.right_thickness,
            mass=spar.mass
        )

        # create wing
        return cls(
            spar=spar,
            airfoil3D=airfoil3d,
        )


@dataclass(frozen=True)
class LateralWing(BaseWing):
    component_name: str = field(init=False, default="LateralWing")

    spar: LateralSpar
    airfoil3D: LateralAirfoil3D

    spar_type: ClassVar[type[LateralSpar]] = field(
        init=False, default=LateralSpar)
    airfoil3D_type: ClassVar[type[LateralAirfoil3D]] = field(
        init=False, default=LateralAirfoil3D)


@dataclass(frozen=True)
class VerticalWing(BaseWing):
    component_name: str = field(init=False, default="VerticalWing")

    spar: VerticalSpar
    airfoil3D: VerticalAirfoil3D

    spar_type: ClassVar[type[VerticalSpar]] = field(
        init=False, default=VerticalSpar)
    airfoil3D_type: ClassVar[type[VerticalAirfoil3D]] = field(
        init=False, default=VerticalAirfoil3D)


class BaseWingPlots(ComponentPlots[BaseWing]):

    @scientific_plot(auto_limits="x")
    def spar_height_vs_width(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        widths = np.array(
            [semi_wing.spar.external_width for semi_wing in semi_wings])

        ax.plot(np.array(heights) * 1e3, widths * 1e3)

        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Spar Width [mm]")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_mass(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        mass = [semi_wing.mass.mass for semi_wing in semi_wings]

        ax.plot(np.array(heights) * 1e3, mass)

        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Wing Mass [kg]")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_wing_thickness(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        wing_thickness = np.array(
            [semi_wing.airfoil3D.wing_thickness for semi_wing in semi_wings])

        ax.plot(np.array(heights) * 1e3, wing_thickness * 1e3)

        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Wing Thickness [mm]")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_maximum_lift_coefficient(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        CL_max = np.array(
            [semi_wing.airfoil3D.solvers.lifting_line_theory.estimated_CL_max()
             for semi_wing in semi_wings]
        )

        ax.plot(np.array(heights) * 1e3, CL_max)

        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Maximum\nLift Coefficient")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_lift_to_drag_ratio(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        lift_to_drag_ratio = [
            semi_wing.airfoil3D.solvers.LD(semi_wing.airfoil3D.setting_angle)
            for semi_wing in semi_wings
        ]

        ax.plot(np.array(heights) * 1e3, lift_to_drag_ratio)

        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("3D Lift-to-Drag Ratio")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_lift_coefficient(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        CL = [
            semi_wing.airfoil3D.solvers.CL(alpha=0) for semi_wing in semi_wings
        ]

        ax.plot(np.array(heights) * 1e3, CL)
        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Lift Coefficient\n(alpha=0 deg)")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_drag_coefficient(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        CD = [
            semi_wing.airfoil3D.solvers.CD(alpha=0) for semi_wing in semi_wings
        ]

        ax.plot(np.array(heights) * 1e3, CD)
        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Drag Coefficient\n(alpha=0 deg)")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_lift_curve_slope(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        a = [
            semi_wing.airfoil3D.solvers.finite_wing.a for semi_wing in semi_wings
        ]

        ax.plot(np.array(heights) * 1e3, a)
        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Lift Curve Slope")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_LD_over_mass(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        LD_over_mass = [
            semi_wing.airfoil3D.solvers.lifting_line_theory.LD(
                alpha=0) / semi_wing.mass.mass
            for semi_wing in semi_wings
        ]

        ax.plot(np.array(heights) * 1e3, LD_over_mass)
        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("(Lift/Drag)/Mass")

    @scientific_plot(auto_limits="x")
    def spar_height_vs_lift_curve_slope_over_mass(self, ax: Axes, heights: list[float], semi_wings: list[BaseWing]):
        LD_over_mass = [
            semi_wing.airfoil3D.solvers.finite_wing.a / semi_wing.mass.mass
            for semi_wing in semi_wings
        ]

        ax.plot(np.array(heights) * 1e3, LD_over_mass)
        ax.set_xlabel("Spar Height [mm]")
        ax.set_ylabel("Lift Curve\nSlope/Mass")

    @scientific_compose()
    def spar_height_overview(self, heights: list[float], semi_wings: list[BaseWing]):
        plots = [
            self.spar_height_vs_width.with_args(
                heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_mass.with_args(
                heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_wing_thickness.with_args(
                heights=heights, semi_wings=semi_wings),
            # self.spar_height_vs_maximum_lift_coefficient.with_args(
            #     heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_lift_to_drag_ratio.with_args(
                heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_lift_coefficient.with_args(
                heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_drag_coefficient.with_args(
                heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_lift_curve_slope.with_args(
                heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_LD_over_mass.with_args(
                heights=heights, semi_wings=semi_wings),
            self.spar_height_vs_lift_curve_slope_over_mass.with_args(
                heights=heights, semi_wings=semi_wings
            )

        ]
        return Composition(
            plots=plots,
            figsize=(6, 1.5 * len(plots)),
            sharex=True,
        )


class BaseWingInfo(ComponentInfo[BaseWing]):

    @scientific_table()
    def geometry(self):
        model = self.model

        airfoil3d = model.airfoil3D
        spar = model.spar
        airfoil2d = airfoil3d.airfoil2d

        return pd.DataFrame({
            "airfoil": airfoil2d.name,
            "camber [/c]": airfoil2d.maximum_camber,
            "camber location [/c]": airfoil2d.maximum_camber_location,
            "thickness [/c]": airfoil2d.thickness,
            "chord [m]": airfoil3d.chord,
            "setting angle [deg]": airfoil3d.setting_angle,
            "length/semi-span [m]": spar.length,
            "airfoil skin material": airfoil3d.skin_material.name,
            "airfoil skin thickness [m]": airfoil3d.skin_thickness,
            "airfoil rib material": airfoil3d.rib_material.name,
            "airfoil rib thickness [m]": airfoil3d.rib_thickness,
            "airfoil rib spacing [m]": airfoil3d.rib_spacing,
            "spar material": spar.material.name,
            "spar height [m]": spar.external_height,
            "spar width [m]": spar.external_width,
            "spar top thickness [m]": spar.top_thickness,
            "spar bottom thickness [m]": spar.bottom_thickness,
            "spar left thickness [m]": spar.left_thickness,
            "spar right thickness [m]": spar.right_thickness
        }, index=[0])
