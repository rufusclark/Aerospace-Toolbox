from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal
from model.util.dataclass import FrozenDataclassBase
from model.aircraft.material import Material
from model.aircraft.spar import WingLoadingDistribution, LateralSpar
from model.aircraft.wing import LateralWing
from model.aircraft.base import Mass, Position
if TYPE_CHECKING:
    from model.aircraft.aircraft import Aircraft
import numpy as np
import math


@dataclass(frozen=True)
class MainWingSizing(FrozenDataclassBase):
    aircraft: "Aircraft"

    load: float
    """load on aircraft wings [N] this should consider the load factor"""
    S_main: float
    """main wing area [m^2]"""

    spar_material: Material
    """spar material"""
    airfoil_skin_material: Material
    """airfoil skin material"""
    airfoil_rib_material: Material
    """airfoil rib material"""

    spar_padding: float
    """padding around spar [m]"""
    spar_top_thickness: float
    """spar top thickness [m]"""
    spar_bottom_thickness: float
    """spar bottom thickness [m]"""
    spar_left_thickness: float
    """spar left side thickness [m]"""
    spar_right_thickness: float
    """spar right side thickness [m]"""
    spar_load_distribution: WingLoadingDistribution
    """spar load distribution model"""

    spar_min_factor_of_safety: float
    """spar minimum allowable safety factor"""
    spar_max_deflection: float
    """spar maximum allowable deflection [m]"""
    spar_max_deflection_angle: float
    """spar maximum allowable deflection angle [deg]"""

    airfoil_setting_angle: float
    """airfoil setting angle [deg]"""
    airfoil_camber: float
    """airfoil maximum camber (/c)"""
    airfoil_camber_location: float
    """airfoil camber location (/c)"""
    airfoil_aspect_ratio: float
    """airfoil aspect ratio"""
    airfoil_spanwise_efficiency_factor: float
    """spanwise/oswald efficiency factor, e"""

    obj_func: Callable[[LateralWing], float] = field(
        default=lambda wing: wing.mass.mass)
    """objective function to maximise where the input is `LiteralWing` and the output is the value `float` to be optimised"""
    obj_type: Literal["maximise", "minimise"] = "minimise"
    """type of optimisation to apply to the objective function"""

    offset: Position = Position()
    """position offset from position automatically calculated from main wing arm"""

    spar_mass: Mass = Mass()
    airfoil_mass: Mass = Mass()
    airfoil_skin_thickness: float | None = None
    airfoil_rib_spacing: float | None = None
    airfoil_rib_thickness: float | None = None

    rho: float = 1.225
    """air density [kg/m^3]"""

    @property
    def airfoil_chord(self) -> float:
        """airfoil chord [m]"""
        return math.sqrt(self.S_main / self.airfoil_aspect_ratio)

    @property
    def span(self) -> float:
        """main wing span [m]"""
        return self.S_main / self.airfoil_chord

    @property
    def semi_span(self) -> float:
        """main wing semi-span [m]"""
        return self.span / 2

    def effective_alpha(self, alpha: float) -> float:
        """effective main wing angle of attack, alpha_h [deg] as a function of angle of attack, alpha [deg]"""
        return alpha + self.airfoil_setting_angle

    @property
    def critical_load(self) -> float:
        """critical load [N] for main wing span"""
        return self.load

    def optimise_wing(self, show: bool = False) -> LateralWing:
        """generate and return an optimal wing that meets the defined requirements (sized for critical load)"""

        def get_wing(height: float) -> LateralWing:
            return LateralWing.from_optimised_thickness(
                length=self.semi_span,
                wing_oswald_efficiency_factor=self.airfoil_spanwise_efficiency_factor,

                spar_material=self.spar_material,
                spar_external_height=height,
                spar_padding=self.spar_padding,
                spar_top_thickness=self.spar_top_thickness,
                spar_bottom_thickness=self.spar_bottom_thickness,
                spar_left_thickness=self.spar_left_thickness,
                spar_right_thickness=self.spar_right_thickness,
                spar_load_distribution=self.spar_load_distribution,
                spar_load=self.critical_load / 2,

                spar_min_factor_of_safety=self.spar_min_factor_of_safety,
                spar_max_deflection=self.spar_max_deflection,
                spar_max_deflection_angle=self.spar_max_deflection_angle,

                airfoil_skin_material=self.airfoil_skin_material,
                airfoil_rib_material=self.airfoil_rib_material,
                airfoil_setting_angle=self.airfoil_setting_angle,
                airfoil_camber=self.airfoil_camber,
                airfoil_camber_location=self.airfoil_camber_location,
                airfoil_chord=self.airfoil_chord,
                airfoil_skin_thickness=self.airfoil_skin_thickness,
                airfoil_rib_spacing=self.airfoil_rib_spacing,
                airfoil_rib_thickness=self.airfoil_rib_thickness,
                airfoil_leading_edge=Position(),
                spar_mass=self.spar_mass,
                airfoil_mass=self.airfoil_mass
            )

        def obj(height: float) -> float:
            wing = get_wing(height)
            return self.obj_func(wing)

        height_min = self.spar_top_thickness + self.spar_bottom_thickness

        heights: list[float] = []
        ys: list[float] = []

        height_step = 1 / 1000  # [m]
        solution_found: bool = False
        for i in range(500):
            height = height_min + height_step * i

            try:
                y = obj(height)
                heights.append(height)
                ys.append(y)
                solution_found = True
            except:
                if solution_found:
                    break

        if not heights:
            raise ValueError(
                f"No valid solution found. Likely due to under or oversized spar requirements. Consider changing the aspect ratio, or for small airfoils using a thinner material."
            )

        if self.obj_type == "minimise":
            i = np.argmin(ys)
        else:
            i = np.argmax(ys)
        optimal_height = heights[i]
        wing = get_wing(optimal_height)

        if show:
            optimal_obj_value = float(ys[i])
            print(
                f"Load (per semi-wing) {self.critical_load / 2:.2f}N\n"
                f"Optimal Height: {optimal_height*1e3:.1f}mm\n"
                f"Optimal Width: {wing.spar.external_width*1e3:.1f}mm\n"
                f"Optimal Objective Value: {optimal_obj_value:.2f}\n"
                f"Airfoil: {wing.airfoil3D.airfoil2d.name}"
            )

            wing.plots.side_elevation(save=False, show=True)
            wing.plots.planform(save=False, show=True)

            wings = [get_wing(height) for height in heights]
            wings[0].plots.spar_height_overview(
                show=True, save=False, heights=heights, semi_wings=wings
            )

        return wing.copy(wing.component_name, offset=self.offset)
