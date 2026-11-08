from __future__ import annotations
from model.util.dataclass import FrozenDataclassBase
from model.util.output import *
from dataclasses import dataclass, field
import pandas as pd
import math


@dataclass(frozen=True)
class LoadFactors(FrozenDataclassBase):
    category = "aircraft"
    name = "load-factor"
    info: LoadFactorInfo = field(init=False, repr=False)
    plots: LoadFactorPlots = field(init=False, repr=False)

    m: float
    """total mass [kg]"""

    CL: float
    """aircraft lift coefficient"""
    S_ref: float
    """CL reference area, usually wing area [m^3]"""
    V_cruise: float
    """cruise airspeed [m/s]"""

    bank_angle_deg: float
    """bank angle [deg]"""

    r_turn: float
    """turn radius [m]"""
    V_turn: float
    """turn airspeed [m/s]"""

    climb_flight_path_angle_deg: float
    """climb flight path angle [deg]"""

    g: float = 9.81
    """gravitational acceleration [m/s^2]"""
    rho: float = 1.225
    """air density [kg/m^3]"""

    def __post_init__(self):
        self._set("info", LoadFactorInfo(self))
        self._set("plots", LoadFactorPlots(self))

    @property
    def bank_angle(self) -> float:
        """bank angle [rad]"""
        return math.radians(self.bank_angle_deg)

    @property
    def climb_flight_path_angle(self) -> float:
        """climb flight path angle [rad]"""
        return math.radians(self.climb_flight_path_angle_deg)

    @property
    def CL_required_for_level_cruise(self) -> float:
        return (
            2 * self.m * self.g
            / (self.rho * self.V_cruise**2 * self.S_ref)
        )

    @property
    def W(self) -> float:
        """aircraft weight [N]"""
        return self.m * self.g

    @property
    def n_cruise(self) -> float:
        L = 0.5 * self.rho * self.V_cruise**2 * self.S_ref * self.CL
        return L / self.W

    @property
    def n_level_turn_from_bank_angle(self) -> float:
        """assumes constant speed and altitude banked turn"""
        return 1 / math.cos(self.bank_angle)

    @property
    def n_level_turn_from_radius(self) -> float:
        """assumes constant speed and altitude banked turn"""
        return math.sqrt(1 + ((self.V_turn**2)/(self.g*self.r_turn))**2)

    @property
    def n_steady_climb(self) -> float:
        """steady climb at constant flight path angle"""
        return math.cos(self.climb_flight_path_angle)

    @property
    def n_max(self) -> float:
        """maximum load factor"""
        return max(self.n_cruise, self.n_level_turn_from_bank_angle, self.n_level_turn_from_radius, self.n_steady_climb)


class LoadFactorPlots(PlotBase[LoadFactors]):
    ...


class LoadFactorInfo(InfoBase[LoadFactors]):

    @scientific_table()
    def overview(self):
        model = self.model
        return pd.DataFrame({
            f"cruise (airspeed: {model.V_cruise:.1f} m/s)": model.n_cruise,
            f"level turn (bank: {model.bank_angle_deg:.1f} deg)": model.n_level_turn_from_bank_angle,
            f"level turn (radius: {model.r_turn:.1f} m, airspeed: {model.V_turn:.1f} m/s)": model.n_level_turn_from_radius,
            f"steady climb": model.n_steady_climb,
            f"CL required for level cruise": model.CL_required_for_level_cruise
        }, index=[0])
