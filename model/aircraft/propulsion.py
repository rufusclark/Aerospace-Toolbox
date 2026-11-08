from __future__ import annotations
from dataclasses import dataclass, replace, field
from scipy.optimize import brentq
from typing import TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from model.aircraft.aircraft import Aircraft
from model.aircraft.component import MassComponent
from model.util.output import *
from model.util.dataclass import FrozenDataclassBase


@dataclass(frozen=True)
class PropulsionSizing(FrozenDataclassBase):
    """Wing dimensions, aerodynamics and aircraft mass are taken from the current working aircraft"""
    category = "aircraft"
    name = "propulsion-sizing"

    aircraft: Aircraft
    lipo: LiPo

    airspeed: float
    """cruise airspeed [m/s]"""
    range: float
    """range [m]"""
    endurance: float
    """flight endurance [min]"""

    speed_controller_efficiency: float
    """speed controller efficiency (0-1)"""
    motor_efficiency: float
    """motor efficiency (0-1)"""
    propulsive_efficiency: float
    """propulsive efficiency (0-1)"""

    g: float = 9.81
    """gravitational acceleration [m/s^2]"""
    rho: float = 1.225
    """air density [kg/m^3]"""

    plots: PropulsionSizingPlots = field(init=False, repr=False)

    def __post_init__(self):
        self._set("plots", PropulsionSizingPlots(self))

    @property
    def endurance_hr(self) -> float:
        return self.endurance / 60

    @property
    def combined_efficiency(self) -> float:
        return self.speed_controller_efficiency * self.motor_efficiency * self.propulsive_efficiency

    @property
    def q(self) -> float:
        """Dynamic pressure, q [Pa] at airspeed"""
        return 0.5 * self.rho * self.airspeed**2

    @property
    def CL_req(self) -> float:
        """Lift Coefficient for steady level flight"""
        return (self.aircraft.mass.mass * self.g) / (self.q * self.aircraft.S_main)

    @property
    def alpha_req(self) -> float:
        """Angle of attack, alpha [deg] required to maintain steady flight"""

        def lift_error(alpha: float) -> float:
            return float(self.aircraft.CL(alpha)) - self.CL_req

        alpha_min = -5.0  # [deg]
        alpha_max = 15.0  # [deg]

        res = brentq(lift_error, alpha_min, alpha_max)

        return float(res)  # type: ignore

    @property
    def CD(self) -> float:
        """Drag Coefficient for steady level flight"""

        alpha = self.alpha_req
        return self.aircraft.CD_0(alpha) + self.aircraft.k * self.CL_req**2

    @property
    def T_req(self) -> float:
        """Thrust [N] required for steady level flight

        this is not the maximum thrust required. This is only for steady level cruise and doesn't consider manoeuvres, cruise or takeoff"""
        return self.q * self.CD * self.aircraft.S_main

    @property
    def P_req_mechanical(self) -> float:
        """Mechanical Power [W] required for steady level flight"""
        return self.T_req * self.airspeed

    @property
    def P_req_electrical(self) -> float:
        """Electrical Power [W] required for steady level flight"""
        return self.P_req_mechanical / self.combined_efficiency

    @property
    def mean_current_req(self) -> float:
        """Mean Current [A] required for steady level flight"""
        return self.P_req_electrical / self.lipo.mean_pack_voltage

    @property
    def endurance_capacity_req(self) -> float:
        """Capacity [Ah] required to meet endurance duration"""
        return self.mean_current_req * self.endurance_hr

    @property
    def range_hr(self) -> float:
        """Duration [hr] required to meet range distance"""
        return (self.range / self.airspeed) / 3600

    @property
    def range_capacity_req(self) -> float:
        """Capacity [Ah] required to meet range distance"""
        return self.mean_current_req * self.range_hr

    @property
    def minimum_capacity_req(self) -> float:
        """Minimum Capacity [Ah] to meet both range and endurance requirements"""
        return max(self.range_capacity_req, self.endurance_capacity_req)

    @property
    def factional_LiPos_req(self) -> float:
        """Fractional number of LiPo's (i.e. 1.31)"""
        return self.minimum_capacity_req / self.lipo.useable_capacity

    @property
    def LiPos_req(self) -> int:
        """Number of LiPo's required (i.e. 1 or 2)"""
        return math.ceil(self.factional_LiPos_req)


class PropulsionSizingPlots(PlotBase[PropulsionSizing]):

    @scientific_plot()
    def airspeed_vs_capacity(self, ax: Axes, V_min: float = 0.01, V_max: float = 30):
        model = self.model

        V = np.linspace(V_min, V_max, 100)

        endurance_capacity_req = np.full_like(V, np.nan)
        range_capacity_req = np.full_like(V, np.nan)

        for i, v in enumerate(V):
            try:
                m = replace(model, airspeed=v)
                endurance_capacity_req[i] = m.endurance_capacity_req
                range_capacity_req[i] = m.range_capacity_req
            except:
                continue

        ax.plot(V, endurance_capacity_req * 1e3, label="Endurance")
        ax.plot(V, range_capacity_req * 1e3, label="Range")

        ax.legend()
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Capacity [mAh]")


@dataclass(frozen=True)
class LiPo(MassComponent):
    n_cells: int = field(kw_only=True)
    capacity: float = field(kw_only=True)
    """LiPo capacity [Ah]"""
    max_cell_voltage: float = 4.2
    """max cell voltage [V]"""
    min_useable_cell_voltage: float = 3.6
    """min useable cell voltage [V]"""
    useable_capacity_fraction: float = 0.8
    """useable capacity fraction (0-1)"""

    @property
    def max_pack_voltage(self) -> float:
        """maximum pack voltage [V]"""
        return self.n_cells * self.max_cell_voltage

    @property
    def min_useable_pack_voltage(self) -> float:
        """minimum usable pack voltage [V]"""
        return self.n_cells * self.min_useable_cell_voltage

    @property
    def mean_pack_voltage(self) -> float:
        """mean pack voltage [V]"""
        return 0.5 * (self.max_pack_voltage + self.min_useable_pack_voltage)

    @property
    def useable_capacity(self) -> float:
        """useable capacity [Ah]"""
        return self.capacity * self.useable_capacity_fraction
