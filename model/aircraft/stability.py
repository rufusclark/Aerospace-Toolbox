from __future__ import annotations
from model.util.dataclass import FrozenDataclassBase
from dataclasses import dataclass, field
from model.util.output import *
import numpy as np


@dataclass(frozen=True)
class StabilityAnalysis(FrozenDataclassBase):
    category = "stability-analysis"
    name: str = field(default="stability-analysis", kw_only=True)

    V_bar_T: float
    """tail volume coefficient"""

    a: float
    """main-wing lift-curve slope, dCL_w/dalpha [1/rad]"""
    a_1: float
    """tailplane lift-curve slope, dCL_T/dalpha_T [1/rad]"""
    a_2: float = field(default=None, kw_only=True)  # type: ignore
    """tail lift derivative with elevator deflection, dCL_T/deta [1/rad]
    
    if not provided or set as none this will be estimated using a empirical relations. elevator_chord_fraction must be provided for this."""
    a_3: float = field(default=0.0, kw_only=True)
    """tail lift derivative with trim-tab deflection, dCL_T/dbeta_eta [1/rad]"""

    Cm_0: float
    """zero lift pitching moment coefficient"""
    CL_w: float | list[float] | np.ndarray
    """main wing lift coefficient at cruise"""

    h: float
    """center of mass as a fraction of mean aerodynamic chord"""
    h_0: float
    """aerodynamic center as a fraction of mean aerodnyamic chord"""

    downwash_gradient: float
    """downwash gradient, d(epsilon)/d(alpha) [rad/rad]"""

    elevator_chord_fraction: float = field(
        default=None, kw_only=True)  # type: ignore

    eta_T_deg: float | list[float] | np.ndarray
    """tailplane setting angle [deg]"""
    beta_eta_deg: float = field(default=0.0, kw_only=True)
    """elevator trim tab angle [deg]"""

    info: StabilityAnalysisInfo = field(init=False, repr=False)
    plots: StabilityAnalysisPlots = field(init=False, repr=False)
    elevator_angle_for_trim_rad: np.ndarray = field(init=False)

    def __post_init__(self):
        self._set("info", StabilityAnalysisInfo(self))
        self._set("plots", StabilityAnalysisPlots(self))

        if self.a_2 is None:
            if self.elevator_chord_fraction is None:
                raise ValueError(
                    "Elevator chord fraction must be provided for a_2 to be infered"
                )
            a_2 = (
                self.a_1 *
                np.sqrt(np.sin(0.5 * self.elevator_chord_fraction * np.pi))
            )
            self._set("a_2", a_2)

        CL_w = np.atleast_1d(self.CL_w)
        self._set("CL_w", CL_w)

        self._set("eta_T_deg", np.atleast_1d(self.eta_T_deg))

        elevator_angle_for_trim_rad = (
            ((self.Cm_0+CL_w*(self.h - self.h_0))/(self.V_bar_T*self.a_2)) -
            ((CL_w * self.a_1)/(self.a*self.a_2)) * (1 - self.downwash_gradient) -
            (self.a_3/self.a_2) * self.beta_eta_rad -
            (self.a_1/self.a_2) * self.eta_T_rad[:, np.newaxis]
        )
        self._set("elevator_angle_for_trim_rad", elevator_angle_for_trim_rad)

    @property
    def beta_eta_rad(self) -> float:
        """elevator trim tab angle [rad]"""
        return np.radians(self.beta_eta_deg)

    @property
    def eta_T_rad(self) -> np.ndarray:
        """tailplane setting angle [rad]"""
        return np.radians(self.eta_T_deg)

    @property
    def elevator_angle_for_trim_deg(self) -> np.ndarray:
        """elevator angle for trim [deg]"""
        return np.degrees(self.elevator_angle_for_trim_rad)


class StabilityAnalysisPlots(PlotBase[StabilityAnalysis]):

    @scientific_plot(auto_limits="x")
    def trim_analysis(self, ax: Axes):
        model = self.model

        assert isinstance(model.eta_T_deg, np.ndarray)

        for _, (elevator_angle, tailplane_setting_angle) in enumerate(zip(model.elevator_angle_for_trim_deg, model.eta_T_deg)):
            ax.plot(
                model.CL_w, elevator_angle,
                label=rf"Tail Setting Angle {tailplane_setting_angle:.2f}$^\circ$"
            )

        ax.legend()

        ax.set_xlabel("Lift Coefficient (Trimmed)")
        ax.set_ylabel("Elevator Angle [deg]")


class StabilityAnalysisInfo(InfoBase[StabilityAnalysis]):
    pass
