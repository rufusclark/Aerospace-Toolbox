from __future__ import annotations
from model.util.dataclass import FrozenDataclassBase
from dataclasses import dataclass, field
from typing import Sequence, Literal
from model.util.output import *
import pandas as pd
import numpy as np
import math


@dataclass(frozen=True)
class ConstraintAnalysis(FrozenDataclassBase):
    category = "constraints-analysis"
    name: str

    aspect_ratio: float
    oswald_efficiency_factor: float
    zero_lift_drag_coefficient: float

    landing_distance: float
    landing_friction_coefficient: float
    landing_lift_coefficient: float
    landing_headwind_speed: float

    takeoff_distance: float
    takeoff_lift_coefficient: float
    takeoff_ground_friction_coefficient: float
    takeoff_rolling_drag_coefficient: float
    """rolling drag or wheel drag coefficient"""
    takeoff_headwind_speed: float

    climb_angle_deg: float
    climb_lift_coefficient: float

    cruise_speed: float | list[float] | np.ndarray

    cruise_lift_coefficient: float
    turn_bank_angle_deg: float | list[float] | np.ndarray
    turn_speed: float | list[float] | np.ndarray

    g: float = 9.81
    """gravitational acceleration"""
    rho: float = 1.225
    """air density"""

    landing_stall_speed_safety_factor: float = 1.2
    takeoff_stall_speed_safety_factor: float = 1.1

    TW_max: float = 1
    WS_max: float = 100
    n: int = 100

    info: ConstraintAnalysisInfo = field(init=False, repr=False)
    plots: ConstraintAnalysisPlots = field(init=False, repr=False)

    design_TW: float = field(init=False, default=0)
    """design thrust to weight point"""
    design_WS: float = field(init=False, default=0)
    """design weight to wing area point"""

    TW: np.ndarray = field(init=False)
    WS: np.ndarray = field(init=False)

    WS_landing: float = field(init=False)
    TW_takeoff_ideal: np.ndarray = field(init=False)
    TW_takeoff: np.ndarray = field(init=False)
    TW_climb: float = field(init=False)
    TW_cruise: np.ndarray = field(init=False)
    TW_turn: np.ndarray = field(init=False)

    def __post_init__(self):
        self._set("info", ConstraintAnalysisInfo(self))
        self._set("plots", ConstraintAnalysisPlots(self))

        self._set("TW", np.linspace(0, self.TW_max, self.n))
        self._set("WS", np.linspace(0.1, self.WS_max, self.n))

        self._set("cruise_speed", np.atleast_1d(self.cruise_speed))
        self._set("turn_bank_angle_deg",
                  np.atleast_1d(self.turn_bank_angle_deg))
        self._set("turn_speed", np.atleast_1d(self.turn_speed))

        if self.turn_speed.shape != self.turn_bank_angle_deg.shape:  # type: ignore
            raise ValueError(
                f"turn speed and turn bank angle must have the same shape")

        # landing (with headwind)
        WS_landing = (
            0.5 * self.rho * self.landing_lift_coefficient * (
                self.landing_headwind_speed +
                (2 * self.g * self.landing_friction_coefficient *
                 self.landing_distance) ** 0.5
            ) ** 2 / self.landing_stall_speed_safety_factor**2
        )
        self._set("WS_landing", WS_landing)

        # landing
        # WS_landing = (
        #     self.landing_distance * self.landing_friction_coefficient *
        #     self.g * self.rho * self.landing_lift_coefficient
        # ) / (self.landing_stall_speed_safety_factor**2)
        # self._set("WS_landing", WS_landing)

        # takeoff ideal (with headwind)
        V_stall = np.sqrt(
            (2 * self.WS) / (self.rho * self.takeoff_lift_coefficient))
        V_TO = self.takeoff_stall_speed_safety_factor * V_stall  # airspeed
        V_ground_TO = V_TO - self.takeoff_headwind_speed  # ground speed

        headwind_factor = V_ground_TO / V_TO

        TW_takeoff_ideal = (
            (self.takeoff_stall_speed_safety_factor**2 * headwind_factor**2) /
            (self.takeoff_lift_coefficient *
             self.rho * self.g * self.takeoff_distance)
        ) * self.WS
        self._set("TW_takeoff_ideal", TW_takeoff_ideal)

        # takeoff ideal
        # TW_takeoff_ideal = ((self.takeoff_stall_speed_safety_factor**2) / (
        #     self.takeoff_lift_coefficient * self.rho * self.g * self.takeoff_distance
        # )) * (self.WS)
        # self._set("TW_takeoff_ideal", TW_takeoff_ideal)

        # takeoff (with headwind)
        V_stall = np.sqrt(
            (2 * self.WS) / (self.rho * self.takeoff_lift_coefficient))
        V_TO = self.takeoff_stall_speed_safety_factor * V_stall  # airspeed
        V_ground_TO = V_TO - self.takeoff_headwind_speed  # ground speed

        headwind_factor = V_ground_TO / V_TO

        EtaTO = (
            self.zero_lift_drag_coefficient
            + self.takeoff_rolling_drag_coefficient
            - self.takeoff_ground_friction_coefficient
            * self.takeoff_lift_coefficient
        )

        TW_takeoff = self.takeoff_ground_friction_coefficient + (
            self.takeoff_stall_speed_safety_factor**2
            * headwind_factor**2
            / self.takeoff_lift_coefficient
        ) * (
            EtaTO / (
                1 - np.exp(
                    -(self.rho * self.g * EtaTO * self.takeoff_distance)
                    / self.WS
                )
            )
        )

        self._set("TW_takeoff", TW_takeoff)

        # takeoff
        # EtaTO = (
        #     self.zero_lift_drag_coefficient + self.takeoff_rolling_drag_coefficient -
        #     self.takeoff_ground_friction_coefficient * self.takeoff_lift_coefficient
        # )
        # TW_takeoff = self.takeoff_ground_friction_coefficient + (
        #     self.takeoff_stall_speed_safety_factor**2 / self.takeoff_lift_coefficient
        # ) * (
        #     EtaTO / (
        #         1 - np.exp(
        #             -(self.rho * self.g * EtaTO * self.takeoff_distance) / self.WS
        #         )
        #     )
        # )
        # self._set("TW_takeoff", TW_takeoff)

        # climb
        climb_drag_coefficient = self.zero_lift_drag_coefficient + \
            self.induced_drag_factor * self.climb_lift_coefficient**2
        climb_lift_to_drag = self.climb_lift_coefficient / climb_drag_coefficient
        TW_climb = self.climb_angle_rad + 1/climb_lift_to_drag
        self._set("TW_climb", TW_climb)

        # cruise
        cruise_dynamic_pressure = 0.5 * self.rho * self.cruise_speed**2  # type: ignore
        TW_cruise = self.zero_lift_drag_coefficient * (
            cruise_dynamic_pressure[:, np.newaxis]/self.WS  # type: ignore
        ) + self.induced_drag_factor * (
            self.WS / cruise_dynamic_pressure[:, np.newaxis]  # type: ignore
        )
        self._set("TW_cruise", TW_cruise)

        # turn
        turn_n = 1 / np.cos(self.turn_bank_angle)
        turn_dynamic_pressure = 0.5 * self.rho * self.turn_speed**2  # type: ignore
        turn_drag_coefficient = self.zero_lift_drag_coefficient + \
            self.induced_drag_factor * self.cruise_lift_coefficient**2
        TW_turn = (
            (turn_dynamic_pressure[:, np.newaxis] *  # type: ignore
             turn_drag_coefficient) / self.WS) + (
            (self.induced_drag_factor * turn_n[:, np.newaxis]**2 * self.WS
             ) / turn_dynamic_pressure[:, np.newaxis]  # type: ignore
        )
        self._set("TW_turn", TW_turn)

    def set_optimal_design_point(self, mode: Literal["Min T/W", "Min T/W @ Max W/S"]) -> None:
        """sets the design point at the point

        Raises:
            ValueError: No valid points found
        """

        # collect all constraints
        constraints = [
            np.asarray(self.TW_takeoff),
            np.full_like(self.TW, self.TW_climb),
            np.asarray(self.TW_takeoff_ideal),
        ]
        constraints.extend(np.asarray(TW) for TW in self.TW_cruise)
        constraints.extend(np.asarray(TW) for TW in self.TW_turn)

        # get minimum TW
        TW_required = np.maximum.reduce(constraints)

        # apply the wing loading limit from landing
        feasible = (
            (self.WS > 0) &
            (self.WS <= self.WS_landing) &
            (TW_required <= self.TW_max)
        )

        if not np.any(feasible):
            raise ValueError("No feasible design point found")

        match mode:
            case "Min T/W":
                # find the feasible point requiring minimum T/W
                indices = np.flatnonzero(feasible)
                i = indices[np.argmin(TW_required[feasible])]

                WS = float(self.WS[i])
                TW = float(TW_required[i]) + (self.TW_max / self.n)
            case "Min T/W @ Max W/S":
                WS = float(self.WS[feasible][-1])
                TW = float(TW_required[feasible][-1])
            case _:
                raise ValueError(f"mode ({mode}) not supported")

        self.set_design_point(WS=WS, TW=TW)

    def set_design_point(self, WS: float, TW: float) -> None:
        self._set("design_TW", TW)
        self._set("design_WS", WS)

    @property
    def climb_angle_rad(self) -> float:
        return math.radians(self.climb_angle_deg)

    @property
    def turn_bank_angle(self):
        return np.radians(self.turn_bank_angle_deg)

    @property
    def induced_drag_factor(self) -> float:
        return 1 / (math.pi * self.oswald_efficiency_factor * self.aspect_ratio)

    @property
    def cruise_stall_speed(self) -> float:
        return math.sqrt((2 * self.design_WS) / (self.rho * self.cruise_lift_coefficient))

    @property
    def takeoff_stall_speed(self) -> float:
        return math.sqrt((2 * self.design_WS) / (self.rho * self.takeoff_lift_coefficient))

    @property
    def takeoff_speed(self) -> float:
        return self.takeoff_stall_speed_safety_factor * self.takeoff_stall_speed

    @property
    def landing_stall_speed(self) -> float:
        return math.sqrt((2 * self.design_WS) / (self.rho * self.landing_lift_coefficient))

    @property
    def landing_speed(self) -> float:
        """airspeed"""
        return self.landing_stall_speed_safety_factor * self.landing_stall_speed - self.landing_headwind_speed

    @property
    def landing_deceleration(self) -> float:
        return self.landing_friction_coefficient * self.g

    @property
    def design_landing_distance(self) -> float:
        return self.landing_speed**2 / (2 * self.landing_deceleration)

    @property
    def climb_stall_speed(self) -> float:
        return math.sqrt((2 * self.design_WS) / (self.rho * self.climb_lift_coefficient))


class ConstraintAnalysisPlots(PlotBase[ConstraintAnalysis]):

    @scientific_plot(outside_legend="right")
    def constraint_analysis(self, ax: Axes, design_points: Sequence[tuple[float, float, str]]):
        model = self.model
        ax.set_xlabel(r"$W/S$, [N/m$^2$]")
        ax.set_ylabel(r"$T/W$")

        WS_min, WS_max = 0, model.WS_max
        TW_min, TW_max = 0, model.TW_max

        # landing
        color = colour_palette[0]
        ax.axvline(
            model.WS_landing, label="Landing", color=color, linestyle="solid"
        )
        ax.axvspan(
            model.WS_landing, WS_max, color=color, alpha=0.1
        )

        # ideal takeoff
        color = colour_palette[1]
        ax.plot(
            model.WS, model.TW_takeoff_ideal, label="Ideal Takeoff", color=color, linestyle="dashed"
        )
        ax.fill_between(
            model.WS, model.TW_takeoff_ideal, TW_min, color=color, alpha=0.1
        )

        # takeoff
        color = colour_palette[2]
        ax.plot(
            model.WS, model.TW_takeoff, label="Takeoff", color=color,
            linestyle="dashed"
        )
        ax.fill_between(
            model.WS, model.TW_takeoff, TW_min, color=color, alpha=0.1
        )

        # climb
        color = colour_palette[3]
        ax.axhline(
            model.TW_climb, label="Climb", color=color, linestyle="solid"
        )
        ax.axhspan(
            model.TW_climb, TW_min, color=color, alpha=0.1
        )

        # cruise
        for i, (cruise_speed, TW) in enumerate(  # type: ignore
            zip(model.cruise_speed, model.TW_cruise)  # type: ignore
        ):
            color = colour_palette[i]
            ax.plot(
                model.WS, TW, label=f"Cruise ({cruise_speed}m/s)", color=color, linestyle="dashdot"
            )
            ax.fill_between(
                model.WS, TW, TW_min, color=color, alpha=0.1
            )

        # turn
        for i, (turn_speed, turn_angle, TW) in enumerate(  # type: ignore
            zip(
                model.turn_speed, model.turn_bank_angle_deg, model.TW_turn  # type: ignore
            )
        ):
            color = colour_palette[i]
            ax.plot(
                model.WS, TW, label=rf"Turn ({turn_angle}$^\circ$ {turn_speed}m/s)", color=color, linestyle="dotted"
            )
            ax.fill_between(
                model.WS, TW, TW_min, color=color, alpha=0.1
            )

        # design point
        for design_point in design_points:
            WS, TW, label = design_point
            ax.scatter(WS, TW, label=label, marker="x", s=50, zorder=2)

        if model.design_TW != 0 and model.design_WS != 0:
            ax.scatter(
                model.design_WS, model.design_TW, label="Design Point", marker="x", s=100, color="black", zorder=3
            )

        ax.legend()
        ax.set_xlim(WS_min, WS_max)
        ax.set_ylim(TW_min, TW_max)


class ConstraintAnalysisInfo(InfoBase[ConstraintAnalysis]):

    @scientific_table()
    def design_point(self):
        model = self.model
        return pd.DataFrame({
            "Thrust to Weight Ratio": model.design_TW,
            "Wing Loading [N/m^2]": model.design_WS,
            "Takeoff Stall Speed [m/s]": model.takeoff_stall_speed,
            "Takeoff Speed [m/s]": model.takeoff_speed,
            "Landing Stall Speed [m/s]": model.landing_stall_speed,
            "Landing Speed [m/s]": model.landing_speed,
            "Landing Deceleration [m/s^2]": model.landing_deceleration,
            "Landing Distance [m]": model.design_landing_distance,
            "Climb Stall Speed [m/s]": model.climb_stall_speed,
            "Cruise Stall Speed [m/s]": model.cruise_stall_speed
        }, index=[0])
