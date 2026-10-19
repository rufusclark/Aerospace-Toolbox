"""Read, process and plot RAS Aero II outputs"""

from typing import TypeVar, Any
from pathlib import Path
import pandas as pd
import numpy as np

from model.util.output import *

T = TypeVar("T", float, np.ndarray)


class Aero:
    category = "aero"

    def __init__(self, path: str | Path) -> None:
        self.plots = AeroPlots(self)
        self.info = AeroInfo(self)

        if not isinstance(path, Path):
            self.path = Path(path)
        else:
            self.path = path

        if not self.path.exists():
            raise FileNotFoundError(f"{self.path.absolute()} not found")

        self.df = pd.read_csv(self.path)

        print(f"Loaded aero data: {self.path.absolute()}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path)
        }

    @property
    def name(self) -> str:
        return self.path.parent.name

    def power_on_drag(self, alpha: int = 0) -> list[list[float]]:
        mask = self.df["Alpha"] == alpha
        columns = ["Mach", "CD Power-On"]
        return self.df[mask][columns].values.tolist()

    def power_off_drag(self, alpha: int = 0) -> list[list[float]]:
        mask = self.df["Alpha"] == alpha
        columns = ["Mach", "CD Power-Off"]
        return self.df[mask][columns].values.tolist()

    def drag_coeff(self, mach: T, alpha: int, power_on: bool = False) -> T:
        column = "CD Power-On" if power_on else "CD Power-Off"

        data = self.df[self.df["Alpha"] == alpha]

        drag_coeff = np.interp(mach, data["Mach"], data[column])

        return drag_coeff  # type: ignore

    def lift_coeff(self, mach: T, alpha: int) -> T:
        data = self.df[self.df["Alpha"] == alpha]

        drag_coeff = np.interp(mach, data["Mach"], data["CL"])

        return drag_coeff  # type: ignore

    def dynamic_pressure(self, mach: T, altitude: float) -> T:
        from model.core.atmosphere import USSA1976
        atm = USSA1976()

        v = mach * atm.speed_of_sound(altitude)

        q = 0.5 * atm.density(altitude) * v**2

        return q

    def drag(self, mach: T, altitude: float, ref_area: float, alpha: int = 0, power_on=False) -> T:
        q = self.dynamic_pressure(mach, altitude)
        C_D = self.drag_coeff(mach=mach, alpha=alpha, power_on=power_on)
        drag = q * C_D * ref_area
        return drag

    def lift(self, mach: T, altitude: float, ref_area: float, alpha: int = 0) -> T:
        q = self.dynamic_pressure(mach, altitude)
        C_L = self.lift_coeff(mach=mach, alpha=alpha)
        lift = q * C_L * ref_area
        return lift


class AeroPlots(PlotBase[Aero]):

    @scientific_plot(auto_limits=None)
    def drag_coefficient_vs_angle_of_attack(self, ax: Axes, alphas: list[int] = [0, 2, 4]):
        df = self.model.df

        for label, col, linestyle in zip(
            ["Power On", "Power Off"],
            ["CD Power-On", "CD Power-Off"],
            ["solid", "dashed"]
        ):
            for alpha in alphas:

                mask = df["Alpha"] == alpha
                columns = ["Mach", col]
                plot_df = df[mask][columns]

                ax.plot(
                    plot_df["Mach"], plot_df[col],
                    label=rf"{label} ($\alpha = {alpha:.1f}^\circ$)",
                    linestyle=linestyle
                )

        ax.set_ylabel("Coefficient of Drag")
        ax.set_xlabel("Mach")
        ax.legend()

        return self.model.name

    @scientific_plot(auto_limits=None)
    def lift_coefficient_vs_angle_of_attack(self, ax: Axes, alphas: list[int] = [0, 2, 4]):
        df = self.model.df

        for alpha in alphas:

            mask = df["Alpha"] == alpha
            columns = ["Mach", "CL"]
            plot_df = df[mask][columns]

            ax.plot(
                plot_df["Mach"], plot_df["CL"],
                label=rf"($\alpha = {alpha:.1f}^\circ$)",
            )

        ax.set_ylabel("Coefficient of Lift")
        ax.set_xlabel("Mach")
        ax.legend()

        return self.model.name

    @scientific_plot()
    def drag_vs_mach(self, ax: Axes, ref_area: float, alts: list[float] = [0, 1000, 2000, 3000, 4000, 5000], mach_lim: tuple[float, float] = (0, 3), alpha: int = 0):
        df = self.model.df

        mach = np.arange(*mach_lim, 0.01)

        for alt in alts:
            drag = self.model.drag(mach, alt, ref_area, alpha)
            ax.plot(mach, drag, label=f"{alt/1000:.1f} km")

        ax.set_ylabel("Drag [N]")
        ax.set_xlabel("Mach")
        ax.legend()

        return self.model.name


class AeroInfo(InfoBase[Aero]):
    pass
