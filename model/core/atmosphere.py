# https://ntrs.nasa.gov/citations/19770009539


import ussa1976
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import TypeVar
from pathlib import Path


from model.util.output import *

T = TypeVar("T", float, np.ndarray)


class DewPoint:
    category = "atmosphere"
    name = "dew-point"

    def __init__(self) -> None:
        self.plots = DewPointPlots(self)

    def dew_point(self, T_C: T, RH: float) -> T:
        if RH < 0 or RH > 100:
            raise ValueError(
                f"RH should be between 0 and 100 inclusive, {RH} provided")

        # magnus formula
        gamma = np.log(RH/100) + ((17.625*T_C)/(243.04 + T_C))
        T_dew = (243.04*gamma) / (17.625 - gamma)
        return T_dew


class DewPointPlots(PlotBase[DewPoint]):

    @scientific_plot()
    def dew_point(self, ax: Axes, RHs: list[float] = [20, 40, 60, 80, 100], tempLow_C: float = -20, tempHigh_C: float = 50):
        model = self.model

        T_C = np.linspace(tempLow_C, tempHigh_C, 100)

        for RH in RHs:
            Tdew_C = model.dew_point(T_C, RH)
            ax.plot(T_C, Tdew_C, label=rf"{RH:.0f}\% RH")

        ax.legend()

        ax.set_xlabel(r"Air Temperature [$^\circ\mathrm{C}$]")
        ax.set_ylabel(r"Dew Temperature [$^\circ\mathrm{C}$]")


class USSA1976:
    category = "atmosphere"
    name = "us-standard-atmosphere-1976"

    def __init__(self, *, table_dir: str | Path = "data/atmosphere/ussa1976-si-table.csv") -> None:
        self.plots = USSA1976Plots(self)
        self.info = USSA1976Info(self)

        table_dir = Path(table_dir)

        if not table_dir.exists():
            raise FileExistsError(
                f"{table_dir} not found. downloading again from https://www.pdas.com/bigtables.html"
            )

        df = pd.read_csv(table_dir)

        df["z"] = df["z"] * 1e3

        self.interpolation_df = df

    def compute(self, z: NDArray, variables: list[str] | None = None):
        return ussa1976.compute(z, variables)

    def _compute(self, z: T, vars: str) -> T:
        match z:
            case np.ndarray():
                return self.compute(z, variables=[vars])[vars].values
            case int():
                return self._compute(float(z), vars=vars)
            case float():
                return float(self._compute(np.array([z]), vars=vars)[0])
            case _:
                raise NotImplementedError(f"{type(z)} is not supported")

    def temperature_K(self, z: T) -> T:
        """get the temp as z [m]

        Args:
            z: altitude [m]

        Returns:
            temp [K]
        """
        return self._compute(z, "t")

    def temperature(self, z: T) -> T:
        """get the temp at z [m]

        Args:
            z: altitude [m]

        Returns:
            temp [°C]
        """
        return self.temperature_K(z) - 273.15

    def pressure(self, z: T) -> T:
        """get the pressure at z [m]

        Args:
            z: altitude [m]

        Returns:
            pressure [Pa]
        """
        return self._compute(z, "p")

    def density(self, z: T) -> T:
        """get the density at z [m]

        Args:
            z: altitude [m]

        Returns:
            density [kg/m^3]
        """
        return self._compute(z, "rho")

    def interpolate_speed_of_sound(self, z: np.ndarray) -> np.ndarray:
        """interpolation based on table here, https://www.pdas.com/bigtables.html"""

        df = self.interpolation_df

        return np.interp(z, df["z"], df["c"])

    def speed_of_sound(self, z: T) -> T:
        """get the density at z [m]

        Args:
            z: altitude [m]

        Returns:
            speed of sound [m/s]
        """
        match z:
            case np.ndarray():
                below = z < 86000
                above = ~below

                result = np.empty_like(z, dtype=float)

                # ussa1976 implementation (up to 86km)
                result[below] = self.compute(
                    z[below], variables=["cs"])["cs"].values

                # interpolation above 86km
                result[above] = self.interpolate_speed_of_sound(z[above])

                return result
            case int():
                return self.speed_of_sound(float(z))
            case float():
                return float(self.speed_of_sound(np.array(z)))
            case _:
                raise NotImplementedError(f"{type(z)} is not supported")


class USSA1976Plots(PlotBase[USSA1976]):

    def all(self, z: NDArray | None = None, show: bool = False, save: bool = True):
        return super().all(z, show=show, save=save)

    @scientific_plot(auto_limits="y")
    def temperature(self, ax: Axes, z: NDArray | None = None):
        if z is None:
            z = ussa1976.core.DEFAULT_Z

        t = self.model.temperature(z)
        z = z / 1000

        ax.plot(t, z)
        ax.set_xlabel(r"Temperature [$^\circ\mathrm{C}$]")
        ax.set_ylabel(r"Altitude [km]")

        return f"{z.min()}km-{z.max()}km"

    @scientific_plot(auto_limits="y")
    def pressure(self, ax: Axes, z: NDArray | None = None):
        if z is None:
            z = ussa1976.core.DEFAULT_Z

        p = self.model.pressure(z) / 1000

        z = z / 1000

        ax.plot(p, z)
        ax.set_xlabel(r"Pressure [kPa]")
        ax.set_ylabel(r"Altitude [km]")

        return f"{z.min()}km-{z.max()}km"

    @scientific_plot(auto_limits="y")
    def density(self, ax: Axes, z: NDArray | None = None):
        if z is None:
            z = ussa1976.core.DEFAULT_Z

        rho = self.model.density(z)
        z = z / 1000

        ax.plot(rho, z)
        ax.set_xlabel(r"Density [$\mathrm{kg\,m^{-3}}$]")
        ax.set_ylabel(r"Altitude [km]")

        return f"{z.min()}km-{z.max()}km"

    @scientific_plot(auto_limits="y")
    def speed_of_sound(self, ax: Axes, z: NDArray | None = None):
        if z is None:
            z = ussa1976.core.DEFAULT_Z

        a = self.model.speed_of_sound(z)
        z = z / 1000

        ax.plot(a, z)
        ax.set_xlabel(r"Speed of Sound [$\mathrm{m\,s^{-1}}$]")
        ax.set_ylabel(r"Altitude [km]")

        return f"{z.min()}km-{z.max()}km"

    @scientific_compose()
    def overview(self, z: NDArray | None = None):
        if z is None:
            z = ussa1976.core.DEFAULT_Z

        plots = [
            self.pressure.with_args(z),
            self.temperature.with_args(z),
            self.density.with_args(z),
            self.speed_of_sound.with_args(z)
        ]

        return Composition(
            plots,
            ncols=len(plots),
            sharey=True
        )


class USSA1976Info(InfoBase[USSA1976]):

    @scientific_table()
    def summary(self, z: NDArray | None = None, z_points: list[float] = [], z_point_labels: list[str] = []):
        model = self.model

        if z is None:
            z = ussa1976.core.DEFAULT_Z

        if z_points:
            if not z_point_labels:
                z_point_labels = ["" for _ in z_points]

        t = model.temperature(z)
        p = model.pressure(z) / 1e3
        rho = model.density(z)
        a = model.speed_of_sound(z)

        z = z / 1000

        df = pd.DataFrame({
            "Quantity": ["Temperature [°C]", "Pressure [kPa]", "Density [kg/m^3]", "Speed of Sounds [m/s]"],
            "Minimum": [float(t.min()), float(p.min()), float(rho.min()), float(a.min())],
            "Maximum": [float(t.max()), float(p.max()), float(rho.max()), float(a.max())],
            **{
                f"{label} ({z_point/1e3:.0f} km)" if label else f"{z_point/1e3:.0f} km": [
                    model.temperature(z_point),
                    model.pressure(z_point) / 1e3,
                    model.density(z_point),
                    model.speed_of_sound(z_point)
                ]
                for label, z_point in zip(z_point_labels, z_points)
            }
        })

        return df
