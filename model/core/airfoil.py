from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass, field
from model.util.path import pathManager
from typing import overload, Literal
from model.util.output import *
from io import StringIO
import pandas as pd
import numpy as np
import subprocess
import pathlib
import math


@dataclass(frozen=True)
class Airfoil:
    name: str
    DAT_path: pathlib.Path
    polar_path: pathlib.Path

    polar_df: pd.DataFrame = field(init=False, repr=False)
    DAT_df: pd.DataFrame = field(init=False, repr=False)

    plots: "AirfoilPlots" = field(init=False, repr=False)
    info: "AirfoilInfo" = field(init=False, repr=False)
    category: str = "airfoil"

    def __post_init__(self):
        object.__setattr__(self, "plots", AirfoilPlots(self))
        object.__setattr__(self, "info", AirfoilInfo(self))
        object.__setattr__(self, "polar_df", self.__read_polar_file())
        object.__setattr__(self, "DAT_df", self.__read_DAT_file())

    def __read_polar_file(self) -> pd.DataFrame:
        columns = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]

        with self.polar_path.open("r") as file:
            lines = file.readlines()

        # find the beginning of the numerical data
        for i, line in enumerate(lines):
            if line.strip().startswith("alpha"):
                data_start = i + 2
                break

        data = "".join(lines[data_start:])

        df = pd.read_csv(StringIO(data), sep=r"\s+", names=columns,)

        if len(df) == 0:
            raise ValueError(
                "xFoil polar data empty. xFoil likey failed due to separation at high angles of atack")

        if not np.isfinite(df.apply(pd.to_numeric, errors="coerce")).all().all():
            raise ValueError("xFoil polar data contains non-finite values.")

        if np.any(np.diff(df["alpha"].to_numpy()) <= 0):
            raise ValueError("xFoil alphas values must be unique and ordered")

        return df

    def __read_DAT_file(self):
        df = pd.read_csv(
            self.DAT_path, sep=r"\s+", skiprows=1, names=["x", "y"]
        )
        return df

    @property
    def CL_polar(self) -> np.ndarray:
        return self.polar_df["CL"].to_numpy()

    @property
    def CD_polar(self) -> np.ndarray:
        return self.polar_df["CD"].to_numpy()

    @property
    def CM_polar(self) -> np.ndarray:
        return self.polar_df["CM"].to_numpy()

    @property
    def alpha_polar(self) -> np.ndarray:
        return self.polar_df["alpha"].to_numpy()

    @property
    def x_points(self) -> np.ndarray:
        return self.DAT_df["x"].to_numpy()

    @property
    def y_points(self) -> np.ndarray:
        return self.DAT_df["y"].to_numpy()

    @property
    def Re(self) -> float:
        """reynolds number for xFoil polar data"""
        return float(self.polar_path.name.split("-Re-")[1].split("-Mach-")[0])

    @property
    def Mach(self) -> float:
        """Mach number for xFoil polar data"""
        return float(self.polar_path.name.split("-Mach-")[1].split("-alpha-")[0])

    def scale_DAT(self, chord: float = 1, angle: float = 0, facing: Literal["left", "right"] = "left") -> tuple[np.ndarray, np.ndarray]:
        """return the x, y positions of the airfoil with the given chord, angle and facing direction"""
        x = self.x_points * chord
        y = self.y_points * chord

        self.SW_TXT_FMT(chord)

        if facing == "right":
            x = -x
        else:
            angle = -angle

        if angle == 0:
            return x, y

        theta = np.deg2rad(angle)
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        return x_rot, y_rot

    def SW_TXT_FMT(self, chord: float, *, SW_unit: Literal["mm"] = "mm") -> None:
        """save airfoil plot in .txt file that can be imported using the SolidWorks XYZ Curve Tool

        Args:
            chord: chord length [m]
        """
        if SW_unit == "mm":
            chord *= 1e3

        x = self.x_points * chord
        y = self.y_points * chord
        z = np.zeros_like(x)

        cords = np.column_stack((x, y, z))

        path = self.DAT_path.parent / f"{self.name}-c-{chord:.1f}mm.txt"
        if not path.exists():
            path.parent.mkdir(exist_ok=True, parents=True)

            np.savetxt(
                path,
                cords,
                delimiter="\t",
                fmt="%.6f"
            )

        upper = (y >= 0)
        upper_cords = cords[upper]
        path = self.DAT_path.parent / f"{self.name}-c-{chord:.1f}mm-upper.txt"
        if not path.exists():
            path.parent.mkdir(exist_ok=True, parents=True)

            np.savetxt(
                path,
                upper_cords,
                delimiter="\t",
                fmt="%.6f"
            )

        lower = (y <= 0)
        lower_cords = cords[lower]
        path = self.DAT_path.parent / f"{self.name}-c-{chord:.1f}mm-lower.txt"
        if not path.exists():
            path.parent.mkdir(exist_ok=True, parents=True)

            np.savetxt(
                path,
                lower_cords,
                delimiter="\t",
                fmt="%.6f"
            )

    @overload
    def CL(self, alpha: float) -> float: ...

    @overload
    def CL(self, alpha: np.ndarray) -> np.ndarray: ...

    def CL(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """lift coefficient, CL of 2D airfoil as a function of angle of attack, alpha [deg]"""
        alphas = self.alpha_polar
        cl = self.CL_polar

        if np.any(alpha < alphas.min()) or np.any(alpha > alphas.max()):
            raise ValueError(
                f"angle of attack ({alpha}) out of bounds ({alphas.min(), alphas.max()})"
            )

        result = np.interp(alpha, alphas, cl)

        if np.isscalar(alpha):
            return float(result)

        return result

    @overload
    def CD(self, alpha: float) -> float: ...

    @overload
    def CD(self, alpha: np.ndarray) -> np.ndarray: ...

    def CD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """drag coefficient, CD of 2D airfoil as a function of angle of attack, alpha [deg]"""
        alphas = self.alpha_polar
        cd = self.CD_polar

        if np.any(alpha < alphas.min()) or np.any(alpha > alphas.max()):
            raise ValueError(
                f"angle of attack ({alpha}) out of bounds ({alphas.min(), alphas.max()})"
            )

        result = np.interp(alpha, alphas, cd)

        if np.isscalar(alpha):
            return float(result)

        return result

    @overload
    def CM(self, alpha: float) -> float: ...

    @overload
    def CM(self, alpha: np.ndarray) -> np.ndarray: ...

    def CM(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """moment coefficient, CM with respect to the quarter chord point (x/c = 0.25) of 2D airfoil as a function of angle of attack, alpha [deg]"""
        alphas = self.alpha_polar
        cm = self.CM_polar

        if np.any(alpha < alphas.min()) or np.any(alpha > alphas.max()):
            raise ValueError(
                f"angle of attack ({alpha}) out of bounds ({alphas.min(), alphas.max()})"
            )

        result = np.interp(alpha, alphas, cm)

        if np.isscalar(alpha):
            return float(result)

        return result

    @overload
    def LD(self, alpha: float) -> float: ...

    @overload
    def LD(self, alpha: np.ndarray) -> np.ndarray: ...

    def LD(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """lift to drag ratio of 2D airfoil as a function of angle of attack, alpha [deg]"""
        return self.CL(alpha) / self.CD(alpha)

    @property
    def alpha_L0(self) -> float:
        """zero lift angle of attack, alpha_L=0 [deg]"""
        alpha = self.alpha_polar
        Cl = self.CL_polar

        idx = np.where(np.diff(np.sign(Cl)) != 0)[0][0]  # type: ignore
        alpha_zero = np.interp(0, Cl[idx:idx + 2], alpha[idx:idx + 2])

        return float(alpha_zero)

    @property
    def CD_0(self) -> float:
        """zero lift drag coefficient, CD0"""
        return self.CD(self.alpha_L0)

    @property
    def a(self) -> float:
        """lift curve slope, a [1/rad]"""
        return self.a_by_alpha(0)

    @overload
    def a_by_alpha(self, alpha: float) -> float: ...

    @overload
    def a_by_alpha(self, alpha: np.ndarray) -> np.ndarray: ...

    def a_by_alpha(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """lift curve slope, a [1/rad] at angle of attack, alpha [deg] after smoothing"""
        alphas = self.alpha_polar
        Cl = self.CL_polar

        spline = UnivariateSpline(alphas, Cl, k=3, s=1e-4)
        slope_at_alpha_deg = spline.derivative()(alpha)

        return slope_at_alpha_deg * 180/np.pi  # type: ignore

    def airfoil_perimeter_length(self, chord: float = 1) -> float:
        """return the perimeter length [m] of the airfoil by summing the area between adjacent points"""
        x = self.x_points
        y = self.y_points

        dx = np.diff(x)
        dy = np.diff(y)

        perimeter = np.sum(np.sqrt(dx**2 + dy**2))

        return float(perimeter * chord)

    def airfoil_cross_sectional_area(self, chord: float = 1) -> float:
        """return the cross-sectional area [m^2] of the airfoil by using the shoelace formula"""
        x = self.x_points
        y = self.y_points

        area = 0.5 * abs(
            np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        )

        return float(area * chord)

    @property
    def maximum_camber_location(self) -> float:
        """maximum camber location fraction"""
        if not self.name.startswith("NACA "):
            raise NotImplementedError(
                f"Only supports NACA 4 and 5 series airfoils")
        out = self.name.removeprefix("NACA ").strip()
        if len(out) == 4:
            return int(out[1]) / 10
        elif len(out) == 5:
            return int(out[1:3]) / 200
        else:
            raise ValueError(f"Unsupported format: {self.name}")

    @property
    def maximum_camber(self) -> float:
        """maximum camber fraction"""
        if not self.name.startswith("NACA "):
            raise NotImplementedError(
                f"Only supports NACA 4 and 5 series airfoils")
        out = self.name.removeprefix("NACA ").strip()
        if len(out) == 4:
            return int(out[0]) / 100
        else:
            raise ValueError(f"Unsupported format: {self.name}")

    @property
    def thickness(self) -> float:
        """thickness fraction (t/c)"""
        if not self.name.startswith("NACA "):
            raise NotImplementedError(
                f"Only supports NACA 4 and 5 series airfoils")
        out = self.name.removeprefix("NACA ").strip()
        if len(out) in (4, 5):
            return int(out[-2:]) / 100
        else:
            raise ValueError(f"Unsupported format: {self.name}")

    @property
    def min_alpha(self) -> float:
        """minimum angle of attack, alpha [deg]"""
        return self.alpha_polar[0]

    @property
    def max_alpha(self) -> float:
        """maximum angle of attack, alpha [deg]"""
        return self.alpha_polar[-1]

    @property
    def rocketpy_airfoil(self) -> tuple[np.ndarray, Literal["degrees"]]:
        """returns the airfoil angle of attack [deg] and lift coefficient in a format that rocketpy can ingest for fin performance"""
        alpha = self.alpha_polar
        CL = self.CL_polar

        mask = alpha >= 0

        return (
            np.column_stack((alpha[mask], CL[mask])),
            "degrees"
        )


class AirfoilPlots(PlotBase[Airfoil]):  # type: ignore

    @scientific_plot(auto_limits=None)
    def CL(self, ax: Axes):
        model = self.model

        ax.plot(model.alpha_polar, model.CL_polar)
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Lift Coefficient")

    @scientific_plot(auto_limits=None)
    def CD(self, ax: Axes):
        model = self.model

        ax.plot(model.alpha_polar, model.CD_polar)
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Drag Coefficient")

    @scientific_plot(auto_limits=None)
    def CM(self, ax: Axes):
        model = self.model

        ax.plot(model.alpha_polar, model.CM_polar)
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Moment Coefficient (about x/c=0.25)")

    @scientific_plot(auto_limits=None)
    def a(self, ax: Axes):
        model = self.model

        alpha = model.alpha_polar
        a = model.a_by_alpha(alpha)

        ax.plot(alpha, a)
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Lift Curve Slope")

    @scientific_plot(auto_limits=None)
    def drag_polar(self, ax: Axes):
        model = self.model

        ax.plot(model.CL_polar, model.CD_polar)
        ax.set_xlabel(r"Lift Coefficient")
        ax.set_ylabel(r"Drag Coefficient")

    @scientific_plot(auto_limits=None)
    def LD(self, ax: Axes):
        model = self.model

        ax.plot(model.alpha_polar, model.LD(model.alpha_polar))
        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Lift to Drag Ratio")

    @scientific_plot(auto_limits="0.1scale")
    def airfoil(self, ax: Axes):
        model = self.model

        ax.plot(model.DAT_df["x"], model.DAT_df["y"])
        ax.set_xlabel(r"x/c")
        ax.set_ylabel(r"y/c")
        ax.set_aspect("equal")


class AirfoilInfo(InfoBase[Airfoil]):

    @scientific_table()
    def overview(self):
        model = self.model

        try:
            camber = model.maximum_camber
        except ValueError:
            camber = "Unknown"

        return pd.DataFrame({
            "Name": model.name,
            "Thickness (t/c)": model.thickness,
            "Camber (/c)": camber,
            "Camber Location (x/c)": model.maximum_camber_location,
            "Zero Lift Angle of Attack, alpha_L=0 [deg]": model.alpha_L0,
            "Zero Lift Drag Coefficient, CD_0": model.CD_0,
            "Lift Curve Slope, a [1/rad]": model.a,
            "Lift Coefficient at 0deg": model.CL(0),
            "Drag Coefficient at 0deg": model.CD(0),
            "Moment Coefficient at 0deg": model.CM(0),
            "Lift to Drag Ratio at 0deg": model.LD(0),
            "Normalised Airfoil Cross-sectional Area": model.airfoil_cross_sectional_area(),
            "Normalised Airfoil Perimeter Length": model.airfoil_perimeter_length()
        }, index=[0])


class AirfoilManager:
    def __init__(self) -> None:
        pass

    def _data_path(self, airfoil_name: str) -> pathlib.Path:
        path = pathlib.Path("data/airfoil") / airfoil_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @overload
    def get_airfoil(
        self,
        *,
        airfoil_name: str,
        Re: float = 1000000,
        Mach: float = 0.15,
        iter_max: int = 100,
        alpha_min: float = -10,
        alpha_max: float = 20,
        alpha_step: float = 0.1
    ) -> Airfoil:
        ...

    @overload
    def get_airfoil(
        self,
        *,
        thickness: float,
        camber: float = 0,
        camber_location: float = 0,
        Re: float = 1000000,
        Mach: float = 0.15,
        iter_max: int = 100,
        alpha_min: float = -10,
        alpha_max: float = 20,
        alpha_step: float = 0.1
    ) -> Airfoil:
        ...

    def get_airfoil(
        self,
        *,
        airfoil_name: str | None = None,
        thickness: float | None = None,
        camber: float = 0,
        camber_location: float = 0,
        Re: float = 1000000,
        Mach: float = 0.15,
        iter_max: int = 100,
        alpha_min: float = -10,
        alpha_max: float = 25,
        alpha_step: float = 0.1
    ) -> Airfoil:
        """get airfoil data for an airfoil

        Args:
            airfoil_name: NACA airfoil name. Defaults to None.
            maximum_thickness: maximum airfoil thickness (i.e. 0.15 for 15% thickness as a fraction of chord). Defaults to None.
            chamber: maximum camber as a fraction of chord (i.e. 0.1 = 10% of chord). Defaults to 0.
            camber_location: location of maximum camber as a fraction of chord (i.e. 0.5 is the middle of the chord). Defaults to 0.
            Re: Reynolds number for simulations. Defaults to 1000000.
            Mach: Mach number for simulations. Defaults to 0.15.
            iter_max: Maximum number of iterations. Defaults to 100.

        Raises:
            ValueError: Invalid airfoil name. Please note this will not catch all errors

        Returns:
            Airfoil object containing airfoil data
        """

        if airfoil_name != None:
            # validate NACA name
            if not airfoil_name.startswith("NACA "):
                raise ValueError(
                    f"airfoil must be the format 'NACA 0012' or 'NACA 23112': {airfoil_name} provided")

        else:
            # generate NACA 4-series airfoil name
            if thickness is not None:
                M = round(camber * 100)
                P = round(camber_location * 10)
                TT = round(thickness * 100)
                airfoil_name = f"NACA {M}{P}{TT:02d}"

                # inputs = {'camber': camber, 'camber_location': camber_location, 'thickness': thickness}
                # print(f"'{airfoil_name}' generated from {inputs}")
            else:
                raise ValueError(
                    f"either airfoil name or parameters must be supplied")

        if airfoil_name == "NACA 0000":
            inputs = {
                'camber': camber, 'camber_location': camber_location, 'thickness': thickness
            }
            print(f"'{airfoil_name}' generated from {inputs}")

            raise ValueError(
                f"invalid airfoil name: {airfoil_name}, {{{airfoil_name, thickness, camber, camber_location}}}")

        # filenames
        DAT_filename = f"{airfoil_name}.DAT"
        polar_filename = f"{airfoil_name}-Re-{Re:.0f}-Mach-{Mach:.2f}-alpha-min-{alpha_min:.1f}-max-{alpha_max:.1f}.txt"
        data_path = self._data_path(airfoil_name)
        DAT_path = data_path / DAT_filename
        polar_path = data_path / polar_filename

        # check cache
        if DAT_path.exists() and polar_path.exists():
            return Airfoil(
                name=airfoil_name,
                DAT_path=DAT_path,
                polar_path=polar_path
            )

        commands = (
            f"{airfoil_name}\n"
            f"PANE\n"
            f"SAVE {DAT_filename}\n"
            f"OPER\n"
            f"OPER\n"
            f"VISC {Re}\n"
            f"MACH {Mach}\n"
            f"ITER {iter_max}\n"
            f"PACC\n"
            f"{polar_filename}\n"
            f"\n"
            f"ASEQ {alpha_min} {alpha_max} {alpha_step}\n"
            f"PACC\n"
            f"QUIT\n"
        )

        result = subprocess.run(
            [pathManager.xFoil],
            input=commands,
            text=True,
            capture_output=True,
            cwd=data_path
        )

        if not polar_path.exists():
            # debug
            print("Airfoil name:", airfoil_name)
            print("Re:", Re)
            print("Mach:", Mach)
            print("Alpha min:", alpha_min)
            print("Alpha max:", alpha_max)
            print("Alpha step:", alpha_step)

            print("XFoil executable:", pathManager.xFoil)
            print("Working directory:", data_path)
            print("Return code:", result.returncode)
            print("STDOUT:", result.stdout[-3000:])
            print("STDERR:", result.stderr[-3000:])

            raise RuntimeError(f"xFoil polar plot not created")

        airfoil = Airfoil(
            name=airfoil_name,
            DAT_path=DAT_path,
            polar_path=polar_path
        )

        return airfoil
