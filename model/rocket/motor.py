from dataclasses import dataclass, asdict, fields, field, replace
from typing import Literal, TypeVar, Sequence, Any
from functools import cached_property
from datetime import datetime
import pathlib
import numpy as np
import requests
import rocketpy
import base64
import json

from model.util.output import *


def total_impulse_to_motor_class(total_impulse: float):
    if total_impulse <= 2.5:
        return 'A'
    return chr(int(np.ceil(np.log2(total_impulse/2.5))) + 65)


T = TypeVar("T", float, Sequence[float])

# TODO: CachedProperties?


@dataclass(frozen=True)
class ThrustCurve:
    time: np.ndarray = field(default_factory=lambda: np.array([0]))
    thrust: np.ndarray = field(default_factory=lambda: np.array([0]))

    def __post_init__(self):
        time = np.asarray(self.time, dtype=float)
        thrust = np.asarray(self.thrust, dtype=float)

        if time.ndim != 1 or thrust.ndim != 1:
            raise ValueError("time and thrust must be 1D")
        if len(time) != len(thrust):
            raise ValueError("time and thrust must have the same length")

        # Make the arrays themselves immutable
        time.flags.writeable = False
        thrust.flags.writeable = False

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "thrust", thrust)

    def __len__(self) -> int:
        return len(self.time)

    def __repr__(self) -> str:
        return f"<ThrustCurve name={self.motor_name} max={self.max_thrust:.1f}N avg={self.average_thrust:.1f}N total_impulse={self.total_impulse:.1f}Ns burn_time={self.burn_time:.1f}s>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.motor_name,
            "max_thrust": self.max_thrust,
            "average_thrust": self.average_thrust,
            "total_impulse": self.total_impulse,
            "burn_time": self.burn_time
        }

    @property
    def first_point(self) -> tuple[float, float]:
        """time [s], thrust [N]"""
        return float(self.time[0]), float(self.thrust[-1])

    @property
    def last_point(self) -> tuple[float, float]:
        """time [s], thrust [N]"""
        return float(self.time[-1]), float(self.thrust[-1])

    @property
    def first_time(self) -> float:
        """time [s]"""
        return float(self.time[0])

    @property
    def first_thrust(self) -> float:
        """thrust [N]"""
        return float(self.thrust[0])

    @property
    def last_time(self) -> float:
        """time [s]"""
        return float(self.time[-1])

    @property
    def last_thrust(self) -> float:
        """thrust [N]"""
        return float(self.thrust[-1])

    @cached_property
    def burn_time(self) -> float:
        if len(self) < 2:
            return 0

        time = self.time[1:]
        thrust = self.thrust[1:]

        idx = np.flatnonzero(thrust <= 0)
        t_end = time[idx[0]] if idx.size else time[-1]

        return float(t_end)

    @property
    def max_thrust(self) -> float:
        return float(np.max(self.thrust))

    @property
    def average_thrust(self) -> float:
        return self.total_impulse / (self.burn_time - self.first_time)

    @cached_property
    def cumulative_total_impulse(self) -> np.ndarray:
        time = self.time
        thrust = self.thrust

        dt = np.diff(time)

        # trapezoidal integral over each interval
        d_impulse = 0.5 * (thrust[:-1] + thrust[1:]) * dt

        return np.concatenate((
            np.array([0.0]),
            np.cumsum(d_impulse)
        ))

    @property
    def total_impulse(self) -> float:
        return self.cumulative_total_impulse[-1]

    @property
    def motor_class(self) -> str:
        """UKRA/NAR/etc rocket motor impulse class, e.g. 'F'"""
        return total_impulse_to_motor_class(self.total_impulse)

    @property
    def motor_name(self) -> str:
        """standard motor name defined by motor class and average thrust, e.g. 'F32'"""
        return f"{self.motor_class}{self.average_thrust:.0f}"

    @property
    def thrust_source(self):
        return np.column_stack((self.time, self.thrust))

    def thrust_at_time(self, t: float) -> float:
        if t < 0 or t > self.burn_time:
            raise ValueError(
                f"out of bounds: 0 < {t} < {self.burn_time:.2f}")

        return np.interp(
            x=t,
            xp=self.time,
            fp=self.thrust
        )

    def time_at_cumulative_impulse(self, total_impulse: float) -> float:
        if total_impulse < 0:
            raise ValueError("total impulse < 0")

        if total_impulse > self.total_impulse + 1e-3:
            raise ValueError(
                f"provided total impulse ({total_impulse}) > "
                f"motor total impulse + 1e-3 ({self.total_impulse + 1e-3})"
            )

        times = self.time
        thrusts = self.thrust

        cumulative_impulse = self.cumulative_total_impulse

        if total_impulse > self.total_impulse:
            return self.last_time

        # Find the interval containing the requested impulse.
        i = np.searchsorted(
            cumulative_impulse,
            total_impulse,
            side="right",
        ) - 1

        # Handle final point.
        if i == len(times) - 1:
            return float(times[-1])

        t0 = times[i]
        t1 = times[i + 1]

        F0 = thrusts[i]
        F1 = thrusts[i + 1]

        dt = t1 - t0
        dI = total_impulse - cumulative_impulse[i]

        # F(t) = F0 + k * (t - t0)
        k = (F1 - F0) / dt

        # Constant thrust over the interval.
        if np.isclose(k, 0.0):
            if np.isclose(F0, 0.0):
                # No impulse is accumulated in this interval.
                return float(t0)

            x = dI / F0

        else:
            # dI = F0*x + 0.5*k*x²
            #
            # Solve for x.
            discriminant = F0**2 + 2.0 * k * dI

            if discriminant < 0:
                raise ValueError(
                    "Negative discriminant while interpolating impulse"
                )

            x = (
                2.0 * dI
                / (F0 + np.sqrt(discriminant))
            )

        return float(t0 + x)

    def prepend(self, time: float, thrust: float) -> "ThrustCurve":
        time = np.atleast_1d(time)  # type: ignore
        thrust = np.atleast_1d(thrust)  # type: ignore
        return ThrustCurve(
            time=np.concatenate((time, self.time)),
            thrust=np.concatenate((thrust, self.thrust)),
        )

    def append(self, time: float | np.ndarray, thrust: float | np.ndarray) -> "ThrustCurve":
        time = np.atleast_1d(time)
        thrust = np.atleast_1d(thrust)
        return ThrustCurve(
            time=np.concatenate((self.time, time)),
            thrust=np.concatenate((self.thrust, thrust))
        )

    def append_discrete_block(self, dtime: float, thrust: float) -> "ThrustCurve":
        """add a fixed `thrust` [N] block to the end of length `dtime` [s]"""
        return self.append(
            time=[self.last_time + 1e-6, self.last_time + dtime],  # type: ignore
            thrust=[thrust, thrust]  # type: ignore
        )

    def append_until_impulse(self, thrust: float, impulse: float) -> "ThrustCurve":
        """add a fixed `thrust` [N] regime to the end of the thrust curve until the `impulse` [Ns] is achieved"""
        temp = self.append(
            time=self.last_time + 1e-6,
            thrust=thrust
        )
        return temp.append(
            time=((impulse - temp.total_impulse) / thrust) + temp.last_time,
            thrust=thrust
        )

    def replace_impulse_region(self, start_impulse: float, end_impulse: float, thrust: float) -> "ThrustCurve":
        """replace the region of the thrust curve between `start_impulse` and `end_impulse` cumulatively with a new region of fixed thrust `thrust`.

        State:
            0: original start point
            1: original end point
            2: new end point

        Args:
            start_impulse: float [Ns]
            end_impulse: float [Ns]
            thrust: float [N]

        Raises:
            ValueError: Incompatible parameters
            ValueError: Out of bounds error

        Returns:
            motor: new mutated `ThrustCurve` object
        """
        if end_impulse <= start_impulse:
            raise ValueError(
                f"Incompatible Parameters: end_impulse ({end_impulse}) <= start_impulse ({start_impulse})"
            )

        if start_impulse < 0 or end_impulse > self.total_impulse + 1e-3:
            raise ValueError(
                f"Out of Bounds: start_impulse ({start_impulse}) < 0 or end_impulse ({end_impulse}) > self.total_impulse + 1e-3 ({self.total_impulse + 1e-3})"
            )

        I0 = start_impulse
        I1 = end_impulse
        dI = I1 - I0

        t0 = self.time_at_cumulative_impulse(start_impulse)
        t1 = self.time_at_cumulative_impulse(end_impulse)
        dt_old = t1 - t0

        T = thrust
        T0 = self.thrust_at_time(t0)
        T1 = self.thrust_at_time(t1)

        # calculate total impulse of connector sections
        dt_slope = 1e-6
        I0_slope = dt_slope * 0.5 * (T + T0)
        I1_slope = dt_slope * 0.5 * (T + T1)

        # determine remaining total impulse and time
        I_remaining = dI - I0_slope - I1_slope
        dt_new = I_remaining / T
        dt = dt_new - dt_old
        t2 = t0 + dt_new

        # construct new curve
        times = self.time
        thrusts = self.thrust

        before = times < t0
        after = times > t1

        before_times = times[before]
        before_thrusts = thrusts[before]
        after_times = times[after] + dt
        after_thrusts = thrusts[after]
        new_region_times = np.array([t0, t0 + dt_slope, t2 - dt_slope, t2])
        new_region_thrusts = np.array([T0, T, T, T1])

        new_times = np.concatenate(
            [before_times, new_region_times, after_times]
        )
        new_thrusts = np.concatenate(
            [before_thrusts, new_region_thrusts, after_thrusts]
        )

        # TODO: Simplify with the above methods

        return ThrustCurve(
            time=new_times,
            thrust=new_thrusts
        )

    def remove_redundant_data_points(self) -> "ThrustCurve":
        """removes redundant data points. This doesn't change the actual thrust curve"""
        thrust = self.thrust

        keep = np.ones(thrust.size, dtype=bool)
        keep[1:-1] = ~((thrust[1:-1] == thrust[:-2])
                       & (thrust[1:-1] == thrust[2:]))

        return ThrustCurve(
            time=self.time[keep],
            thrust=self.thrust[keep]
        )

    def remove_small_gaps(self, threshold: float = 1e-4) -> "ThrustCurve":
        """remove datapoints where both neighbours are within thresholds seconds"""
        if threshold < 0:
            raise ValueError(f"Out of Bounds: threshold ({threshold}) < 0")

        time = self.time
        thrust = self.thrust

        while True:
            if len(time) < 3:
                return self

            dt_previous = np.diff(time[:-1])
            dt_next = np.diff(time[1:])

            remove = ((dt_previous < threshold) & (dt_next < threshold))
            if not np.any(remove):
                return ThrustCurve(
                    time=time,
                    thrust=thrust
                )

            idxs = np.flatnonzero(remove) + 1

            time = np.delete(time, idxs)
            thrust = np.delete(thrust, idxs)

    def round_to_impulse(self, impulse: float) -> "ThrustCurve":
        """round the thrust curve to a specific total impulse by either trimming the end or adding a fixed thrust section"""
        if self.total_impulse < impulse:
            return self.append_until_impulse(self.last_thrust, impulse)

        if np.isclose(self.total_impulse, impulse):
            return self

        time_cut = self.time_at_cumulative_impulse(impulse)

        mask = self.time < time_cut

        times = self.time[mask]
        thrusts = self.thrust[mask]

        thrust_cut = self.thrust_at_time(time_cut)

        return ThrustCurve(
            time=np.append(times, time_cut),
            thrust=np.append(thrusts, thrust_cut)
        )

    def scale(self, *, time_factor: float = 1, thrust_factor: float = 1) -> "ThrustCurve":
        return ThrustCurve(
            time=self.time * time_factor,
            thrust=self.thrust * thrust_factor
        )


@dataclass(frozen=True)
class Motor:
    """all units are SI"""
    category = "motor"

    diameter: float
    length: float
    prop_mass: float
    total_mass: float
    thrust_curve: ThrustCurve = field(default_factory=lambda: ThrustCurve())
    manufacturer: str = "Not Applicable"
    manufacturer_abbreviation: str = "NA"
    custom_name: str | None = None
    delay: Literal["P"] | int = "P"
    grain_number: int = 1

    info: "MotorInfo" = field(init=False)
    plots: "MotorPlot" = field(init=False)

    @classmethod
    def from_fixed_thrust(
        cls,
        thrust: float,
        impulse: float,
        diameter: float,
        length: float,
        prop_mass: float,
        total_mass: float,
        *,
        manufacturer: str = "Not Applicable",
        manufacturer_abbreviation: str = "NA",
        custom_name: str | None = None,
        delay: Literal["P"] | int = "P",
        grain_number: int = 1
    ) -> "Motor":
        return Motor(
            diameter=diameter,
            length=length,
            prop_mass=prop_mass,
            total_mass=total_mass,
            thrust_curve=ThrustCurve().append_until_impulse(
                thrust=thrust,
                impulse=impulse
            ),
            manufacturer=manufacturer,
            manufacturer_abbreviation=manufacturer_abbreviation,
            custom_name=custom_name,
            delay=delay,
            grain_number=grain_number
        )

    def __post_init__(self):
        object.__setattr__(self, "info", MotorInfo(self))
        object.__setattr__(self, "plots", MotorPlot(self))

        if self.thrust_curve.first_time != 0 and self.thrust_curve.first_thrust != 0:
            object.__setattr__(
                self, "thrust_curve", self.thrust_curve.prepend(0, 0)
            )

        object.__setattr__(
            self, "thrust_curve", self.thrust_curve.remove_redundant_data_points().remove_small_gaps())

    @classmethod
    def from_RASP(
        cls,
        filepath: str | pathlib.Path,
        *,
        custom_name: str | None = None,
        grain_number: int = 1
    ) -> "Motor":
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Provided filepath: {filepath} does not exist"
            )

        with filepath.open("r") as f:
            lines = f.readlines()

        try:

            # remove blank lines and comments
            lines = [
                line.strip() for line in lines if line.strip() and not line.strip().startswith(";")
            ]

            header = lines[0].split()

            name = header[0]  # e.g. F32
            diameter = float(header[1]) / 1e3  # [m]
            length = float(header[2]) / 1e3  # [m]

            # e.g. 1-3-5 or P
            delays = header[3]
            if len(delays.split("-")) > 1:
                delay = delays.split("-")[-1]
            else:
                delay = delays
            if delay.lower() == "p":
                delay = "P"
            else:
                try:
                    delay = int(delay)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Error decoding delay, {delay}: {e}")
                    delay = "P"

            prop_mass = float(header[4])  # [kg]
            total_mass = float(header[5])  # [kg]

            manufacturer_abbreviation = header[6]

            time = []
            thrust = []
            for line in lines[1:]:
                if line == ";":
                    break

                values = line.split()

                if len(values) >= 2:
                    time.append(float(values[0]))
                    thrust.append(float(values[1]))

        except Exception as e:
            raise RuntimeError(f"Error decoding {filepath}: {e}")

        return Motor(
            diameter=diameter,
            length=length,
            prop_mass=prop_mass,
            total_mass=total_mass,
            thrust_curve=ThrustCurve(
                time=time,  # type: ignore
                thrust=thrust  # type: ignore
            ),
            manufacturer_abbreviation=manufacturer_abbreviation,
            custom_name=custom_name or name,
            delay=delay,
            grain_number=grain_number
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} thrust_curve={self.thrust_curve} total_mass={self.total_mass:.3f}kg prop_mass={self.prop_mass:.3f} length={self.length:.2f}m diameter={self.diameter*1e3:.1f}mm >"

    @property
    def name(self) -> str:
        if self.custom_name is not None:
            return self.custom_name
        else:
            return self.thrust_curve.motor_name

    def to_RASP_file(self, filepath: str | pathlib.Path) -> None:
        """filepath should end in '.eng'"""
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)

        if filepath.suffix != ".eng":
            raise ValueError(
                f"RASP filepath should end in .eng, provided: {filepath.suffix}"
            )

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # TODO: Ensure no negative thrust (trim as appropriate)
        # TODO: See https://www.thrustcurve.org/info/raspformat.html
        # TODO: See https://www.rocketryforum.com/threads/help-adding-a-eng-file-to-rasaero.66010/
        # 1. ensure final thrust is zero
        # 2. ensure double semi colons (comments on both sides)
        # 3. remove hyphens from motor names

        with filepath.open("w") as f:
            f.write(f"; {self.manufacturer} {self.name}\n")
            f.write(f"; created using software developed by Rufus Clark\n")
            f.write(f"; created {datetime.now()}\n")
            f.write(
                f"{self.name} {self.diameter*1e3:.0f} {self.length*1e3:.0f} {self.delay} "
                f"{self.prop_mass:.4f} {self.total_mass:.4f} {self.manufacturer_abbreviation}\n"
            )

            for time, thrust in zip(self.thrust_curve.time, self.thrust_curve.thrust):
                f.write(f"{time:10.7f} {thrust:9.2f}\n")

            # add trailing thrust if it doesn't exist
            if self.thrust_curve.last_thrust != 0:
                f.write(f"{self.thrust_curve.last_time+1e-6:10.7f} {0:9.2f}\n")

            f.write(";")

        print(f"Saved RASP ENG file to {filepath}")

    def to_SolidMotor(self) -> tuple[rocketpy.SolidMotor, float]:
        total_mass = self.total_mass
        prop_mass = self.prop_mass
        length = self.length
        diameter = self.diameter
        burn_time = self.thrust_curve.burn_time
        grain_number = self.grain_number

        # fixed assumptions
        grain_separation = 2 / 1000

        # calculations
        dry_mass = total_mass - prop_mass
        radius = diameter / 2
        throat_radius = radius * 0.25
        nozzle_radius = radius * 0.8
        grain_initial_inner_radius = radius * 0.25
        grain_outer_radius = radius * 0.75
        length_nozzle = ((nozzle_radius - throat_radius)/np.tan(np.deg2rad(45))) + \
            ((nozzle_radius - throat_radius)/np.tan(np.deg2rad(15)))
        length_grains = length - length_nozzle
        grain_height = length_grains / grain_number - grain_separation
        grain_volume = length_grains * np.pi * \
            (grain_outer_radius**2 - grain_initial_inner_radius**2)
        grain_density = prop_mass / grain_volume
        grain_center_of_mass_position = 0.5 * length_grains

        Ixx = 0.5 * dry_mass * grain_outer_radius**2
        Iyy = Izz = (1/12) * dry_mass * \
            (3*grain_outer_radius**2 + length_grains**2)
        centre_of_dry_mass_position = (0.5 * length) - length_nozzle

        motor = rocketpy.SolidMotor(
            thrust_source=self.thrust_curve.thrust_source,
            dry_mass=dry_mass,
            dry_inertia=(Izz, Iyy, Ixx),
            nozzle_radius=nozzle_radius,
            grain_number=grain_number,
            grain_density=grain_density,
            grain_outer_radius=grain_outer_radius,
            grain_initial_inner_radius=grain_initial_inner_radius,
            grain_initial_height=grain_height,
            grain_separation=grain_separation,
            grains_center_of_mass_position=grain_center_of_mass_position,
            center_of_dry_mass_position=centre_of_dry_mass_position,
            throat_radius=throat_radius,
            burn_time=burn_time,
            nozzle_position=-length_nozzle,
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )

        position_offset = length_nozzle

        # create motor object
        return motor, position_offset

    def append_until_impulse(self, thrust: float, impulse: float) -> "Motor":
        """add a fixed thrust [N] regime to the end of the thrust curve until the impulse [Ns] is achieved"""
        return replace(
            self,
            thrust_curve=self.thrust_curve.append_until_impulse(
                thrust, impulse)
        )

    def append_discrete_block(self, dtime: float, thrust: float) -> "Motor":
        """add a fixed thrust [N] block to the end of length dtime [s]"""
        return replace(
            self,
            thrust_curve=self.thrust_curve.append_discrete_block(dtime, thrust)
        )

    def replace_impulse_region(self, start_impulse: float, end_impulse: float, thrust: float) -> "Motor":
        """replace the region of the thrust curve between `start_impulse` and `end_impulse` cumulatively with a new region of fixed thrust `thrust`.

        State:
            0: original start point
            1: original end point
            2: new end point

        Args:
            start_impulse: float [Ns]
            end_impulse: float [Ns]
            thrust: float [N]

        Raises:
            ValueError: Incompatible parameters
            ValueError: Out of bounds error

        Returns:
            motor: new mutated `Motor` object
        """

        return replace(
            self,
            thrust_curve=self.thrust_curve.replace_impulse_region(
                start_impulse=start_impulse,
                end_impulse=end_impulse,
                thrust=thrust
            )
        )

    def round_to_impulse(self, impulse: float) -> "Motor":
        """round the thrust curve to a specific total impulse by either trimming the end or adding a fixed thrust section"""
        return replace(
            self,
            thrust_curve=self.thrust_curve.round_to_impulse(impulse)
        )


class MotorInfo(InfoBase[Motor]):

    @scientific_table()
    def overview(self):
        model = self.model

        import pandas as pd

        df = pd.DataFrame({
            "Name": model.name,
            "Max Thrust [N]": model.thrust_curve.max_thrust,
            "Average Thrust [N]": model.thrust_curve.average_thrust,
            "Total Impulse [Ns]": model.thrust_curve.total_impulse,
            "Burn Time [s]": model.thrust_curve.burn_time
        }, index=[0])

        return df


class MotorPlot(PlotBase[Motor]):

    @scientific_plot(figsize=(6, 2), outside_legend="right", auto_limits=None)
    def draw(self, ax: Axes):
        model = self.model

        from rocketpy.plots.solid_motor_plots import _SolidMotorPlots

        solidMotor, _ = model.to_SolidMotor()
        motor = _SolidMotorPlots(solidMotor)

        nozzle = motor._generate_nozzle(
            translate=(motor.motor.nozzle_position, 0), csys=motor.motor._csys
        )
        chamber = motor._generate_combustion_chamber(
            translate=(motor.motor.grains_center_of_mass_position, 0)
        )
        grains = motor._generate_grains(
            translate=(motor.motor.grains_center_of_mass_position, 0)
        )
        outline = motor._generate_motor_region(
            list_of_patches=[nozzle, chamber, *grains]
        )

        ax.add_patch(outline)
        ax.add_patch(chamber)
        for grain in grains:
            ax.add_patch(grain)
        ax.add_patch(nozzle)

        motor._draw_center_of_mass(ax)
        ax.set_aspect("equal")
        ax.set_ymargin(0.8)

        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Radius [m]")

    @scientific_plot(auto_limits=None)
    def thrust_curve(self, ax: Axes):
        model = self.model

        ax.plot(model.thrust_curve.time, model.thrust_curve.thrust)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Thrust [N]")


AEROTECH_RMS_GRAIN_COUNT = {
    # 18 mm
    "RMS-18/20": 1,

    # 24 mm Hobby Line
    "RMS-24/20-40": 1,
    "RMS-24/40": 1,
    "RMS-24/60": 1,

    # 29 mm
    "RMS-29/40-120": 1,
    "RMS-29/60": 1,
    "RMS-29/100": 1,
    "RMS-29/120": 2,
    "RMS-29/180": 3,
    "RMS-29/240": 4,
    "RMS-29/360": 6,

    # 38 mm
    "RMS-38/120": 1,
    "RMS-38/240": 2,
    "RMS-38/360": 3,
    "RMS-38/480": 4,
    "RMS-38/600": 5,
    "RMS-38/720": 6,
    "RMS-38/1080": 9,

    # 54 mm
    "RMS-54/426": 1,
    "RMS-54/852": 2,
    "RMS-54/1280": 3,
    "RMS-54/1706": 4,
    "RMS-54/2560": 6,

    # 75 mm
    "RMS-75/1280": 1,
    "RMS-75/2560": 2,
    "RMS-75/3840": 3,
    "RMS-75/5120": 4,
    "RMS-75/6400": 5,
    "RMS-75/7680": 6,

    # 98 mm
    "RMS-98/2560": 1,
    "RMS-98/5120": 2,
    "RMS-98/7680": 3,
    "RMS-98/10240": 4,
    "RMS-98/15360": 6,
}

MOTOR_MANUFACTURERS = Literal["AeroTech", "Cesaroni", "Loki Research", "Estes"]
IMPULSE_CLASSES = Literal[
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"
]


@dataclass
class ThrustCurveAPIResult:
    motorId: str
    manufacturer: str
    manufacturerAbbrev: str
    designation: str
    commonName: str
    impulseClass: str
    diameter: float
    length: float
    type: str
    certOrg: str
    avgThrustN: float
    totImpulseNs: float
    burnTimeS: float
    dataFiles: int
    totalWeightG: float
    propWeightG: float
    delays: str
    delayAdjustable: bool
    availability: str
    source_url: str
    maxThrustN: float = 0
    propInfo: str = ""
    caseInfo: str = ""
    thrustCurvePath: str = "TBD"

    @classmethod
    def from_dict(cls, data: dict):
        field_names = {f.name for f in fields(cls)}
        return cls(**{
            k: v for k, v in data.items() if k in field_names
        })

    def __str__(self) -> str:
        return f"{self.manufacturerAbbrev}-{self.commonName}"

    @property
    def grain_number(self) -> int:
        if self.caseInfo == "":
            return 1
        if self.manufacturer == "Cesaroni Technology":
            caseInfo = self.caseInfo
            if caseInfo.endswith("XL"):
                try:
                    return int(caseInfo[0])
                except:
                    return 1
            else:
                return int(caseInfo.split("-")[1].rstrip("G"))
        if self.manufacturer == "AeroTech":
            return AEROTECH_RMS_GRAIN_COUNT.get(self.caseInfo, 1)
        else:
            import warnings
            warnings.warn(
                f"Grain number not implemented for manufacturer: {self.manufacturer} and caseInfo: {self.caseInfo}. Assuming grain_number=1"
            )
            return 1

    def to_motor(self) -> Motor:
        return Motor.from_RASP(
            filepath=self.thrustCurvePath,
            grain_number=self.grain_number
        )


class MotorSearch:
    def __init__(self, *, _suppress_prints: bool = False):
        self._suppress_prints = _suppress_prints
        path = self.cache_path
        (path / "rasp").mkdir(parents=True, exist_ok=True)
        (path / "search").mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> pathlib.Path:
        return pathlib.Path() / "data" / "motor"

    def search_cache_path(self, payload: dict) -> pathlib.Path:
        copy = dict(payload)
        copy.pop("availability", "")
        copy.pop("maxResults", "")
        (k, v), = copy.items()
        return self.cache_path / "search" / f"{k}={v}.json"

    def RASP_ENG_path(self, motor: ThrustCurveAPIResult) -> pathlib.Path:
        return self.cache_path / "rasp" / f"{motor.manufacturerAbbrev}-{motor.commonName}.eng"

    def update_search_cache(self, search_path: pathlib.Path, *motors: ThrustCurveAPIResult) -> None:
        with search_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            file_motors = [ThrustCurveAPIResult(**motor) for motor in data]

        existing = {
            motor.motorId: motor for motor in file_motors
        }

        for motor in motors:
            existing[motor.motorId] = motor

        with search_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(motor) for motor in existing.values()], f)

    def search_api(self, motor_name: str = "", *, manufacturer: str = "", impulseClass: str = "") -> tuple[list[ThrustCurveAPIResult], pathlib.Path]:

        if impulseClass:
            payload = {
                "impulseClass": impulseClass,
                "availability": "all",
                "maxResults": 9999
            }
        elif manufacturer:
            payload = {
                "manufacturer": manufacturer,
                "availability": "all",
            }
        elif motor_name:
            payload = {
                "commonName": motor_name,
                "availability": "all",
            }
        else:
            raise ValueError()

        motors: list[ThrustCurveAPIResult] = []

        search_path = self.search_cache_path(payload)
        if search_path.exists():
            with search_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                motors = [ThrustCurveAPIResult(**motor) for motor in data]

        else:
            if not self._suppress_prints:
                print(f"Searching thrustcurve.org for {payload}")

            url = "https://www.thrustcurve.org/api/v1/search.json"
            response = requests.post(url, json=payload)

            response.raise_for_status()

            data = response.json()

            if not self._suppress_prints:
                print(f"Search matches: {data.get("matches", 0)}")

            for result in data["results"]:
                try:
                    motor = ThrustCurveAPIResult.from_dict(result)
                except TypeError as e:
                    if not self._suppress_prints:
                        print(
                            f"Skipping {result["manufacturer"]}-{result["commonName"]} due to {e}"
                        )
                    continue

                motors.append(motor)

            search_path.parent.mkdir(parents=True, exist_ok=True)
            with search_path.open("w", encoding="utf-8") as f:
                json.dump([asdict(motor) for motor in motors], f)

        if manufacturer:
            motors = [
                motor for motor in motors if manufacturer in motor.manufacturerAbbrev
            ]
        if motor_name:
            motors = [
                motor for motor in motors if motor_name in motor.commonName
            ]

        return (motors, search_path)

    def download_api(self, motors: list[ThrustCurveAPIResult], search_path: pathlib.Path) -> list[ThrustCurveAPIResult]:
        out_motors: list[ThrustCurveAPIResult] = []

        TBD_flag = False

        for motor in motors:
            match motor.thrustCurvePath:
                case "TBD":
                    TBD_flag = True

                    RASP_ENG_path = self.RASP_ENG_path(motor)

                    if RASP_ENG_path.exists():
                        RASP_ENG = RASP_ENG_path.read_text(encoding="utf-8")

                    else:

                        url = "https://www.thrustcurve.org/api/v1/download.json"
                        payload = {
                            "motorIds": [motor.motorId],
                            "format": "RASP",
                            "data": "file",
                        }
                        response = requests.post(url, json=payload)

                        response.raise_for_status()

                        data = response.json()
                        results = data["results"]

                        if not results:
                            if not self._suppress_prints:
                                print(
                                    f"No thrust curve data for {motor}"
                                )
                                motor.thrustCurvePath = "MISSING"
                            continue

                        RASP_ENG = base64.b64decode(
                            data["results"][0]["data"]).decode("utf-8").replace("\r\n", "\n")

                        RASP_ENG_path.parent.mkdir(parents=True, exist_ok=True)
                        RASP_ENG_path.write_text(RASP_ENG, encoding="utf-8")

                        if not self._suppress_prints:
                            print(
                                f"Downloaded thrust curve for: {motor}"
                            )

                    motor.thrustCurvePath = str(RASP_ENG_path)
                    # motor._thrust_data = motor.read_thrust_curve()

                    out_motors.append(motor)

                case "MISSING":
                    pass
                case _:
                    out_motors.append(motor)

        # update json
        if TBD_flag:
            self.update_search_cache(search_path, *motors)

        return out_motors

    def get_motors(self, name: str = "", *, manufacturer: MOTOR_MANUFACTURERS | str = "", impulseClass: IMPULSE_CLASSES | str = "") -> list[ThrustCurveAPIResult]:

        motors, search_path = self.search_api(
            motor_name=name,
            manufacturer=manufacturer,  # type: ignore
            impulseClass=impulseClass
        )

        motors = self.download_api(
            motors=motors,
            search_path=search_path
        )

        return motors

    def get_motor(self, name: str, *, manufacturer: MOTOR_MANUFACTURERS | str = "", impulseClass: IMPULSE_CLASSES | str = "") -> ThrustCurveAPIResult:
        motors = self.get_motors(
            name=name,
            manufacturer=manufacturer,
            impulseClass=impulseClass
        )

        match len(motors):
            case 0:
                payload = {
                    "commonName": name,
                    "manufacturerAbbrev": manufacturer,
                    "impulseClass": impulseClass
                }
                raise ValueError(
                    f"No motors results returned for '{payload}'")
            case 1:
                motor = motors[0]
            case _:
                for i, motor in enumerate(motors):
                    print(f"[{i}] {motor}")

                choice = int(input("Select an option: "))

                motor = motors[choice]

        print(f"Selected: {motor}")

        return motor
