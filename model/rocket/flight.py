from typing import Literal, Iterator, Any, Callable
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import numpy as np
import rocketpy

from model.rocket.rocket import FlightState, RocketState, LauncherState, EnvironmentState, FlightGenerator
from model.rocket.motor import Motor
from model.util.output import *


class SimulationFailedError(RuntimeError):
    ...


class Flight:
    category = "flight"

    def __init__(
            self,
            flight_state: FlightState | FlightGenerator,
            *,
            name: str | None = None,
            verbose: bool = False,
            terminate_on_apogee: bool = False,
            max_time_step: float | None = None,
            max_time: int | None = None,
    ):
        self.plots = FlightPlots(self)  # type: ignore
        self.info = FlightInfo(self)

        if isinstance(flight_state, FlightGenerator):
            flight_state = flight_state.ideal()

        self.flight_state = flight_state

        rocket = flight_state.rocket
        env = flight_state.env
        launcher = flight_state.launcher

        self.name = name or rocket.name
        self.verbose = verbose
        self.terminate_on_apogee = terminate_on_apogee

        self.rocketpy_flight = self.__simulate(
            rocket=rocket,
            env=env,
            launcher=launcher,
            max_time=max_time,
            max_time_step=max_time_step
        )

    def __simulate(
        self,
        rocket: RocketState,
        env: EnvironmentState,
        launcher: LauncherState,
        *,
        max_time: int | None = None,
        max_time_step: float | None = None,
        _n_retry: int = 0
    ) -> rocketpy.Flight:
        small_rocket = self.total_impulse < 64 or self.motor_burnout_time < 1.2

        max_time_step = max_time_step or (0.01 if small_rocket else np.inf)
        max_time = max_time or (300 if small_rocket else 1200)

        flight = rocketpy.Flight(
            rocket=rocket.make(),
            environment=env.make(),
            rail_length=launcher.length,
            inclination=launcher.inclination,
            heading=launcher.heading,
            max_time_step=max_time_step,
            max_time=max_time,
            time_overshoot=True,
            verbose=self.verbose,
            terminate_on_apogee=self.terminate_on_apogee,
        )

        if flight.time[-1] == max_time:
            if _n_retry > 1:
                raise SimulationFailedError(
                    f"Simulation Time = Maximum Simulation Time = {max_time}s with max_time_step = {max_time_step}. Consider decreasing the max_time_step."
                )
            else:
                new_max_time = [1200, 2400][_n_retry]
                new_max_time_step = [0.01, 0.001][_n_retry]

                import warnings
                warnings.warn(
                    f"Retry={_n_retry}. Simulation Time = Maximum Simulation Time = {max_time}s. Consider decreasing the max_time_step. Retrying with max_time_step={new_max_time_step} and max_time={new_max_time}"
                )

                return self.__simulate(
                    rocket=rocket,
                    env=env,
                    launcher=launcher,
                    max_time=new_max_time,
                    max_time_step=new_max_time_step,
                    _n_retry=_n_retry + 1
                )

        return flight

    @property
    def launch_time(self) -> float:
        return 0.0

    @property
    def out_of_rail_time(self) -> float:
        return self.rocketpy_flight.out_of_rail_time

    @property
    def motor(self) -> Motor:
        if self.flight_state.rocket.motor is not None and self.flight_state.rocket.motor.motor is not None:
            return self.flight_state.rocket.motor.motor
        else:
            raise ValueError("Value Error: No motor was added to the rocket")

    @property
    def motor_burnout_time(self) -> float:
        return self.motor.thrust_curve.burn_time

    @property
    def total_impulse(self) -> float:
        return self.motor.thrust_curve.total_impulse

    @property
    def apogee_time(self) -> float:
        return self.rocketpy_flight.apogee_time

    @property
    def landing_time(self) -> float:
        return self.rocketpy_flight.time[-1]

    @property
    def coast_duration(self) -> float:
        return self.ascent_duration - self.motor_burnout_time

    @property
    def ascent_duration(self) -> float:
        return self.apogee_time

    @property
    def descent_duration(self) -> float:
        return self.flight_duration - self.ascent_duration

    @property
    def flight_duration(self) -> float:
        return self.landing_time

    @property
    def max_dynamic_pressure_time(self) -> float:
        return self.rocketpy_flight.max_dynamic_pressure_time

    @property
    def parachute_events(self):
        return self.rocketpy_flight.parachute_events

    def flight_events(self) -> Iterator[tuple[float, str, Any]]:
        times = [
            0.0,
            self.out_of_rail_time,
            self.motor_burnout_time,
            self.max_dynamic_pressure_time,
            self.apogee_time,
            *[t_p for t_p, _ in self.parachute_events],
            self.landing_time
        ]
        labels = [
            "Launch Point",
            "Out of Rail",
            "Motor Burnout",
            "Maximum Dynamic Pressure",
            "Apogee",
            *[f"{p.name} Deployed" for _, p in self.parachute_events],
            "Landing Point"
        ]
        colours = [colour_palette[1:][i % (len(colour_palette) - 1)]
                   for i, _ in enumerate(times)]

        return zip(times, labels, colours)

    def thrust_energy(self) -> float:
        tf = self.rocketpy_flight
        t = tf.time
        t = t[t <= tf.rocket.motor.burn_out_time]

        return cumulative_trapezoid(
            tf.thrust_power(t), t, initial=0  # type: ignore
        )[-1]

    def drag_work(self) -> float:
        tf = self.rocketpy_flight
        t = tf.time
        t = t[t <= tf.apogee_time]

        return cumulative_trapezoid(
            tf.drag_power(t), t, initial=0  # type: ignore
        )[-1]

    def gravity_work(self) -> float:
        tf = self.rocketpy_flight

        t = tf.time
        t = t[t <= tf.apogee_time]

        m = tf.rocket.total_mass(t)
        g = tf.env.gravity(t)
        vz = tf.vz(t)  # type: ignore

        g_power = m * g * vz

        return -cumulative_trapezoid(g_power, t, initial=0)[-1]

    def potential_energy_change(self) -> float:
        tf = self.rocketpy_flight

        return (
            tf.potential_energy(tf.apogee_time)  # type: ignore
            - tf.potential_energy(0.0)  # type: ignore
        )

    def kinetic_energy_change(self) -> float:
        tf = self.rocketpy_flight

        return (
            tf.kinetic_energy(tf.apogee_time)  # type: ignore
            - tf.potential_energy(0.0)  # type: ignore
        )

    def residual_energy(self) -> float:
        return (
            self.thrust_energy()
            + self.drag_work()
            + self.gravity_work()
            - self.kinetic_energy_change()
        )

    def system_efficiency(self) -> float:
        return abs(self.potential_energy_change() / self.thrust_energy()) * 100

    def energies(self) -> Iterator[tuple[float, str]]:
        energies = [
            self.thrust_energy(),
            self.drag_work(),
            self.gravity_work(),
            self.residual_energy(),
            self.kinetic_energy_change(),
            self.potential_energy_change()
        ]

        labels = [
            "Thrust Work",
            "Drag Work",
            "Gravity Work",
            "Other / Residual Work",
            "Apogee Kinetic Energy",
            "Ideal Performance Energy"
        ]

        return zip(energies, labels)

    def simulation_failed(self) -> bool:
        try:
            self.rocketpy_flight.altitude(0.0)  # type: ignore
            self.rocketpy_flight.thrust_power(0.0)  # type: ignore
            return False
        except:
            return True

    @property
    def apogee(self) -> float:
        """apogee altitude [m]"""
        return self.rocketpy_flight.altitude(self.apogee_time)  # type: ignore

    def max_thrust(self) -> float:
        """maximum thrust [N]"""
        flight = self.rocketpy_flight
        t = flight.time
        return float(flight.net_thrust(t).max())  # type: ignore

    def average_thrust(self) -> float:
        """average thrust [N]"""
        return self.total_impulse / self.motor_burnout_time

    @property
    def out_of_rail_velocity(self) -> float:
        """out of rail velocity [m/s]"""
        return self.rocketpy_flight.out_of_rail_velocity

    @property
    def x_landing_position(self) -> float:
        """positive east of launch site"""
        return self.rocketpy_flight.x_impact

    @property
    def y_landing_position(self) -> float:
        """positive north of launch site"""
        return self.rocketpy_flight.y_impact


class FlightPlots(PlotBase[Flight]):

    @scientific_plot(auto_limits=None, outside_legend="right", figsize=(6, 1.5))
    def draw(self, ax: Axes, vis_args=None, plane="xz") -> str:

        rocket_plots = self.model.flight_state.rocket.plots
        plot_func = rocket_plots.draw.func
        assert plot_func is not None
        plot_func(rocket_plots, ax, vis_args=vis_args, plane=plane)

        return self.model.name

    def _plot_events(self, ax: Axes, style: Literal["marker", "axvline"], y: Callable[..., np.ndarray] | None = None, t_lims: tuple[float, float] | None = None) -> None:
        if style == "marker":
            if y is None:
                raise ValueError(f"y must be provided when style='marker'")

        for t, label, colour in self.model.flight_events():

            if t_lims:
                if t < t_lims[0] or t > t_lims[1]:
                    continue

            label = f"{label} ({t:.1f}s)"

            match style:
                case "marker":
                    ax.plot(
                        t,
                        y(t),  # type: ignore
                        marker="x",
                        label=label,
                        color=colour
                    )
                case "axvline":
                    ax.axvline(
                        x=t,
                        label=label,
                        color=colour,
                        alpha=0.6
                    )
                case _:
                    raise ValueError(f"{style} is not supported")

        ax.legend()

    @scientific_plot(auto_limits=None)
    def altitude(self, ax: Axes, event_labels: Literal["marker", "axvline"] | None = "axvline"):
        tf = self.model.rocketpy_flight
        x = tf.time
        y = tf.altitude

        ax.plot(x, y(x) / 1e3)  # type: ignore

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Altitude [km]")

        if event_labels:
            self._plot_events(ax, style=event_labels, y=y)

        return self.model.name

    @scientific_plot(auto_limits=None)
    def total_velocity(self, ax: Axes, event_labels: Literal["marker", "axvline"] | None = "axvline"):
        tf = self.model.rocketpy_flight
        x = tf.time
        y = tf.vz_body_frame

        ax.plot(x, y(x))  # type: ignore

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")

        if event_labels:
            self._plot_events(ax, style=event_labels, y=y)

        return self.model.name

    @scientific_plot(auto_limits=None)
    def mach(self, ax: Axes, event_labels: Literal["marker", "axvline"] | None = "axvline"):
        tf = self.model.rocketpy_flight
        x = tf.time
        y = tf.mach_number

        ax.plot(x, y(x))  # type: ignore

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mach")

        if event_labels:
            self._plot_events(ax, style=event_labels, y=y)

        return self.model.name

    @scientific_plot(auto_limits=None)
    def total_acceleration(self, ax: Axes, event_labels: Literal["marker", "axvline"] | None = "axvline"):
        tf = self.model.rocketpy_flight
        x = tf.time
        y = tf.az_body_frame

        ax.plot(x, y(x) / 9.81)  # type: ignore

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Acceleration [g]")

        if event_labels:
            self._plot_events(ax, style=event_labels, y=y)

        return self.model.name

    @scientific_plot(auto_limits=None)
    def dynamic_pressure(self, ax: Axes, event_labels: Literal["marker", "axvline"] | None = "axvline"):
        tf = self.model.rocketpy_flight
        x = tf.time
        y = tf.dynamic_pressure

        ax.plot(x, y(x) / 1e3)  # type: ignore

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Dynamic Pressure [kPa]")

        if event_labels:
            self._plot_events(ax, style=event_labels, y=y)

        return self.model.name

    @scientific_compose()
    def overview(self, event_labels: Literal["marker", "axvline"] | None = "axvline"):
        plots = [
            self.altitude.with_args(event_labels=event_labels),
            self.total_velocity.with_args(event_labels=event_labels),
            self.mach.with_args(event_labels=event_labels),
            self.total_acceleration.with_args(event_labels=event_labels)
        ]

        return Composition(
            plots=plots,
            figsize=(6, len(plots)*2),
            sharex=True,
            outside_legend="lower",
            outside_legend_cols=2
        )

    @scientific_plot(auto_limits=None)
    def altitude_vs_distance_diagram(self, ax: Axes, event_labels: Literal["axvline"] | None = "axvline"):
        tf = self.model.rocketpy_flight
        times = tf.time

        ax.plot(tf.y(times), tf.z(times))  # type: ignore
        ax.set_xlabel("Horizontal Distance")
        ax.set_ylabel("Altitude")

        if event_labels:
            self._plot_events(ax, style=event_labels, y=tf.z)

        ax.legend()
        ax.grid(False)

        # remove axis numbers and ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # add arrows
        from matplotlib.patches import FancyArrowPatch

        ascent_time = self.model.rocketpy_flight.apogee_time
        descent_time = self.model.rocketpy_flight.time[-1] - ascent_time
        arrow_times = [ascent_time * 0.35, ascent_time + descent_time * 0.4]

        for t in arrow_times:
            x = tf.y
            y = tf.z

            arrow = FancyArrowPatch(
                (x(t), y(t)),  # type: ignore
                (x(t+0.1), y(t+0.1)),  # type: ignore
                arrowstyle='->',
                mutation_scale=20,
                # linewidth=2,
                color=colour_palette[0]
            )
            ax.add_patch(arrow)

        return self.model.name

    @scientific_plot(auto_limits=None)
    def altitude_vs_time_diagram(self, ax: Axes, event_labels: Literal["marker", "axvline"] | None = "marker"):
        tf = self.model.rocketpy_flight
        times = tf.time

        ax.plot(times, tf.z(times))  # type: ignore
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Altitude [m]")

        if event_labels:
            self._plot_events(ax, style=event_labels, y=tf.z)

        ax.legend()
        ax.grid(False)

        # remove axis numbers and ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # add arrows
        from matplotlib.patches import FancyArrowPatch

        ascent_time = self.model.rocketpy_flight.apogee_time
        descent_time = self.model.rocketpy_flight.time[-1] - ascent_time
        arrow_times = [ascent_time * 0.35, ascent_time + descent_time * 0.4]

        for t in arrow_times:
            y = tf.z

            arrow = FancyArrowPatch(
                (t, y(t)),  # type: ignore
                (t+0.1, y(t+0.1)),  # type: ignore
                arrowstyle='->',
                mutation_scale=20,
                # linewidth=2,
                color=colour_palette[0]
            )
            ax.add_patch(arrow)

        return self.model.name

    @scientific_plot(auto_limits=None)
    def flight_forces(self, ax: Axes, ascent_only: bool = False, event_labels: Literal["axvline"] | None = None) -> str:
        """plots the rockets body forces (these are in the body axis reference frame). i.e. the rocket may not be pointing upwards, some energy may be 'wasted'

        Approximate resultant assuming almost vertical ascent"""
        tf = self.model.rocketpy_flight

        t = tf.time
        if ascent_only:
            t_apogee = tf.apogee_time
            t = t[t < t_apogee]

        thrust = tf.net_thrust(t)  # type: ignore
        drag = -tf.aerodynamic_drag(t)  # type: ignore
        lift = tf.aerodynamic_lift(t)  # type: ignore
        weight = -tf.rocket.total_mass(t) * tf.env.gravity(t)

        resultant = thrust + drag + weight

        # !: Along the rockets body axis (not vertical)

        ax.plot(t, thrust, label="Thrust")
        ax.plot(t, drag, label="Drag")
        ax.plot(t, lift, label="Lift")
        ax.plot(t, weight, label="Weight")
        ax.plot(t, resultant, label="Resultant")

        if event_labels:
            self._plot_events(ax, style=event_labels)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Force [N]")
        ax.legend()

        return f"{self.model.name}{"-ascent-only" if ascent_only else ""}"

    @scientific_plot(auto_limits=None)
    def landing_dispersion(self, ax: Axes, width: float = 1000):
        model = self.model

        ax.scatter(0, 0, marker="*", color=colour_palette[0])

        ax.set_aspect("equal", adjustable='box')
        ax.scatter(
            model.y_landing_position,
            model.x_landing_position,
            marker="x",
            color=colour_palette[1]
        )

        ax.set_xlim(-0.5*width, 0.5*width)
        ax.set_ylim(-0.5*width, 0.5*width)

    @scientific_plot()
    def rail_button_normal_forces(self, ax: Axes, rail_only: bool = True):
        model = self.model

        time = model.rocketpy_flight.time

        if rail_only:
            time = time[time < model.rocketpy_flight.out_of_rail_time]

        ax.plot(time, model.rocketpy_flight.rail_button1_normal_force(
            time), label="Forward")  # type: ignore
        ax.plot(time, model.rocketpy_flight.rail_button2_normal_force(
            time), label="Aft")  # type: ignore

        ax.legend()

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Normal Force [N]")

    @scientific_plot()
    def rail_button_shear_forces(self, ax: Axes, rail_only: bool = True):
        model = self.model

        time = model.rocketpy_flight.time

        if rail_only:
            time = time[time < model.rocketpy_flight.out_of_rail_time]

        ax.plot(time, model.rocketpy_flight.rail_button1_shear_force(
            time), label="Forward")  # type: ignore
        ax.plot(time, model.rocketpy_flight.rail_button2_shear_force(
            time), label="Aft")  # type: ignore

        ax.legend()

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Shear Force [N]")

    @scientific_plot()
    def rail_button_bending_moment(self, ax: Axes, rail_only: bool = True):
        model = self.model

        height = model.flight_state.rocket.rail_buttons.height  # type: ignore
        assert height

        time = model.rocketpy_flight.time

        if rail_only:
            time = time[time < model.rocketpy_flight.out_of_rail_time]

        ax.plot(time, model.rocketpy_flight.rail_button1_normal_force(
            time) * height, label="Forward")  # type: ignore
        ax.plot(time, model.rocketpy_flight.rail_button2_normal_force(
            time) * height, label="Aft")  # type: ignore

        ax.legend()

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Bending Moment [Nm]")

    @scientific_compose()
    def rail_button_forces(self, rail_only: bool = True):
        return Composition(
            plots=[
                self.rail_button_normal_forces.with_args(rail_only=rail_only),
                self.rail_button_shear_forces.with_args(rail_only=rail_only),
                self.rail_button_bending_moment.with_args(rail_only=rail_only)
            ],
            figsize=(6, 6),
            sharex=True,
        )

    def trajectory_3d(self):
        # TODO: How can I impose my plotting rules and save files on this?
        # ? Copy the implementation myself
        self.model.rocketpy_flight.plots.trajectory_3d()


class FlightInfo(InfoBase[Flight]):

    @scientific_table()
    def flight_event_summary(self):
        tf = self.model.rocketpy_flight
        times = tf.time

        rows = []
        for t, label, colour in self.model.flight_events():

            a_mask = (times >= t-1) & (times <= t+1)
            a = tf.az_body_frame(times)[a_mask]  # type: ignore

            rows.append({
                "Name": label,
                "Time [s]": t,
                "Altitude [km]": tf.z(t) / 1000,  # type: ignore
                "Velocity (Up) [m/s]": tf.vz_body_frame(t),  # type: ignore
                "Acceleration (Up) [g]": (
                    tf.az_body_frame(t) / 9.81),  # type: ignore
                "Acceleration (+/- 1 s) (Up) [g]": a[np.argmax(np.abs(a))] / 9.81
            })

        df = pd.DataFrame(rows)

        return df

    @scientific_table()
    def flight_energies(self):
        E_T = self.model.thrust_energy()

        rows = []

        for energy, label in self.model.energies():
            rows.append({
                "Name": label,
                "Energy [MJ]": energy / 1e6,
                r"% of Thrust": (energy / E_T) * 100
            })

        df = pd.DataFrame(rows)

        return df

    @scientific_table()
    def overview(self):
        model = self.model

        df = pd.DataFrame({
            "Name": model.name,
            "Altitude [km]": model.apogee / 1e3,
            "Maximum Mach": model.rocketpy_flight.max_mach_number,
            "System Efficiency [%]": model.system_efficiency(),
            "Total Impulse [Ns]": model.total_impulse,
            "Maximum Thrust [N]": model.max_thrust(),
            "Average Thrust [N]": model.average_thrust(),
            "Burn Time [s]": model.motor_burnout_time
        }, index=[0])

        return df
