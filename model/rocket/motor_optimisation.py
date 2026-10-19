from typing import Sequence, overload, Callable
from scipy.optimize import minimize_scalar
from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass
from itertools import pairwise
import warnings
import numpy as np
import copy

from model.rocket.motor import MotorSearch, MOTOR_MANUFACTURERS, IMPULSE_CLASSES, Motor
from model.rocket.rocket import FlightGenerator
from model.util.progress import ProgressBar
from model.util.output.info import Table
from model.rocket.flight import Flight, SimulationFailedError
from model.util.output import *


@dataclass
class Constraints:
    min_thrust: float  # [N]
    max_thrust: float  # [N]
    min_speed_out_of_rail: float  # [m/s]
    total_impulse: float  # [Ns]
    thrust_tolerance: float = 0.1  # [N]


@dataclass
class Assumptions:
    motor_total_mass: float  # [kg]
    motor_prop_mass: float  # [kg]
    motor_diameter: float  # [m]
    motor_length: float  # [m]


@dataclass
class Result:
    success: bool
    thrust: float | None = None
    motor: Motor | None = None
    flight: Flight | None = None


class Results:
    category = "optimisation-results"

    def __init__(self, name: str) -> None:
        self.info = ResultsInfo(self)
        self.plots = ResultsPlots(self)

        self.name = name
        self.n_step = 0
        self.n_pass = 0

        self.results: list[Result] = []

        self.best_results: list[Result] = []

    def add(self, result: Result) -> None:
        self.results.append(result)

    def number_of_success(self) -> float:
        return sum(result.success for result in self.results)

    def is_no_success(self) -> bool:
        return self.number_of_success() == 0

    def next_step(self) -> None:
        if len(self.results) == 0:
            return

        self.n_step += 1
        self.best_results.append(self.best())
        self.results = []

    def next_pass(self) -> None:
        if len(self.results) == 0:
            return

        self.n_pass += 1
        self.next_step()
        self.n_step = 0

    def best(self) -> Result:
        return max(
            (result for result in self.results if result.success),
            key=lambda result: result.flight.apogee  # type: ignore
        )

    def last(self) -> Result:
        return self.results[-1]

    def last_best_result(self) -> Result | None:
        if len(self.best_results) > 0:
            return self.best_results[-1]
        else:
            return None

    @property
    def thrusts_and_apogees(self) -> tuple[np.ndarray, np.ndarray]:
        """returns thrust and apogee points for points where `success=True`

        Returns:
            thrusts: np.ndarray[float] [N]
            apogees: np.ndarray[float] [m]
        """
        points = [
            (result.thrust, result.flight.apogee)
            for result in self.results
            if result.success and result.thrust and result.flight
        ]

        if not points:
            return np.array([]), np.array([])

        thrusts, apogees = np.asarray(points).T

        order = np.argsort(thrusts)

        return thrusts[order], apogees[order]

    def poly(self, deg: int = 3) -> np.polynomial.Polynomial:
        return np.polynomial.Polynomial.fit(*self.thrusts_and_apogees, deg=deg)

    def poly_optimal(self, deg: int = 3) -> tuple[float, float]:
        """returns the optimal thrust to maximise apogee determined mathematically from polynomial fit

        Args:
            deg: polynomial degrees of freedom. Defaults to 3.

        Returns:
            thrust: float [N]
            apogee: float [m]
        """
        thrusts, _ = self.thrusts_and_apogees
        lower, upper = thrusts.min(), thrusts.max()

        poly = self.poly(deg=deg)

        # find real stationary points in range
        stationary = poly.deriv().roots()
        stationary = stationary[np.isreal(stationary)].real
        stationary = stationary[
            (stationary >= lower) &
            (stationary <= upper)
        ]

        # include end points
        candidates = np.r_[lower, stationary, upper]

        # find maximum
        optimal_thrust = candidates[np.argmax(poly(candidates))]
        optimal_apogee = poly(optimal_thrust)

        return optimal_thrust, optimal_apogee

    def poly_interpolate(self, deg: int = 3):
        """returns interpolated thrust and apogee points for points where `success=True` over the entire range of thrust where `success=True`

        Args:
            deg: polynomial degrees of freedom

        Returns:
            thrusts_fit: np.ndarray[float] [N]
            apogees_fit: np.ndarray[float] [m]
        """
        thrusts, _ = self.thrusts_and_apogees

        thrusts_fit = np.linspace(thrusts.min(), thrusts.max(), 50)
        apogees_fit = self.poly(deg=deg)(thrusts_fit)

        return thrusts_fit, apogees_fit


class ResultsInfo(InfoBase[Results]):
    pass


class ResultsPlots(PlotBase[Results]):

    @scientific_plot(auto_limits=None)
    def progress(self, ax: Axes):
        model = self.model

        ax.set_title(
            f"Optimisation Progress (pass={model.n_pass}, step={model.n_step})")

        apogees = np.array([
            result.flight.apogee for result in [*model.best_results, model.best()] if result.flight
        ])
        steps = np.arange(len(apogees))

        ax.plot(steps, apogees / 1e3, marker="x")

        ax.set_ylabel("Altitude [km]")
        ax.set_xlabel("Steps")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        return model.name

    @scientific_plot(auto_limits=None)
    def best_thrust_curve(self, ax: Axes):
        model = self.model

        result = model.best()
        if result.motor is None or result.flight is None:
            raise ValueError()
        motor = result.motor
        flight = result.flight

        thrust_curve_func = motor.plots.thrust_curve.func
        assert thrust_curve_func

        thrust_curve_func(motor.plots, ax)

        ax.set_title(
            f"Thrust Curve (pass={model.n_pass}, step={model.n_step})")

        flight.plots._plot_events(ax, "axvline", t_lims=(
            0, motor.thrust_curve.burn_time))

        return model.name

    @scientific_plot(auto_limits=None)
    def thrust_vs_altitude(self, ax: Axes, degs: list[int] = []):
        model = self.model

        thrusts, apogees = model.thrusts_and_apogees

        ax.set_title(
            f"Altitude Optimisation (pass={model.n_pass}, step={model.n_step})")

        ax.plot(thrusts, apogees / 1e3, marker="x", label="Simulated Data")

        for deg in degs:
            thrusts_fit, apogees_fit = model.poly_interpolate(deg)

            line, = ax.plot(
                thrusts_fit,
                apogees_fit / 1e3,
                label=f"Interpolation (deg={deg})",
                alpha=0.4
            )

            thrust_optimal, apogee_optimal = model.poly_optimal(deg)
            ax.scatter(
                thrust_optimal,
                apogee_optimal / 1e3,
                marker="*",
                label=f"Optimal (deg={deg})",
                color=line.get_color(),
                alpha=0.6
            )

        ax.set_xlabel("Thrust [N]")
        ax.set_ylabel("Apogee [km]")
        ax.legend()

        return model.name


@dataclass
class _Region:
    min: float
    max: float

    @property
    def range(self) -> float:
        return self.max - self.min

    def __contains__(self, x: float) -> bool:
        return x >= self.min and x <= self.max

    def points(self, spacing: float) -> list[float]:
        n = int(self.range // spacing)
        return [
            self.min + i * self.range / (n + 1)
            for i in range(1, n + 1)
        ]


def variable_density_1d_grid(xmin: float, xmax: float, xinterest: Sequence[float], points: int, interest_bias: float = 2) -> np.ndarray:
    total_region = _Region(xmin, xmax)

    spacing = total_region.range / points
    interest_spacing = spacing / interest_bias

    # add interest points and next neighbours
    grid: list[float] = []
    for x in xinterest:
        grid.append(x)

        upper_point = x + interest_spacing
        if upper_point in total_region:
            grid.append(upper_point)

        lower_point = x - interest_spacing
        if lower_point in total_region:
            grid.append(lower_point)
    grid = list(set(grid))
    grid = sorted(grid)

    # add the rest of the points
    regions: list[_Region] = []
    for a, b in pairwise(grid):
        regions.append(_Region(a, b))

    remaining_points = points - len(grid)
    if remaining_points < 0:
        raise ValueError(
            f"not enough points to include all interest points and neighbours: need {len(grid)}, got {points}"
        )

    if remaining_points == 0:
        return np.array(grid)

    # iterate through regions adding points if theirs enough space
    regions = sorted(regions, key=lambda region: region.range, reverse=True)
    while regions:
        remaining_points = points - len(grid)
        regions_range = sum([region.range for region in regions])

        if regions_range <= 0:
            continue

        spacing = regions_range / remaining_points
        if spacing <= 0:
            continue
        region = regions.pop()

        grid.extend(region.points(spacing))
        grid = list(set(grid))

    grid = sorted(grid)
    return np.array(grid)


class MotorOptimisation:
    def __init__(
            self,
            flight: FlightGenerator,
            name: str | None = None,
            motor_position: float | None = None,
    ) -> None:
        if name is None:
            self.name = flight.name
        else:
            self.name = name

        if motor_position is None:
            if flight.rocket.motor is None:
                raise ValueError(
                    f"If motor_position is not provided the flight must include a motor and this motor position will be assumed"
                )
            self.motor_position = flight.rocket.motor.position.ideal()
        else:
            self.motor_position = motor_position

        self.flight = flight

    def _sim(self, motor: Motor, verbose: bool) -> Flight:
        if verbose:
            print(
                f"total_impulse: {motor.thrust_curve.total_impulse:.2f}Ns - last_thrust_point: {motor.thrust_curve.last_thrust:.2f}N"
            )
        rocket = copy.deepcopy(self.flight.rocket)

        rocket.add_motor(
            motor=motor,
            position=self.motor_position
        )

        return Flight(
            name=f"{self.name}-{motor.name}",
            flight_state=FlightGenerator(
                rocket=rocket,
                env=self.flight.env,
                launcher=self.flight.launcher,
            ),
            verbose=verbose,
            terminate_on_apogee=True
        )

    def _minimise_f(
            self,
            bounds: tuple[float, float],
            func: Callable[[float], float],
            thrust_tolerance: float,
            max_iter: int
    ):
        sorted_bounds = tuple(sorted(bounds))
        minimize_scalar(
            func,
            bounds=sorted_bounds,
            method="bounded",
            options={
                "xatol": thrust_tolerance,
                "maxiter": max_iter
            }
        )

    def _optimise(
            self,
            T0: float,
            T1: float,
            T_interest: list[float],
            n_sims: int,
            verbose: bool,
            func: Callable[[float], float],
            thrust_tolerance: float
    ):
        n_sims_adaptive_grid = 10
        n_sims = n_sims - n_sims_adaptive_grid
        if n_sims < 0:
            raise ValueError(
                f"n_sims must be atleast {n_sims_adaptive_grid}"
            )

        thrusts = variable_density_1d_grid(
            xmin=T0,
            xmax=T1,
            xinterest=T_interest,
            points=n_sims_adaptive_grid
        )

        if verbose:
            print("\nStarting Course Grid Pass")
        apogees = np.array([-func(thrust) for thrust in thrusts])

        # sort by apogee
        order = np.argsort(apogees)[::-1]
        sorted_apogees = apogees[order]
        sorted_thrusts = thrusts[order]

        max_apogee = sorted_apogees[0]
        max_thrust = sorted_thrusts[0]

        if max_apogee < 0:
            print(f"\nno solutions found")
            return None

        if max_thrust == T1:
            if verbose:
                print("Checking maximum thrust condition")

            apogee = -func(max_thrust - 1)

            if apogee < max_apogee:
                if verbose:
                    print("Found optimal at maximum thrust condition")
            else:
                if verbose:
                    print("Searching near maximum thrust condition")

                self._minimise_f(
                    bounds=(sorted_thrusts[0], sorted_thrusts[1]),
                    func=func,
                    thrust_tolerance=thrust_tolerance,
                    max_iter=n_sims
                )

        elif max_thrust == T0:
            if verbose:
                print("Checking minimum thrust condition")

            apogee = -func(max_thrust + 1)

            if apogee < max_apogee:
                if verbose:
                    print("Found optimal at minimum thrust condition")
            else:
                if verbose:
                    print("Searching near minimum thrust condition")

                self._minimise_f(
                    bounds=(sorted_thrusts[0], sorted_thrusts[1]),
                    func=func,
                    thrust_tolerance=thrust_tolerance,
                    max_iter=n_sims
                )

        else:
            if verbose:
                print("Searching around maximum apogee condition")

            idx = np.argmax(apogees)

            self._minimise_f(
                bounds=(thrusts[idx-1], sorted_thrusts[idx+1]),
                func=func,
                thrust_tolerance=thrust_tolerance,
                max_iter=n_sims
            )

    def optimise_thrust_curve(
            self,
            constraints: Constraints,
            assumptions: Assumptions,
            *,
            passes_points: list[int] = [10, 10, 10],
            verbose: bool = False,
    ) -> tuple[Flight, Motor, Table] | None:
        """computes a near-optimal thrust curve to maximise altitudes given the provided constraints and assumptions.

        This works by breaking up the thrust curve into discrete impulse blocks and then determining the optimal thrust to maximise apogee for each point over multiple passes. The first pass builds up the impulse until the target total impulse is met and successive passes change the thrust curve from previous passes. This uses an adaptive-grid search and local minimisation to determine the thrust at each point.

        This can take a very long time to produce an optimal thrust curve based on the input provided. Its not recommended to change configuration variables, particularly to increase them without consideration.

        Plots are generated and saved to file as the function works to follow the progress of the optimisation. The saved location of these plots is printed when the function is called. The RASP ENG thrust curve is saved after each pass and flight plots and information is plotted and saved once the optimisation is complete.

        Critically, this produces thrust curves with instantaneous thrust increases that are unlikely to be realistically achieved.

        Args:
            constraints: `Constraints` object that defines all the constraints on the final thrust curve.
            assumptions: `Assumptions` object that defines all the assumption made to be used in the rocket simulations. It's recommended to choose similar values to existing motors (by impulse class) available from thrustcurve.org.
            passes_points: How many points to be optimised for each pass of the optimiser. The number of passes is defined by the length of the list provided. Defaults to [10, 10, 10].
            verbose: Enable verbose mode to print debug and simulation information during optimisation. Defaults to False.

        Returns:
            ...
        """

        if not verbose:
            nperr = np.geterr()
            np.seterr(all="ignore")
            warnings.simplefilter("ignore")

        master_motor = Motor(
            custom_name="Optimal",
            total_mass=assumptions.motor_total_mass,
            prop_mass=assumptions.motor_prop_mass,
            diameter=assumptions.motor_diameter,
            length=assumptions.motor_length,
        )

        results = Results(name=f"optimise-thrust-curve-{self.name}")

        print(f"\nSee outputs/{results.category}/{results.name} for progress plots throughout optimisation.\nDepending on input parameters this may take a long time...\n")

        # 1. determine maximum total impulse to leave the rail with speed >= min speed
        total_impulse_out_of_rail = 0
        for thrust in np.linspace(constraints.min_thrust, constraints.max_thrust, 15):

            # 1.1 determine out of rail time
            motor = master_motor.append_until_impulse(
                thrust=thrust,
                impulse=constraints.total_impulse
            )

            try:
                flight = self._sim(motor=motor, verbose=verbose)
            except SimulationFailedError as e:
                warnings.warn(
                    f"optimise_thrust_curve initial out of rail screen simulation failed (thrust={thrust:.2f}N): {e}"
                )
                continue

            if flight.simulation_failed():
                continue

            # 1.2 check stability constraint out of rail
            if flight.out_of_rail_velocity < constraints.min_speed_out_of_rail:
                continue

            motor = master_motor.append_discrete_block(
                dtime=flight.out_of_rail_time,
                thrust=thrust
            )

            if verbose:
                print(
                    f"thrust: {thrust:>10.2f}N   "
                    f"time_out_of_rail: {flight.out_of_rail_time:>8.2f}s   "
                    f"velocity_out_of_rail: {flight.out_of_rail_velocity:>9.2f}m/s   "
                    f"total_impulse_out_of_rail: {motor.thrust_curve.total_impulse:>10.4f}Ns   "
                    f"apogee: {flight.apogee/1e3:>10.4f}km   "
                )

            # 1.3 determine maximum total impulse to leave the rail
            total_impulse_out_of_rail = max(
                total_impulse_out_of_rail, motor.thrust_curve.total_impulse)

        def get_target_total_impulse(n_point: int, n_points: int) -> float:
            return (
                (constraints.total_impulse - total_impulse_out_of_rail)
                * (n_point/(n_points-1)) + total_impulse_out_of_rail
            )

        def get_target_total_impulses(n_points: int):
            targets = [
                get_target_total_impulse(
                    n_point=n_point,
                    n_points=n_points,
                ) for n_point in range(n_points)
            ]

            yield from pairwise([1e-6] + targets)

        for n_pass, n_points in enumerate(passes_points):

            results.next_pass()

            progress_bar = ProgressBar(
                tasks=n_points,
                name=f"{results.name} - pass: {n_pass}"
            )

            for n_point, (last_target_impulse, target_impulse) in enumerate(get_target_total_impulses(n_points)):

                if verbose:
                    print(
                        f"\npass: {n_pass} - step: {n_point} - impulse_target: {target_impulse:.2f}Ns - last_target_impulse: {last_target_impulse:.2f}Ns"
                    )

                results.next_step()
                msg: list[str] = []

                def f(thrust: float) -> float:
                    if n_pass == 0:
                        motor = master_motor.append_until_impulse(
                            thrust=thrust,
                            impulse=target_impulse
                        )
                    else:
                        motor = master_motor.replace_impulse_region(
                            start_impulse=last_target_impulse,
                            end_impulse=target_impulse,
                            thrust=thrust
                        ).round_to_impulse(constraints.total_impulse)

                    try:
                        flight = self._sim(motor=motor, verbose=verbose)
                    except SimulationFailedError as e:
                        warnings.warn(
                            f"optimise_thrust_curve simulation failed (pass={n_pass}, point={n_point}, thrust={thrust:.2f}N): {e}")
                        results.add(Result(success=False))
                        return 1e12

                    if flight.simulation_failed():
                        results.add(Result(success=False))
                        return 1e12

                    if flight.out_of_rail_velocity < constraints.min_speed_out_of_rail:
                        results.add(Result(success=False))
                        return 1e12

                    if verbose:
                        print(
                            f"thrust:{thrust:10.2f}N -> "
                            f"apogee:{flight.apogee/1e3:10.4f}km"
                        )

                    results.add(
                        Result(success=True, thrust=thrust,
                               motor=motor, flight=flight)
                    )
                    return -flight.apogee

                # lower and upper limit
                T0 = constraints.min_thrust
                T1 = constraints.max_thrust
                T_interest = [T0, T1]

                # last optimal point
                last = results.last_best_result()
                T_interest.extend(
                    [last.thrust] if last and last.thrust else []
                )

                self._optimise(
                    T0=T0,
                    T1=T1,
                    T_interest=T_interest,
                    n_sims=30,
                    verbose=verbose,
                    func=f,
                    thrust_tolerance=constraints.thrust_tolerance
                )

                msg.append(f"sims: {len(results.results)}")

                if results.is_no_success():
                    print(f"\nstep: {n_point} - no solutions found")
                    return None

                best_result = results.best()
                assert best_result.motor
                assert best_result.flight
                assert best_result.thrust

                master_motor = best_result.motor

                thrust = best_result.thrust
                total_impulse = master_motor.thrust_curve.total_impulse
                apogee = best_result.flight.apogee

                msg = [
                    f"step: {n_point}",
                    f"thrust: {thrust:.2f}N",
                    f"impulse: {total_impulse:.2f}Ns",
                    f"apogee: {apogee/1e3:.2f}km",
                    *msg
                ]

                progress_bar.update(n_point+1, " - ".join(msg))

                if verbose:
                    print()

                results.plots.thrust_vs_altitude(suppress_prints=not verbose)
                results.plots.best_thrust_curve(suppress_prints=not verbose)
                results.plots.progress(suppress_prints=not verbose)

            assert best_result.motor
            best_result.motor.to_RASP_file(
                filepath=f"outputs/{results.category}/{results.name}/RASP/{best_result.motor.name} (pass={results.n_pass}).eng",
            )

        progress_bar.complete()

        assert best_result.motor
        assert best_result.flight
        best_motor = best_result.motor
        best_flight = best_result.flight

        best_motor.info.all()
        best_flight.info.all()
        best_motor.plots.all()
        best_flight.plots.all()

        if not verbose:
            np.seterr(**nperr)  # type: ignore
            warnings.simplefilter("default")

        return best_flight, best_motor, best_flight.info.overview(save=False, show=False)

    @overload
    def find_optimal_motor(self, *, motors: Sequence[Motor], min_speed_out_of_rail: float = 0) -> tuple[Flight, Motor, Table] | None:
        ...

    @overload
    def find_optimal_motor(self, *, impulseClass: IMPULSE_CLASSES | str = "", manufacturers: Sequence[MOTOR_MANUFACTURERS] = [], min_speed_out_of_rail: float = 0) -> tuple[Flight, Motor, Table] | None:
        ...

    def find_optimal_motor(
            self,
            *,
            motors: Sequence[Motor] = [],
            impulseClass: IMPULSE_CLASSES | str = "",
            manufacturers: Sequence[MOTOR_MANUFACTURERS] = [],
            verbose: bool = False,
            min_speed_out_of_rail: float = 0
    ) -> tuple[Flight, Motor, Table] | None:

        if not verbose:
            nperr = np.geterr()
            np.seterr(all="ignore")
            warnings.simplefilter("ignore")

        if not motors:
            if not impulseClass and not manufacturers:
                raise ValueError(
                    f"atleast one of `motors`, `impulseClass` and `manufacturers` must be provided"
                )

            motor_search = MotorSearch(_suppress_prints=True)

            if manufacturers:
                motors_results = []
                for manufacturer in manufacturers:
                    motors_results.extend(motor_search.get_motors(
                        impulseClass=impulseClass, manufacturer=manufacturer
                    ))
            else:
                motors_results = motor_search.get_motors(
                    impulseClass=impulseClass)

            motors = [motor.to_motor() for motor in motors_results]

        progress_bar = ProgressBar(
            tasks=len(motors),
            name=f"find optimal motor"
        )

        results = Results(name=f"optimal-motor-{self.name}")

        def f(motor: Motor) -> float:
            try:
                flight = self._sim(motor=motor, verbose=verbose)
            except SimulationFailedError as e:
                warnings.warn(
                    f"find_optimal_motor simulation failed (motor={motor.name:.2f}): {e}")
                results.add(Result(success=False))
                return 1e12

            if flight.simulation_failed():
                results.add(Result(success=False))
                return 1e12

            if flight.out_of_rail_velocity < min_speed_out_of_rail:
                results.add(Result(success=False))
                return 1e12

            if verbose:
                print(
                    f"motor:{motor.name:20} -> "
                    f"apogee:{flight.apogee/1e3:10.4f}km"
                )

            results.add(
                Result(
                    success=True,
                    thrust=-1,
                    motor=motor,  # type: ignore
                    flight=flight
                )
            )
            return -flight.apogee

        for i, motor in enumerate(motors):
            f(motor)

            result = results.last()
            if result is None:
                msg = []
            else:
                assert result.motor is not None
                assert result.flight is not None

                motor_name = result.motor.name
                total_impulse = result.motor.thrust_curve.total_impulse
                apogee = result.flight.apogee

                msg = [
                    f"motor: {motor_name}",
                    f"impulse: {total_impulse:.2f}Ns",
                    f"apogee: {apogee/1e3:.2f}km",
                ]
            progress_bar.update(i, " - ".join(msg))

        if results.is_no_success():
            print("no solutions found")
            return None

        table: Table | None = None

        for result in results.results:
            if result.success:
                assert result.flight
                out_table = result.flight.info.overview(save=False, show=False)
                table = table + out_table if table is not None else out_table

        if not verbose:
            np.seterr(**nperr)  # type: ignore
            warnings.simplefilter("default")

        progress_bar.complete()

        assert table is not None
        table.show(
            sort_column="Altitude [km]", sort_ascending=False, suppress_prints=not verbose)
        table.save(
            sort_column="Altitude [km]", sort_ascending=False, suppress_prints=not verbose)

        best_result = results.best()

        assert best_result.flight
        assert best_result.motor
        best_flight = best_result.flight
        best_motor = best_result.motor

        results.plots.thrust_vs_altitude(suppress_prints=not verbose)

        return best_flight, best_motor, table

    def optimise_fixed_thrust(self, constraints: Constraints, assumptions: Assumptions, verbose: bool = False, n_sims: int = 30) -> tuple[Flight, Motor, Table] | None:

        results = Results(name=f"optimise-fixed-thrust-{self.name}")

        if not verbose:
            nperr = np.geterr()
            np.seterr(all="ignore")
            warnings.simplefilter("ignore")

        def f(thrust: float) -> float:
            motor = Motor.from_fixed_thrust(
                custom_name=f"Fixed-{thrust:.2f}N",
                total_mass=assumptions.motor_total_mass,
                prop_mass=assumptions.motor_prop_mass,
                diameter=assumptions.motor_diameter,
                length=assumptions.motor_length,
                thrust=thrust,
                impulse=constraints.total_impulse
            )

            try:
                flight = self._sim(motor=motor, verbose=verbose)
            except SimulationFailedError as e:
                warnings.warn(
                    f"optimise_fixed_thrust simulation failed (thrust={thrust:.2f}N): {e}")
                results.add(Result(success=False))
                return 1e12

            if flight.simulation_failed():
                results.add(Result(success=False))
                return 1e12

            if flight.out_of_rail_velocity < constraints.min_speed_out_of_rail:
                results.add(Result(success=False))
                return 1e12

            if verbose:
                print(
                    f"thrust:{thrust:10.2f}N -> "
                    f"apogee:{flight.apogee/1e3:10.4f}km"
                )

            results.add(
                Result(success=True, thrust=thrust,
                       motor=motor, flight=flight)
            )
            return -flight.apogee

        # lower and upper limit
        T0 = constraints.min_thrust
        T1 = constraints.max_thrust
        T_interest = [T0, T1]

        self._optimise(
            T0=constraints.min_thrust,
            T1=constraints.max_thrust,
            T_interest=[constraints.min_thrust, constraints.max_thrust],
            n_sims=n_sims,
            verbose=verbose,
            func=f,
            thrust_tolerance=constraints.thrust_tolerance
        )

        if results.is_no_success():
            print("no solutions found")
            return None

        table: Table | None = None

        for result in results.results:
            if result.success:
                assert result.flight
                out_table = result.flight.info.overview(save=False, show=False)
                table = table + out_table if table else out_table

        if not verbose:
            np.seterr(**nperr)  # type: ignore
            warnings.simplefilter("default")

        assert table is not None
        table.show(
            sort_column="Altitude [km]", sort_ascending=False, suppress_prints=not verbose
        )
        table.save(
            sort_column="Altitude [km]", sort_ascending=False, suppress_prints=not verbose
        )

        best_result = results.best()

        assert best_result.flight
        assert best_result.motor
        best_flight = best_result.flight
        best_motor = best_result.motor

        results.plots.thrust_vs_altitude(suppress_prints=not verbose)

        return best_flight, best_motor, table
