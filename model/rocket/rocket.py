from typing import Literal, overload, Protocol, TypeVar, Callable, Any, ClassVar, runtime_checkable
from dataclasses import dataclass, fields, field, is_dataclass
import pandas as pd
import numpy as np
import rocketpy

from model.rocket.motor import MotorSearch, Motor
from model.util.output import *
from model.rocket.aero import Aero
import model.rocket.rasaero as rasaero

rng = np.random.default_rng(28092001)

T = TypeVar("T", covariant=True)

__all__ = (
    "Uniform",
    "Normal",
    "RocketGenerator",
    "EnvironmentGenerator",
    "LauncherGenerator",
    "FlightGenerator",
)

#
# Types
#

airfoil_type = (
    tuple[str, Literal["degrees", "radians"]]
    | tuple[np.ndarray, Literal["degrees", "radians"]]
    | None
)

nose_types = Literal[
    "von karman", "conical", "ogive", "lvhaack", "powerseries"
]

parachute_types = Literal["apogee", "motor ejection", "altitude"]


#
# RAS Aero II Config
#


@dataclass
class RASAeroConfig:
    surface: rasaero.Surfaces
    use_modified_barrowman: bool = True
    turbulence_only_flight: bool = False


#
# Distribution
#
# TODO: Include an ensemble type?


@runtime_checkable
class SupportsSample(Protocol[T]):
    def ideal(self) -> T: ...
    def sample(self) -> T: ...


class Quantity(SupportsSample[float]):
    category = "distribution"

    def __init__(self, x: float) -> None:
        self.x = x

        self.plots = QuantityPlots(self)

    @property
    def name(self) -> str:
        return self.__class__.__name__.rstrip("Distribution")

    def __float__(self) -> float: return self.x
    def ideal(self) -> float: return self.x
    def sample(self) -> float: return self.x

    def __repr__(self) -> str:
        params = " ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"<{self.__class__.__name__} ideal={self.ideal()} {params}>"


class QuantityPlots(PlotBase[Quantity]):

    @scientific_plot()
    def distribution(self, ax: Axes, n: int = 1000):
        model = self.model

        data = [model.sample() for _ in range(n)]

        ax.hist(data, bins=30, density=True)

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")


class UniformDistribution(Quantity):
    def __init__(self, low: float, high: float) -> None:
        super().__init__(0.5 * (low + high))
        self.low = low
        self.high = high

    def sample(self) -> float:
        return rng.uniform(self.low, self.high)

    @classmethod
    def from_mean(cls, x: float, width: float) -> "UniformDistribution":
        return UniformDistribution(
            low=x - 0.5 * width,
            high=x + 0.5 * width
        )

    @classmethod
    def from_mean_frac(cls, x: float, fraction: float) -> "UniformDistribution":
        return cls.from_mean(
            x=x,
            width=x * fraction
        )


class NormalDistribution(Quantity):
    def __init__(self, mean: float, std: float) -> None:
        super().__init__(mean)
        self.mean = mean
        self.std = std

    def sample(self) -> float:
        return rng.normal(self.mean, self.std)

    @classmethod
    def from_mean(cls, x: float, std: float) -> "NormalDistribution":
        return NormalDistribution(mean=x, std=std)

    @classmethod
    def from_mean_frac(cls, x: float, std_frac: float) -> "NormalDistribution":
        return NormalDistribution(
            mean=x,
            std=x * std_frac
        )


Uniform = UniformDistribution
Normal = NormalDistribution

Q = Quantity | float


def STANDARD_RADIAL_DISTANCE_DIST(x): return Normal.from_mean(x, 0.4 / 1000)
def STANDARD_AXIAL_DISTANCE_DIST(x): return Normal.from_mean(x, 0.004)
def STANDARD_MASS_DIST(x): return Normal.from_mean(x, 0.01)
def STANDARD_RADIAL_ANGLE_DIST(x): return Normal.from_mean(x, 10)

#
# Input handling
#


@overload
def _handle_input(
    x: Q,
    default: Callable[[float], Quantity] | None = None,
) -> Quantity: ...


@overload
def _handle_input(
    x: tuple[Q, Q, Q],
    default: Callable[[float], Quantity] | None = None,
) -> tuple[Quantity, Quantity, Quantity]: ...


@overload
def _handle_input(
    x: tuple[Q, ...],
    default: Callable[[float], Quantity] | None = None,
) -> tuple[Quantity, ...]: ...


def _handle_input(x: Q | tuple[Q, ...], default: Callable[[float], Quantity] | None = None) -> Quantity | tuple[Quantity, ...]:
    if isinstance(x, (float, int)):
        if default is not None:
            return default(x)
        else:
            return Quantity(x)
    elif isinstance(x, Quantity):
        return x
    elif isinstance(x, tuple):
        return tuple(_handle_input(y, default) for y in x)
    else:
        raise TypeError(f"Invalid Inputs: {type(x)}")

#
# State Base
#


@dataclass
class State[T]:
    ...

    def to_dict(self):
        return State.__to_dict(self)

    def to_df(self):
        return pd.json_normalize(self.to_dict())

    @staticmethod
    def __to_dict(obj: Any):
        if is_dataclass(obj):
            return {
                field.name: State.__to_dict(getattr(obj, field.name))
                for field in fields(obj)
            }
        if isinstance(obj, dict):
            return {k: State.__to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [State.__to_dict(v) for v in obj]
        return obj

    def hash(self) -> str:
        return str(self.to_hash_dict())

    def to_hash_dict(self) -> str:
        return State.__to_hash_dict(self)

    @staticmethod
    def __to_hash_dict(obj: Any):
        SKIP_KEYS = ("info", "plots")

        if is_dataclass(obj):
            result = {}

            for field in fields(obj):
                if field.name in SKIP_KEYS:
                    continue

                value = getattr(obj, field.name)

                # Don't include lists/tuples > 4
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 4:
                    continue

                result[field.name] = State.__to_hash_dict(value)

            return result

        if isinstance(obj, dict):
            return {
                k: State.__to_hash_dict(v)
                for k, v in obj.items()
                if k not in SKIP_KEYS
                and not isinstance(v, (list, tuple, np.ndarray)) and len(v) > 4
            }

        if isinstance(obj, (list, tuple, np.ndarray)):
            return [State.__to_hash_dict(v) for v in obj]

        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        return obj


@dataclass
class ComponentState(State[T]):
    def add(self) -> T:
        ...


@dataclass
class AssemblyState(State[T]):
    def make(self) -> T:
        ...

#
# Generator Base
#


@dataclass
class ComponentGenerator(SupportsSample[T]):
    _output_type: ClassVar[type]

    @staticmethod
    def _transform_value(
        value: Any,
        transform: Callable[[SupportsSample], Any],
    ) -> Any:
        if isinstance(value, SupportsSample):
            return transform(value)
        elif isinstance(value, tuple):
            return tuple(
                ComponentGenerator._transform_value(item, transform)
                for item in value
            )
        elif isinstance(value, list):
            return [
                ComponentGenerator._transform_value(item, transform)
                for item in value
            ]
        else:
            return value

    def _transform(self, transform: Callable[[SupportsSample], Any]) -> T:
        values = {
            field.name: ComponentGenerator._transform_value(
                value=getattr(self, field.name),
                transform=transform
            )
            for field in fields(self)
        }

        return self._output_type(**values)

    def ideal(self) -> T:
        return self._transform(lambda q: q.ideal())

    def sample(self) -> T:
        return self._transform(lambda q: q.sample())

#
# Rocket
#


@dataclass
class RocketState(AssemblyState[rocketpy.Rocket]):
    category = "rocket"

    name: str
    diameter: float
    mass_without_motor: float
    inertia_without_motor: tuple[float, float, float]
    center_of_mass_without_motor: float
    power_off_drag: np.ndarray | None = None
    power_on_drag: np.ndarray | None = None
    aero: Aero | Literal["estimate"] | None = None

    rail_buttons: "RailButtonState | None" = None
    nose: "NoseState | None" = None
    fins: "TrapezoidalFinsState | None" = None
    tails: list["TailState"] = field(default_factory=list)
    parachutes: list["ParachuteState"] = field(default_factory=list)
    motor: "MotorState | None" = None

    def __post_init__(self):
        self.plots = RocketStatePlot(self)
        self.info = RocketStateInfo(self)

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def body_length(self) -> float:
        assert self.nose
        assert self.fins
        return self.fins.position + self.fins.root_chord - self.nose.length

    def make(self) -> rocketpy.Rocket:
        rocket = rocketpy.Rocket(
            radius=self.radius,
            mass=self.mass_without_motor,
            inertia=self.inertia_without_motor,
            power_off_drag=self.power_off_drag,
            power_on_drag=self.power_on_drag,
            center_of_mass_without_motor=self.center_of_mass_without_motor,
            coordinate_system_orientation="nose_to_tail"
        )

        if self.rail_buttons is not None:
            self.rail_buttons.add(rocket)
        if self.nose is not None:
            self.nose.add(rocket)
        if self.fins is not None:
            self.fins.add(rocket)
        if self.tails:
            [tail.add(rocket) for tail in self.tails]
        if self.parachutes:
            [parachute.add(rocket) for parachute in self.parachutes]
        if self.motor:
            self.motor.add(rocket)

        return rocket


@dataclass
class RocketGenerator(ComponentGenerator[RocketState]):
    name: str
    diameter: Quantity
    mass_without_motor: Quantity
    inertia_without_motor: tuple[Quantity, Quantity, Quantity]
    center_of_mass_without_motor: Quantity
    power_off_drag: np.ndarray | None = None
    power_on_drag: np.ndarray | None = None
    aero: Aero | Literal["estimate"] | None = None

    rail_buttons: "RailButtonGenerator | None" = None
    nose: "NoseGenerator | None" = None
    fins: "TrapezoidalFinsGenerator | None" = None
    tails: list["TailGenerator"] = field(default_factory=list)
    parachutes: list["ParachuteGenerator"] = field(default_factory=list)
    motor: "MotorGenerator | None" = None

    _output_type = RocketState

    def __init__(
        self,
        name: str,
        diameter: Q,
        mass_without_motor: Q,
        inertia_without_motor: tuple[Q, Q, Q],
        center_of_mass_without_motor: Q,
        aero: Aero | Literal["estimate"] | None = None,
        power_off_drag: np.ndarray | None = None,
        power_on_drag: np.ndarray | None = None
    ):
        self.name = name

        self.diameter = _handle_input(
            diameter,
            default=STANDARD_RADIAL_DISTANCE_DIST
        )
        self.mass_without_motor = _handle_input(
            mass_without_motor,
            default=STANDARD_AXIAL_DISTANCE_DIST
        )
        self.inertia_without_motor: tuple[Quantity, Quantity, Quantity] = _handle_input(
            inertia_without_motor,
            default=STANDARD_AXIAL_DISTANCE_DIST
        )
        self.center_of_mass_without_motor = _handle_input(
            center_of_mass_without_motor,
            default=STANDARD_AXIAL_DISTANCE_DIST
        )

        if aero != "estimate":
            if isinstance(aero, Aero):
                power_off_drag = aero.power_off_drag()  # type: ignore
                power_on_drag = aero.power_on_drag()  # type: ignore
            elif power_on_drag is None and power_off_drag is not None:
                power_on_drag = power_off_drag

            if power_on_drag is None or power_off_drag is None:
                raise ValueError(
                    f"Atleast power_off_drag or aero must be provided."
                )

        self.aero = aero
        self.power_on_drag = power_on_drag
        self.power_off_drag = power_off_drag

        self.rail_buttons: RailButtonGenerator | None = None
        self.nose: NoseGenerator | None = None
        self.fins: TrapezoidalFinsGenerator | None = None
        self.tails: list[TailGenerator] = []
        self.parachutes: list[ParachuteGenerator] = []
        self.motor: MotorGenerator | None = None

    def add_rail_buttons(
        self,
        upper_button_position: Q,
        lower_button_position: Q,
        angular_position: Q,
        diameter: float = 9.7 / 1000,
        height: float = 7.5 / 1000
    ):
        self.rail_buttons = RailButtonGenerator(
            upper_button_position=_handle_input(
                upper_button_position,
                default=STANDARD_AXIAL_DISTANCE_DIST
            ),
            lower_button_position=_handle_input(
                lower_button_position,
                default=STANDARD_AXIAL_DISTANCE_DIST
            ),
            angular_position=_handle_input(
                angular_position,
                default=STANDARD_RADIAL_ANGLE_DIST
            ),
            diameter=diameter,
            height=height
        )

    @overload
    def add_nose(self, length: Q, kind: Literal["von karman", "conical", "ogive", "lvhaack"], *, position: Q = 0, blunt_radius: float | None = None):
        ...

    @overload
    def add_nose(self, length: Q, kind: Literal["powerseries"], *, power: float, position: Q = 0, blunt_radius: float | None = None):
        ...

    def add_nose(
        self,
        length: Q,
        kind: nose_types,
        position: Q = 0,
        power: float | None = None,
        blunt_radius: float | None = None
    ):
        self.nose = NoseGenerator(
            length=_handle_input(
                length,
                default=STANDARD_AXIAL_DISTANCE_DIST),
            kind=kind,
            position=_handle_input(position),
            power=power,
            blunt_radius=blunt_radius
        )

    def add_trapezoidal_fins(
        self,
        n: int,
        root_chord: Q,
        tip_chord: Q,
        span: Q,
        position: Q,
        sweep_length: Q | None = None,
        airfoil: airfoil_type = None,
        airfoil_type: rasaero.FinAirfoilSections = "Subsonic NACA",
        leading_edge_radius: float | None = None,
        thickness: float | None = None
    ):
        """
        Args:
            n: number of fins
            root_chord: root chord [m]
            tip_chord: tip chord [m]
            span: height [m]
            position: nose to tip of fins [m]
            sweep_length: sweep length [m]. Defaults to None.
            airfoil: airfoil angle of attack data. Estimates aerodynamic data if None based on airfoil type. Either generates aerofoil aerodynamic data from xFoil or assumes flat plate. Defaults to None. Provide (np.columnstack(alpha, CL), "degrees" or "radians") or (file_name, "degrees" or "radians") if providing own data.
            airfoil_type: airfoil type for RASAero Aerodynamic Simulations. Defaults to "Subsonic NACA".
            leading_edge_radius: leading edge radius [m] of fins for RASAero Aerodynamic Simulations. Defaults to None.
            thickness: fin thickness [m] for RASAero Aerodynamic Simulations. Defaults to None.
        """
        root_chord = _handle_input(root_chord, STANDARD_AXIAL_DISTANCE_DIST)
        tip_chord = _handle_input(tip_chord, STANDARD_AXIAL_DISTANCE_DIST)

        if airfoil is None:
            match airfoil_type:
                case "Square":
                    airfoil = None  # flat plate assumption
                case "Subsonic NACA":
                    if thickness is None:
                        raise ValueError(
                            "Thickness must be provided to estimate subsonic NACA performance"
                        )
                    chord = 0.5 * (tip_chord.ideal() + root_chord.ideal())
                    thickness_ratio = thickness / chord

                    from model.core.airfoil import AirfoilManager

                    airfoil = AirfoilManager().get_airfoil(
                        thickness=thickness_ratio,
                        alpha_min=0,
                        alpha_max=30,
                        Mach=0.3
                    ).rocketpy_airfoil
                case _:
                    airfoil = None  # flat plate assumption

        self.fins = TrapezoidalFinsGenerator(
            n=n,
            root_chord=root_chord,
            tip_chord=tip_chord,
            span=_handle_input(
                span,
                default=STANDARD_AXIAL_DISTANCE_DIST
            ),
            position=_handle_input(
                position,
                default=STANDARD_AXIAL_DISTANCE_DIST
            ),
            sweep_length=None if sweep_length is None else _handle_input(
                sweep_length,
                default=STANDARD_AXIAL_DISTANCE_DIST
            ),
            airfoil=airfoil,
            airfoil_type=airfoil_type,
            leading_edge_radius=leading_edge_radius,
            thickness=thickness
        )

    def add_tail(
        self,
        top_diameter: Q,
        bottom_diameter: Q,
        length: Q,
        position: Q
    ):
        self.tails.append(TailGenerator(
            top_diameter=_handle_input(
                top_diameter,
                default=STANDARD_RADIAL_DISTANCE_DIST
            ),
            bottom_diameter=_handle_input(
                bottom_diameter,
                default=STANDARD_RADIAL_DISTANCE_DIST
            ),
            length=_handle_input(
                length,
                default=STANDARD_AXIAL_DISTANCE_DIST
            ),
            position=_handle_input(
                position,
                default=STANDARD_AXIAL_DISTANCE_DIST
            )
        ))

    @overload
    def add_parachute(self, name: str, cd: Q, diameter: Q, *, trigger: Literal["apogee"]):
        ...

    @overload
    def add_parachute(self, name: str, cd: Q, diameter: Q, *, motor_delay: Q, trigger: Literal["motor ejection"]):
        ...

    @overload
    def add_parachute(self, name: str, cd: Q, diameter: Q, *, deploy_altitude: Q, trigger: Literal["altitude"]):
        ...

    def add_parachute(
        self,
        name: str,
        cd: Q,
        diameter: Q,
        *,
        trigger: parachute_types = "apogee",
        motor_delay: Q | None = None,
        deploy_altitude: Q | None = None
    ):
        if trigger == "motor ejection" and motor_delay is None:
            raise ValueError(
                f"motor delay must be supplied when motor ejection trigger is selected")
        if trigger == "altitude" and deploy_altitude is None:
            raise ValueError(
                f"deploy altitude must be supplied when altitude trigger is selected")

        self.parachutes.append(ParachuteGenerator(
            name=name,
            cd=_handle_input(
                cd,
                default=lambda x: Normal.from_mean_frac(x, 0.05)
            ),
            diameter=_handle_input(
                diameter,
                default=STANDARD_RADIAL_DISTANCE_DIST
            ),
            trigger=trigger,
            motor_delay=None if motor_delay is None else _handle_input(
                motor_delay,
                default=lambda x: Normal.from_mean(x, 0.4)
            ),
            deploy_altitude=None if deploy_altitude is None else _handle_input(
                deploy_altitude,
                default=lambda x: Normal.from_mean(x, 20)
            ),
        ))

    @overload
    def add_motor(self, *, motor_name: str, position: Q, thrust_multiplier: Q = 1, time_multiplier: Q = 1, mass_multiplier: Q = 1, nozzle_diameter: float | None = None):
        """Add a commercial motor from thrustcurve.org based on the motor name.

        This function searches thrustcurve.org, then downloads and caches the motor metadata and thrust curve. Where their are multiple matches the correct motor can be selected through the command line.

        Args:
            motor_name: `str`, name of motor.
            position: `Quantity | float`, distance from the nose cone to the aft of the motor [m]
        """
        ...

    @overload
    def add_motor(self, *, motor: Motor, position: Q, thrust_multiplier: Q = 1, time_multiplier: Q = 1, mass_multiplier: Q = 1, nozzle_diameter: float | None = None):
        """Add a motor that inherits `Motor`, such as `CustomMotor` generated from custom data points, or with an optimised thrust curve, `FixedThrustMotor`, or `ThrustCurveMotor` from thrustcurve.org data.

        Example:
            >>> motor = FixedThrustMotor(
            >>>     name="2kN",
            >>>     total_mass=8988 / 1000,
            >>>     prop_mass=5719  1000,
            >>>     diameter=98 / 1000,
            >>>     length=732 / 1000,
            >>>     thrust=2000,
            >>>     total_impulse_target=10239
            >>> )
            >>> rocket.add_motor(motor, 2325 / 1000)

        Args:
            motor: motor that inherits `Motor`
            position: `Quantity | float`, distance from teh nose cone to the aft of the motor [m]
        """
        ...

    def add_motor(
            self,
            *,
            position: Q,
            motor_name: str | None = None,
            motor: Motor | None = None,
            thrust_multiplier: Q = 1,
            time_multiplier: Q = 1,
            mass_multiplier: Q = 1,
            nozzle_diameter: float | None = None
    ):
        if motor_name is not None:
            motor_obj = MotorSearch().get_motor(motor_name).to_motor()

        if motor is not None:
            motor_obj = motor

        if motor_obj is None:
            raise ValueError(
                "either motor name of object that inherits `Motor must be supplied"
            )

        self.motor = MotorGenerator(
            motor=motor_obj,
            position=_handle_input(
                position,
                default=STANDARD_AXIAL_DISTANCE_DIST,
            ),
            thrust_multiplier=_handle_input(
                thrust_multiplier,
                default=lambda x: Normal.from_mean(x, 0.02)
            ),
            time_multiplier=_handle_input(
                time_multiplier,
                default=lambda x: Normal.from_mean(x, 0.05)
            ),
            mass_multiplier=_handle_input(
                mass_multiplier,
                default=STANDARD_MASS_DIST
            ),
            nozzle_diameter=nozzle_diameter
        )


class RocketStatePlot(PlotBase[RocketState]):

    @scientific_plot(auto_limits=None, outside_legend="right", figsize=(6, 1.5))
    def draw(self, ax: Axes, vis_args=None, plane="xz") -> str:
        """adapted from rocketpy.plots.rocket_Plots"""
        from rocketpy.plots.rocket_plots import _RocketPlots

        rocket = _RocketPlots(self.model.make())
        rocket._RocketPlots__validate_aerodynamic_surfaces(  # type: ignore
            plane)

        if vis_args is None:
            vis_args = {
                "background": "#FFFFFF",
                "tail": "black",
                "nose": "black",
                "body": "black",
                "fins": "black",
                "motor": "black",
                "buttons": "black",
                "line_width": 1.0,
            }

        ax.set_aspect("equal")
        ax.set_facecolor(vis_args["background"])

        csys = rocket.rocket._csys
        reverse = csys == 1
        surfaces = rocket.rocket.aerodynamic_surfaces.sort_by_position(
            reverse=reverse)

        drawn_surfaces = rocket._draw_aerodynamic_surfaces(
            ax, vis_args, plane, surfaces)
        last_radius, last_x = rocket._draw_tubes(ax, drawn_surfaces, vis_args)
        rocket._draw_motor(last_radius, last_x, ax, vis_args)
        rocket._draw_rail_buttons(ax, vis_args)
        rocket._draw_center_of_mass_and_pressure(ax)
        rocket._draw_sensors(ax, rocket.rocket.sensors, plane)

        radius = float(rocket.rocket.radius)

        ax.set_xlim()
        ax.set_ylim((-radius * 4, radius * 4))
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Radius [m]")

        return self.model.name


class RocketStateInfo(InfoBase[RocketState]):

    @scientific_table()
    def overview(self) -> pd.DataFrame:
        return self.model.to_df()


@dataclass
class RailButtonState(ComponentState[rocketpy.RailButtons]):
    upper_button_position: float
    lower_button_position: float
    angular_position: float
    diameter: float
    """RAS Aero II parameter for drag estimation"""
    height: float
    """RAS Aero II parameter for drag estimation"""

    def add(self, rocket: rocketpy.Rocket) -> rocketpy.RailButtons:
        return rocket.set_rail_buttons(
            upper_button_position=self.upper_button_position,
            lower_button_position=self.lower_button_position,
            angular_position=self.angular_position,  # type: ignore
        )


@dataclass
class RailButtonGenerator(ComponentGenerator[RailButtonState]):
    upper_button_position: Quantity
    lower_button_position: Quantity
    angular_position: Quantity
    diameter: float
    """RAS Aero II parameter for drag estimation"""
    height: float
    """RAS Aero II parameter for drag estimation"""

    _output_type = RailButtonState


@dataclass
class NoseState(ComponentState[rocketpy.NoseCone]):
    length: float
    kind: nose_types
    position: float
    power: float | None = None
    blunt_radius: float | None = None
    """RAS Aero II parameter for drag estimation"""

    def add(self, rocket: rocketpy.Rocket) -> rocketpy.NoseCone:
        return rocket.add_nose(
            length=self.length,
            kind=self.kind,
            position=self.position,
            power=self.power
        )


@dataclass
class NoseGenerator(ComponentGenerator[NoseState]):
    length: Quantity
    kind: nose_types
    position: Quantity
    power: float | None = None
    blunt_radius: float | None = None
    """RAS Aero II parameter for drag estimation"""

    _output_type = NoseState


@dataclass
class TrapezoidalFinsState(ComponentState[rocketpy.TrapezoidalFins]):
    n: int
    root_chord: float
    tip_chord: float
    span: float
    position: float
    sweep_length: float | None
    airfoil: airfoil_type = None

    thickness: float | None = None
    """RAS Aero II parameter for drag estimation"""
    leading_edge_radius: float | None = None
    """RAS Aero II parameter for drag estimation"""
    airfoil_type: rasaero.FinAirfoilSections = "Subsonic NACA"
    """RAS Aero II parameter for drag estimation"""

    def add(self, rocket: rocketpy.Rocket) -> rocketpy.TrapezoidalFins:
        return rocket.add_trapezoidal_fins(
            n=self.n,
            root_chord=self.root_chord,
            tip_chord=self.tip_chord,
            span=self.span,
            position=self.position,
            sweep_length=self.sweep_length,
            airfoil=self.airfoil
        )


@dataclass
class TrapezoidalFinsGenerator(ComponentGenerator[TrapezoidalFinsState]):
    n: int
    root_chord: Quantity
    tip_chord: Quantity
    span: Quantity
    position: Quantity
    sweep_length: Quantity | None = None
    airfoil: airfoil_type = None

    thickness: float | None = None
    """RAS Aero II parameter for drag estimation"""
    leading_edge_radius: float | None = None
    """RAS Aero II parameter for drag estimation"""
    airfoil_type: rasaero.FinAirfoilSections = "Subsonic NACA"
    """RAS Aero II parameter for drag estimation"""

    _output_type = TrapezoidalFinsState


@dataclass
class TailState(ComponentState[rocketpy.Tail]):
    top_diameter: float
    bottom_diameter: float
    length: float
    position: float

    @property
    def top_radius(self) -> float:
        return self.top_diameter / 2

    @property
    def bottom_radius(self) -> float:
        return self.bottom_diameter / 2

    def add(self, rocket: rocketpy.Rocket) -> rocketpy.Tail:
        return rocket.add_tail(
            top_radius=self.top_radius,
            bottom_radius=self.bottom_radius,
            length=self.length,
            position=self.position
        )


@dataclass
class TailGenerator(ComponentGenerator[TailState]):
    top_diameter: Quantity
    bottom_diameter: Quantity
    length: Quantity
    position: Quantity

    _output_type = TailState


@dataclass
class ParachuteState(ComponentState[rocketpy.Parachute]):
    name: str
    cd: float
    diameter: float
    trigger: parachute_types
    motor_delay: float | None
    deploy_altitude: float | None

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def cd_s(self) -> float:
        return self.cd * np.pi * self.radius ** 2

    def add(self, rocket: rocketpy.Rocket) -> rocketpy.Parachute:
        match self.trigger:
            case "apogee":
                trigger_func = "apogee"
            case "motor ejection":
                trigger_func = (
                    "time", rocket.motor.burn_out_time, self.motor_delay)
            case "altitude":
                trigger_func = self.deploy_altitude

        return rocket.add_parachute(
            name=self.name,
            cd_s=self.cd_s,
            trigger=trigger_func,
            radius=self.radius
        )


@dataclass
class ParachuteGenerator(ComponentGenerator[ParachuteState]):
    name: str
    cd: Quantity
    diameter: Quantity
    trigger: parachute_types
    motor_delay: Quantity | None = None
    deploy_altitude: Quantity | None = None

    _output_type = ParachuteState


@dataclass
class MotorState(ComponentState[rocketpy.SolidMotor]):
    motor: Motor
    position: float
    thrust_multiplier: float
    time_multiplier: float
    mass_multiplier: float

    nozzle_diameter: float | None = None
    """RAS Aero II parameter for drag estimation"""

    def add(self, rocket: rocketpy.Rocket) -> rocketpy.SolidMotor:

        # TODO: Adjust thrust curve with thrust and time multiplier (within motor.py)

        rocketpy_motor, position_offset = self.motor.to_SolidMotor()
        rocket.add_motor(
            motor=rocketpy_motor,
            position=self.position - position_offset
        )
        return rocketpy_motor


@dataclass
class MotorGenerator(ComponentGenerator[MotorState]):
    motor: Motor
    position: Quantity
    thrust_multiplier: Quantity
    time_multiplier: Quantity
    mass_multiplier: Quantity

    nozzle_diameter: float | None = None
    """RAS Aero II parameter for drag estimation"""

    _output_type = MotorState


@dataclass
class LauncherState(AssemblyState):
    length: float
    inclination: float
    heading: float


@dataclass
class LauncherGenerator(ComponentGenerator[LauncherState]):
    length: Quantity
    heading: Quantity
    inclination: Quantity

    _output_type = LauncherState

    def __init__(
        self,
        length: Q = 2,
        heading: Q = 0,
        inclination: Q = 90,
    ) -> None:
        self.length = _handle_input(
            length,
            default=lambda x: Normal.from_mean(x, 10 / 1000)
        )
        self.heading = _handle_input(
            heading,
            default=lambda x: Normal.from_mean(x, 10)
        )
        self.inclination = _handle_input(
            inclination,
            default=lambda x: Normal.from_mean(x, 2)
        )

#
# Environment
#


@dataclass
class EnvironmentState(AssemblyState[rocketpy.Environment]):
    latitude: float
    longitude: float
    elevation: float
    atmospheric_model: "AtmosphericModelState | None" = None

    def make(self) -> rocketpy.Environment:
        env = rocketpy.Environment(
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation
        )

        if self.atmospheric_model is not None:
            self.atmospheric_model.add(env)

        return env


@dataclass
class EnvironmentGenerator(ComponentGenerator[EnvironmentState]):
    latitude: Quantity
    longitude: Quantity
    elevation: Quantity
    atmospheric_model: "AtmosphericModelGenerator | None" = None

    _output_type = EnvironmentState

    def __init__(
        self,
        latitude: Q,
        longitude: Q,
        elevation: Q = 0
    ) -> None:
        self.latitude = _handle_input(latitude)
        self.longitude = _handle_input(longitude)
        self.elevation = _handle_input(
            elevation,
            default=lambda x: Normal.from_mean(x, 5)
        )

    @overload
    def set_atmospheric_model(self, *, wind_speed: Q, wind_bearing: Q):
        ...

    @overload
    def set_atmospheric_model(self, *, wind_u: Q, wind_v: Q):
        ...

    def set_atmospheric_model(
        self,
        *,
        wind_u: Q | None = None,
        wind_v: Q | None = None,
        wind_speed: Q | None = None,
        wind_bearing: Q | None = None
    ):
        if wind_u is not None and wind_v is not None:
            self.atmospheric_model = AtmosphericModelGenerator(
                wind_u=_handle_input(
                    wind_u,
                    default=lambda X: Normal.from_mean(X, 1)
                ),
                wind_v=_handle_input(
                    wind_v,
                    default=lambda X: Normal.from_mean(X, 1)
                ),
            )
        elif wind_speed is not None and wind_bearing is not None:
            self.atmospheric_model = AtmosphericModelGenerator(
                wind_speed=_handle_input(
                    wind_speed,
                    default=lambda X: Normal.from_mean(X, np.sqrt(2))
                ),
                wind_bearing=_handle_input(
                    wind_bearing,
                    default=STANDARD_RADIAL_ANGLE_DIST
                )
            )
        else:
            raise ValueError(
                "either wind_u and wind_v or wind_speed and wind_bearing must be provided"
            )


@dataclass
class AtmosphericModelState:
    wind_u: float | None = None
    wind_v: float | None = None
    wind_speed: float | None = None
    wind_bearing: float | None = None
    temperature: None = None
    pressure: None = None
    type: str = "custom_atmosphere"

    @property
    def abs_wind_speed(self) -> float:
        if self.wind_speed is not None:
            return self.wind_speed
        elif self.wind_u is not None and self.wind_v is not None:
            return np.sqrt(self.wind_u**2 + self.wind_v**2)
        else:
            raise ValueError(
                "either wind_u and wind_v or wind_speed and wind_bearing must be provided"
            )

    def add(self, env: rocketpy.Environment):
        if self.wind_u is not None and self.wind_v is not None:
            env_wind_u = self.wind_u
            env_wind_v = self.wind_v
        elif self.wind_speed is not None and self.wind_bearing is not None:
            env_wind_u: float = self.wind_speed * \
                np.sin(np.deg2rad(self.wind_bearing))
            env_wind_v: float = self.wind_speed * \
                np.cos(np.deg2rad(self.wind_bearing))
        else:
            env_wind_u = 0
            env_wind_v = 0

        env.set_atmospheric_model(
            type=self.type,
            pressure=self.pressure,
            temperature=self.temperature,
            wind_u=env_wind_u,  # type: ignore
            wind_v=env_wind_v,  # type: ignore
        )


@dataclass
class AtmosphericModelGenerator(ComponentGenerator[AtmosphericModelState]):
    wind_u: Quantity | None = None
    wind_v: Quantity | None = None
    wind_speed: Quantity | None = None
    wind_bearing: Quantity | None = None
    temperature: None = None
    pressure: None = None
    type: str = "custom_atmosphere"

    _output_type = AtmosphericModelState

#
# Flight
#


@dataclass
class FlightState(AssemblyState):
    category = "flight-state"

    rocket: RocketState
    env: EnvironmentState
    launcher: LauncherState
    rasaero_config: RASAeroConfig | None = None

    def __post_init__(self):
        self.plots = FlightStatePlots(self)
        self.info = FlightStateInfo(self)

    @property
    def name(self) -> str:
        return self.rocket.name

    def to_rasaero(
        self,
    ) -> rasaero.RASAeroDocument:
        from model.rocket.rasaero import m_to_ft, C_to_F, K_to_C, m_to_in, Pa_to_inhg, ms_to_mph, kg_to_lb, K_to_F, RAIL_BUTTON_DISTANCE

        rocket = self.rocket
        env = self.env
        launcher = self.launcher

        if self.rasaero_config is None:
            raise ValueError("RASAero Config must be provided")
        rasaero_config = self.rasaero_config

        rocketpy_rocket = rocket.make()
        rocketpy_env = env.make()

        if rocket.nose is None:
            raise ValueError("nose must be provided")

        if rocket.fins is None:
            raise ValueError("fins must be provided")

        nose_map = {
            "von karman": "Von Karman Ogive",
            "ogive": "Tangent Ogive"
        }
        nose_shape = nose_map.get(rocket.nose.kind, None)
        if nose_shape is None:
            raise ValueError(
                f"{rocket.nose.kind} is not supported/implemented")

        if rocket.nose.blunt_radius is None:
            raise ValueError("Nose blunt radius must be provided")

        nose = rasaero._NoseCone(
            Length=m_to_in(rocket.nose.length),
            Diameter=m_to_in(rocket.diameter),
            Shape=nose_shape,  # type: ignore
            BluntRadius=m_to_in(rocket.nose.blunt_radius),
        )
        cumulative_length_in = nose.Length

        body_tubes = []
        if rocket.rail_buttons is not None:
            fwd_body_section_length = rocket.rail_buttons.upper_button_position - \
                rocket.nose.length - RAIL_BUTTON_DISTANCE
            fwd_body_section = rasaero._BodyTube(
                Length=m_to_in(fwd_body_section_length),
                Diameter=m_to_in(rocket.diameter),
                Location=cumulative_length_in
            )
            cumulative_length_in += fwd_body_section.Length
            body_tubes.append(fwd_body_section)
        else:
            fwd_body_section_length = 0

        if rocket.rail_buttons is None:
            rail_button_diameter = 0
            rail_button_height = 0
        else:
            rail_button_diameter = rocket.rail_buttons.diameter
            rail_button_height = rocket.rail_buttons.height

        if rocket.fins.thickness is None:
            raise ValueError("Fin thickness must be provided")

        if rocket.fins.leading_edge_radius is None:
            raise ValueError("Fin leading edge radius must be provided")

        aft_body_section = rasaero._BodyTube(
            Length=m_to_in(rocket.body_length - fwd_body_section_length),
            Diameter=m_to_in(rocket.diameter),
            Location=cumulative_length_in,
            Fin=rasaero._Fin(
                Count=rocket.fins.n,
                Chord=m_to_in(rocket.fins.root_chord),
                Span=m_to_in(rocket.fins.span),
                SweepDistance=m_to_in(rocket.fins.sweep_length),
                TipChord=m_to_in(rocket.fins.tip_chord),
                Thickness=m_to_in(rocket.fins.thickness),
                LERadius=m_to_in(rocket.fins.leading_edge_radius),
                Location=m_to_in(rocket.fins.root_chord),
                AirfoilSection=rocket.fins.airfoil_type,
            ),
            RailGuideDiameter=m_to_in(rail_button_diameter),
            RailGuideHeight=m_to_in(rail_button_height),
        )
        cumulative_length_in += aft_body_section.Length
        body_tubes.append(aft_body_section)

        if rocket.tails:
            if len(rocket.tails) == 1:
                _tail = rocket.tails[0]
                tail = rasaero._BoatTail(
                    Length=m_to_in(_tail.length),
                    Diameter=m_to_in(_tail.top_diameter),
                    Location=cumulative_length_in,
                    RearDiameter=m_to_in(_tail.bottom_diameter)
                )
                transition = None
                cumulative_length_in += tail.Length
            elif len(rocket.tails) == 2:
                sorted_tails = sorted(rocket.tails, key=lambda x: x.position)
                _transition = sorted_tails[0]
                _tail = sorted_tails[1]

                transition = rasaero._Transition(
                    Length=m_to_in(_transition.length),
                    Diameter=m_to_in(_transition.bottom_diameter),
                    # ! This works and is required due to a RAS Aero II bug
                    Location=cumulative_length_in,
                    RearDiameter=m_to_in(_transition.bottom_diameter)
                )
                cumulative_length_in += transition.Length

                tail = rasaero._BoatTail(
                    Length=m_to_in(_tail.length),
                    Diameter=m_to_in(_tail.top_diameter),
                    Location=cumulative_length_in,
                    RearDiameter=m_to_in(_tail.bottom_diameter)
                )
                cumulative_length_in += tail.Length
            else:
                raise ValueError("Only 0, 1 or 2 tails supported")
        else:
            tail = None
            transition = None

        if env.atmospheric_model is None:
            raise ValueError(f"Atmospheric model must be provided")

        launch_pressure = float(
            rocketpy_env.pressure(env.elevation)  # type: ignore
        )
        launch_temperature = float(
            rocketpy_env.temperature(env.elevation)  # type: ignore
        )

        launch_site = rasaero._LaunchSite(
            Altitude=m_to_ft(env.elevation),
            Pressure=Pa_to_inhg(launch_pressure),
            RodAngle=90 - launcher.inclination,
            RodLength=m_to_ft(launcher.length),
            Temperature=K_to_F(launch_temperature),
            WindSpeed=ms_to_mph(env.atmospheric_model.abs_wind_speed)
        )

        if len(rocket.parachutes) > 2:
            raise ValueError("Maximum of 2 recovery events supported")

        sorted_parachutes = sorted(
            rocket.parachutes, key=lambda x: x.trigger != "apogee")

        if len(sorted_parachutes) == 0:
            recovery = None
        else:
            if len(sorted_parachutes) == 1:
                main = sorted_parachutes[0]
                drogue = None
            else:
                main = sorted_parachutes[0]
                drogue = sorted_parachutes[1]

                event_type_map = {
                    "apogee": "Apogee",
                    "altitude": "Altitude"
                }

                event_type_1 = event_type_map.get(main.trigger, None)
                if event_type_1 is None:
                    raise ValueError(f"{main.trigger} is not supported")

                if drogue is None:
                    event_type_2 = "Apogee"
                else:
                    event_type_2 = event_type_map.get(drogue.trigger, None)
                    if event_type_2 is None:
                        raise ValueError(f"{drogue.trigger} is not supported")

                recovery = rasaero._Recovery(
                    Altitude1=0 if main.deploy_altitude is None else m_to_ft(
                        main.deploy_altitude),
                    Size1=m_to_in(main.diameter),
                    EventType1=event_type_1,  # type: ignore
                    CD1=main.cd,
                    Altitude2=0 if drogue is None else (
                        0 if drogue.deploy_altitude is None else m_to_ft(
                            drogue.deploy_altitude)
                    ),
                    Size2=0 if drogue is None else m_to_in(drogue.diameter),
                    EventType2=event_type_2,  # type: ignore
                    CD2=0 if drogue is None else drogue.cd,
                )

        mach_alt = rasaero._MachAlt(
            Item=[]
        )

        cg = float(rocketpy_rocket.evaluate_center_of_mass()(0))

        if rocket.motor is None:
            raise ValueError("Motor must be provided")

        # automatic nozzle exit diameter estimation
        if rocket.motor.nozzle_diameter is None:
            nozzle_diameter = rocket.diameter * 0.4
        else:
            nozzle_diameter = rocket.motor.nozzle_diameter

        launch_mass = rocket.mass_without_motor + rocket.motor.motor.total_mass

        simulation_list = rasaero._SimulationList(
            Simulation=[rasaero._Simulation(
                SustainerEngine=rocket.motor.motor.name,
                SustainerLaunchWt=kg_to_lb(launch_mass),
                SustainerNozzleDiameter=m_to_in(nozzle_diameter),
                SustainerCG=m_to_in(cg),
            )]
        )

        cp = float(
            rocketpy_rocket.evaluate_center_of_pressure()(0.3)  # type: ignore
        )

        return rasaero.RASAeroDocument(
            RocketDesign=rasaero._RocketDesign(
                NoseCone=nose,
                BodyTube=body_tubes,
                Transition=transition,
                BoatTail=tail,
                Surface=rasaero_config.surface,
                CP=m_to_in(cp),
                ModifiedBarrowman=rasaero_config.use_modified_barrowman,
                Turbulence=rasaero_config.turbulence_only_flight,
                SustainerNozzle=m_to_in(nozzle_diameter),
            ),
            LaunchSite=launch_site,
            Recovery=recovery,
            MachAlt=mach_alt,
            SimulationList=simulation_list
        )


class FlightStatePlots(PlotBase[FlightState]):
    pass


class FlightStateInfo(InfoBase[FlightState]):

    @scientific_table()
    def overview(self) -> pd.DataFrame:
        return self.model.to_df()


@dataclass
class FlightGenerator(ComponentGenerator[FlightState]):
    rocket: RocketGenerator
    env: EnvironmentGenerator
    launcher: LauncherGenerator
    rasaero_config: RASAeroConfig | None = None

    _output_type = FlightState

    def __post_init__(self):
        if self.rocket.aero == "estimate":
            if self.rasaero_config is None:
                raise ValueError(
                    "rasaero_config must be provided with aero == 'estimate'"
                )
            rasaero_config = self.rasaero_config

            flight_state = self.ideal()

            from model.rocket.rasaero import RasAeroII

            # temporary set to keep rocketpy happy
            self.rocket.power_off_drag = 1  # type: ignore
            self.rocket.power_on_drag = 1  # type: ignore

            # convert to rasaero document
            rasaero_doc = RasAeroII(self.name, self.ideal().to_rasaero())

            aero_data_path = rasaero_doc.data_path / "aeroData.csv"
            aero_data_hash_path = rasaero_doc.data_path / "hash.json"

            # update rasaero aero plots based if hash changes (or is missing)
            if not aero_data_path.exists():
                print("cache missing: generating new RASAero aero data")
                rasaero_doc.extract_aero_plots()
            elif not aero_data_hash_path.exists():
                print("hash missing: generating new RASAero aero data")
                rasaero_doc.extract_aero_plots()
            elif aero_data_hash_path.read_text(encoding="utf-8") != flight_state.hash():
                print("hash mismatch: generating new RASAero aero data")
                rasaero_doc.extract_aero_plots()
            else:
                print("Using cached RASAero aero data")

            self.rocket.aero = Aero(aero_data_path)
            self.rocket.power_on_drag = self.rocket.aero.power_on_drag()  # type: ignore
            self.rocket.power_off_drag = self.rocket.aero.power_off_drag()  # type: ignore

            # write hash to file
            aero_data_hash_path.write_text(
                flight_state.hash(), encoding="utf-8")

    @property
    def name(self) -> str:
        return self.rocket.name

# TODO: Read and write to file
# TODO: Implement Monte Carlo Method
# TODO: Implement failures (as a method parameter)


# TODO: Make _handle a class method?
# TODO: Overload Normal and Uniform to support all reasonable inputs
# TODO: Add default variation (using `rocket` as an input to scale appropriately`)
# TODO: Add Aero.powerOn()... wrapper to support ideal and sample
# TODO: Add add_eccentricity...


# TODO: Add categories and default distributions for all
