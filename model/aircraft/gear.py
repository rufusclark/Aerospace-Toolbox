from dataclasses import dataclass, field
from model.aircraft.component import MassComponent
from model.aircraft.base import Position
from model.util.dataclass import FrozenDataclassBase
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from model.aircraft.aircraft import Aircraft


@dataclass(frozen=True)
class Gear(MassComponent):
    CD_0: float = field(kw_only=True)
    """Zero Lift Drag Coefficient of gear using the main wing area as a reference"""
    fixed_center: Position = field(init=False, default=Position())
    dynamic_center: None = field(init=False, default=None)
    component_name: str = field(init=False, default="Landing Gear")


gear_configuration = Literal["fwd tricycle"]


@dataclass(frozen=True)
class GearSizing(FrozenDataclassBase):
    aircraft: "Aircraft"

    configuration: gear_configuration

    nose_weighting: float
    """nose gear weight weighting"""
    main_weighting: float
    """nose gear weight weighting"""

    gear: Gear
    """landing gear mass component (for a single gear)"""

    main_position: float
    """main gear position relative to aircraft cg [m]
    
    uses the aircraft body reference frame directions (negative is aft)"""
    lateral_position: float
    """main gear lateral position [m] from center of fuselage"""

    g: float = 9.81
    """gravitational acceleration [m/s^2]"""

    def __post_init__(self):
        if self.configuration != "fwd tricycle":
            raise ValueError(f"{self.configuration} not supported")

    @property
    def aircaft_weight(self) -> float:
        """aircraft weight [N]"""
        return self.aircraft.mass.mass * self.g

    @property
    def nose_weight(self) -> float:
        """nose loading weight [N]"""
        W = self.aircaft_weight
        x = (self.nose_weighting) / (self.nose_weighting + self.main_weighting)
        return W * x

    @property
    def main_weight(self) -> float:
        """main loading weight [N]"""
        W = self.aircaft_weight
        x = (self.main_weighting) / (self.nose_weighting + self.main_weighting)
        return W * x

    @property
    def nose_position(self) -> float:
        """nose position in x-direction [m]"""
        return -self.main_position * (self.main_weighting / self.nose_weighting)

    def get_main_gear(self) -> Gear:
        if not hasattr(self.aircraft, "fuselage"):
            raise ValueError(f"fuselage must be defined")

        return self.gear.copy(
            name="Main Gear",
            offset=Position(
                x=self.aircraft.cg + self.main_position,
                y=self.lateral_position,
                z=(
                    self.aircraft.fuselage.center.z -
                    self.aircraft.fuselage.external_height / 2 -
                    self.gear.dz / 2
                )
            )
        )

    def get_nose_gear(self) -> Gear:
        if not hasattr(self.aircraft, "fuselage"):
            raise ValueError(f"fuselage must be defined")

        return self.gear.copy(
            name="Nose Gear",
            offset=Position(
                x=self.aircraft.cg + self.nose_position,
                y=0,
                z=(
                    self.aircraft.fuselage.center.z -
                    self.aircraft.fuselage.external_height / 2 -
                    self.gear.dz / 2
                )
            )
        )
