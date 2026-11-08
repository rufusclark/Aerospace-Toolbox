from dataclasses import dataclass, field
from model.util.dataclass import FrozenDataclassBase
from model.aircraft.component import MassComponent
from model.aircraft.base import Position


@dataclass(frozen=True)
class Payload(MassComponent):
    component_name: str = field(init=False, default="Payload")


@dataclass(frozen=True)
class PayloadSizing(FrozenDataclassBase):
    payload: Payload

    SM: float
    """target static margin (/c) through longitudinal placement"""

    offset: Position = Position()
    """position offset"""
