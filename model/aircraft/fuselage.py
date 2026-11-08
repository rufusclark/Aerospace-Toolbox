from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from model.aircraft.base import Position, Mass
from model.aircraft.material import Material
from model.aircraft.spar import LongitudinalSpar
from model.util.dataclass import FrozenDataclassBase
if TYPE_CHECKING:
    from model.aircraft.aircraft import Aircraft


@dataclass(frozen=True)
class FuselageSizing(FrozenDataclassBase):
    aircraft: "Aircraft"

    material: Material
    """Fuselage Material"""

    width: float
    """fuselage width [m]"""
    height: float
    """fuselage height [m]"""

    top_thickness: float
    """fuselage top thickness [m]"""
    bottom_thickness: float
    """fuselage bottom thickness [m]"""
    left_thickness: float
    """spar left side thickness [m]"""
    right_thickness: float
    """spar right side thickness [m]"""

    aft_length: float
    """length [m] aft of furthest of either tail or fin trailing edge"""
    fwd_length: float
    """length [m] fwd of the furthest fwd of main wing leading edge and payload"""

    offset: Position
    """position offset.
    
    Typically used to offset vertical (z)"""

    def size_fuselage(self) -> Fuselage:
        """generate an appropriately sized fuselage based on defined requirements"""
        if (
            not hasattr(self.aircraft, "port_tail_wing") and
            not hasattr(self.aircraft, "fin_wing")
        ) or not hasattr(self.aircraft, "port_main_wing"):
            raise RuntimeError(
                "Main Wing and at least the tail or fin must be added first"
            )

        payload_leading_edge = (self.aircraft.payload.center.x +
                                self.aircraft.payload.dx / 2) if hasattr(self.aircraft, "payload") else 0
        main_wing_leading_edge = self.aircraft.port_main_wing.airfoil3D.leading_edge.x
        leading_edge = max(payload_leading_edge, main_wing_leading_edge)

        tail_trailing_edge = (
            self.aircraft.port_tail_wing.airfoil3D.trailing_edge.x
            if hasattr(self.aircraft, "port_tail_wing") else
            float("inf")
        )
        fin_trailing_edge = (
            self.aircraft.fin_wing.airfoil3D.trailing_edge.x
            if hasattr(self.aircraft, "fin_wing") else
            float("inf")
        )
        trailing_edge = min(tail_trailing_edge, fin_trailing_edge)

        length = (
            self.fwd_length +
            leading_edge - trailing_edge +
            self.aft_length
        )

        return Fuselage(
            mass=Mass(),
            center=(
                self.offset +
                Position(
                    x=-(0.5 * length) + self.fwd_length + payload_leading_edge
                )
            ),
            material=self.material,
            external_height=self.height,
            external_width=self.width,
            length=length,
            top_thickness=self.top_thickness,
            bottom_thickness=self.bottom_thickness,
            left_thickness=self.left_thickness,
            right_thickness=self.right_thickness
        )

    @property
    def A(self) -> float:
        """cross-sectional area of fuselage"""
        return self.width * self.height

    @property
    def CD_0(self) -> float:
        """Zero Lift Drag Coefficient of fuselage using the main wing area as a reference

        Uses the frontal are method. Very first-order approximation based on Cd here, https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/shape-effects-on-drag/
        """
        if not hasattr(self.aircraft, "port_main_wing"):
            return 0.0

        CD_0 = 0.07  # drag coefficient of a sphere
        A_f = self.A
        A_ref = self.aircraft.S_main
        return CD_0 * (A_f / A_ref)


@dataclass(frozen=True)
class Fuselage(LongitudinalSpar):
    component_name: str = field(init=False, default="Fuselage")

    @property
    def leading_edge(self) -> Position:
        """leading edge position along the center of the fuselage"""
        return self.center + Position(x=0.5 * self.length)

    @property
    def trailing_edge(self) -> Position:
        """trailing edge position along the center of the fuselage"""
        return self.center + Position(x=-0.5 * self.length)
