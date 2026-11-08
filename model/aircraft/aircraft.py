from __future__ import annotations

from model.util.output import *

from model.aircraft.propulsion import PropulsionSizing, LiPo
from model.aircraft.base import ParentComponent, Position, ComponentPlots, ComponentInfo
from model.aircraft.component import MassComponent
from model.aircraft.wing import LateralWing, VerticalWing
from model.aircraft.tail import TailSizing
from model.aircraft.fin import FinSizing
from model.aircraft.fuselage import Fuselage, FuselageSizing
from model.aircraft.payload import PayloadSizing, Payload
from model.aircraft.gear import GearSizing, Gear

from dataclasses import field, dataclass
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import math


@dataclass(frozen=True)
class Aircraft(ParentComponent):
    category = "aircraft"

    verbose: bool = False
    """verbose output (i.e. optimisation steps)"""

    port_main_wing: LateralWing = field(init=False)
    starboard_main_wing: LateralWing = field(init=False)

    port_tail_wing: LateralWing = field(init=False)
    starboard_tail_wing: LateralWing = field(init=False)

    fin_wing: VerticalWing = field(init=False)

    fuselage: Fuselage = field(init=False)

    gear: dict[str, Gear] = field(init=False, default_factory=lambda: dict())

    lipos: list[LiPo] = field(init=False, default_factory=lambda: list())

    payload: Payload = field(init=False)

    mass_components: dict[str, MassComponent] = field(
        init=False, default_factory=lambda: dict())

    n_max: int = 100
    """maximum interactions for converging on aircraft geometry"""
    abs_tol: float = 1e-3
    """absolute tolerance for stop condition in converging methods"""
    static_margin_abs_tol: float = 1e-1
    """absolute tolerance for static margin stop condition in converging methods"""

    _props: PropulsionSizing = field(init=False, repr=False)
    _tail: TailSizing = field(init=False, repr=False)
    _fin: FinSizing = field(init=False, repr=False)
    _fuselage: FuselageSizing = field(init=False, repr=False)
    _payload: PayloadSizing = field(init=False, repr=False)
    _gear: GearSizing = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        self._set("plots", AircraftPlots(self))
        self._set("info", AircraftInfo(self))

    #
    # Properties and methods
    #

    @property
    def S_main(self) -> float:
        """Main wing area [m^2]"""
        return self.port_main_wing.airfoil3D.solvers.S_full

    def CL(self, alpha: float) -> float:
        """Whole aircraft Lift Coefficient at angle of attack, alpha [deg]"""
        return self.port_main_wing.airfoil3D.solvers.lifting_line_theory.CL(alpha)

    def CD_0(self, alpha: float) -> float:
        """Whole aircraft Zero-Lift Drag Coefficient at angle of attack, alpha [deg]"""
        S_main = self.port_tail_wing.airfoil3D.solvers.S_full

        CD_0_main = (
            self.port_main_wing.airfoil3D
            .solvers.lifting_line_theory.CD_profile(alpha)
        )

        S_tail = self.port_tail_wing.airfoil3D.solvers.S_full
        CD_0_tail = (
            self.port_tail_wing.airfoil3D
            .solvers.lifting_line_theory.CD_profile(alpha)
        ) * (S_tail / S_main)

        S_fin = self.fin_wing.airfoil3D.solvers.S_full
        CD_0_fin = (
            self.fin_wing.airfoil3D
            .solvers.lifting_line_theory.CD_profile(0)
        ) * (S_fin / S_main)

        CD_0_fuselage = (
            self._fuselage.CD_0 if hasattr(self, "_fuselage") else 0)
        CD_0_landing_gear = sum(
            gear.CD_0 for gear in self.gear.values()
        )

        CD_0 = CD_0_main + CD_0_tail + CD_0_fin + CD_0_fuselage + CD_0_landing_gear

        return CD_0

    @property
    def k(self) -> float:
        """Whole aircraft induced drag factor at angle of attack, alpha [deg]"""
        main_wing = self.port_main_wing.airfoil3D.solvers
        AR = main_wing.AR
        e = main_wing.lifting_line_theory.e

        return 1 / (math.pi * e * AR)

    def CD(self, alpha: float) -> float:
        """Whole aircraft drag coefficient at angle of attack, alpha [deg]"""
        return self.CD_0(alpha) + self.k * self.CL(alpha)**2

    @property
    def CL_max(self) -> float:
        """estimated maximum Lift Coefficient, CL_max"""
        return self.port_main_wing.airfoil3D.solvers.lifting_line_theory.estimated_CL_max()

    @property
    def cg(self) -> float:
        """center of gravity, cg [m] position along the x-axis"""
        return self.mass.cg.x

    @property
    def np(self) -> float:
        """estimate neutral point, np [m] position along the x-axis based on first order approximations. See implementation comments"""
        # main wing aerodynamic center [m]
        x_ac_w = self.port_main_wing.airfoil3D.aerodynamic_center.x
        # tail wing aerodynamic center [m]
        x_ac_t = self.port_tail_wing.airfoil3D.aerodynamic_center.x
        # main wing mean aerodynamic chord [m]
        c_w = self.port_main_wing.airfoil3D.chord
        # tail dynamic pressure ratio
        # approximate as 1 for initial design
        eta_t = 1
        # tail lift curve slope [1/rad]
        a_t = self.port_tail_wing.airfoil3D.solvers.finite_wing.a
        # main wing lift curve slope [1/rad]
        a_w = self.port_main_wing.airfoil3D.solvers.finite_wing.a
        # main wing aspect ratio
        AR_w = self.port_main_wing.airfoil3D.solvers.AR
        # main wing area [m^2]
        S_w = self.port_main_wing.airfoil3D.solvers.S_full
        # tail area [m^2]
        S_t = self.port_tail_wing.airfoil3D.solvers.S_full

        # tail arm (between main wing and tail aerodynamic centers) [m]
        l_t = x_ac_w - x_ac_t

        # horizontal tail volume coefficient
        V_H = (S_t * l_t) / (S_w * c_w)

        # downwash gradient
        downwash_gradient = self.port_main_wing.downwash_gradient

        return x_ac_w - c_w * eta_t * V_H * (a_t/a_w) * (1-downwash_gradient)

    @property
    def SM(self) -> float:
        """static margin (/chord) position along the x-axis"""
        return (self.cg - self.np) / self.port_main_wing.airfoil3D.chord

    @property
    def estimate_stall_speed(self) -> float:
        """returns the estimated theoretical stall speed"""
        g = 9.81
        rho = 1.225  # [kg/m^3]

        return math.sqrt((2 * self.mass.mass * g) / (rho * self.S_main * self.CL_max))

    @property
    def optimal_range_speed(self) -> float:
        """returns the theoretical optimal speed to maximise range"""
        g = 9.81
        rho = 1.225  # [kg/m^3]
        alpha = 0  # [deg]

        return math.sqrt(
            (2 * self.mass.mass * g) / (rho * self.S_main)
        ) * math.pow((self.k / self.CD_0(alpha)), 0.25)

    @property
    def optimal_endurance_speed(self) -> float:
        """returns the theoretical optimal speed to maximise endurance"""
        return self.optimal_range_speed / math.pow(3, 0.25)

    #
    # Add geometry
    #

    def add_main_wing(self, wing: LateralWing) -> None:
        if not hasattr(self, "_fuselage"):
            raise RuntimeError("Fuselage must be added first")

        port_wing = wing.copy("Port Main Wing", Position(
            y=-(self._fuselage.width + wing.spar.length) / 2))
        starboard_wing = wing.copy("Starboard Main Wing", Position(
            y=(self._fuselage.width + wing.spar.length) / 2))
        self._set("port_main_wing", port_wing)
        self._set("starboard_main_wing", starboard_wing)

    def add_mass_component(self, mass_component: MassComponent, *, mirror_laterally: bool = False) -> None:
        if mirror_laterally:
            if mass_component.center.y > 0:
                name1 = f"Starboard {mass_component.component_name}"
                name2 = f"Port {mass_component.component_name}"
            else:
                name1 = f"Port {mass_component.component_name}"
                name2 = f"Starboard {mass_component.component_name}"

            offset = Position(y=-2 * mass_component.center.y)

            comp1 = mass_component.copy(name=name1)
            comp2 = mass_component.copy(name=name2, offset=offset)

            self.add_mass_component(comp1)
            self.add_mass_component(comp2)
        else:
            self.mass_components[mass_component.component_name] = mass_component

    def add_gear(self, gear: GearSizing) -> None:
        self._set("_gear", gear)
        self.__size_gear()

    def __add_gear(self, gear: Gear, *, mirror_laterally: bool = False) -> None:
        if mirror_laterally:
            if gear.center.y > 0:
                name1 = f"Starboard {gear.component_name}"
                name2 = f"Port {gear.component_name}"
            else:
                name1 = f"Port {gear.component_name}"
                name2 = f"Starboard {gear.component_name}"

            offset = Position(y=-2 * gear.center.y)

            comp1 = gear.copy(name=name1)
            comp2 = gear.copy(name=name2, offset=offset)

            self.__add_gear(comp1)
            self.__add_gear(comp2)
        else:
            self.gear[gear.component_name] = gear

    def __size_gear(self) -> None:
        """recursively reposition landing gear until desired wight distribution is achieved"""
        if not hasattr(self, "_gear"):
            raise RuntimeError(f"Gear Sizing must be created first")

        gear = self._gear

        last_main_position = 0
        for i0 in range(self.n_max):
            # add main gear

            self.__add_gear(
                gear=gear.get_main_gear(),
                mirror_laterally=True
            )

            last_nose_position = 0
            for i1 in range(self.n_max):
                # add nose gear

                self.__add_gear(
                    gear=gear.get_nose_gear(),
                    mirror_laterally=False
                )

                nose_position = self.gear["Nose Gear"].center.x
                d_nose_position = abs(last_nose_position - nose_position)

                if self.verbose:
                    print(
                        f"[nose gear] iter: {i1}, nose gear position delta: {d_nose_position*1e3:.2f}mm"
                    )

                if d_nose_position < self.abs_tol:
                    break
                last_nose_position = nose_position
            else:
                raise RuntimeError("Nose Gear Position did not converge")

            main_position = self.gear["Port Main Gear"].center.x
            d_main_position = abs(last_main_position - main_position)

            if self.verbose:
                print(
                    f"[main gear] iter: {i0}, main gear position delta: {d_main_position*1e3:.2f}mm"
                )

            if d_main_position < self.abs_tol:
                return
            last_main_position = main_position
        else:
            raise RuntimeError("Main Gear Position did not converge")

    def add_lipos(self, prop: PropulsionSizing) -> None:
        self._set("_props", prop)
        self.__size_propulsion_system()

    def __size_propulsion_system(self) -> None:
        """recursively resize aircraft batteries until the mass converges"""
        if not hasattr(self, "_props"):
            raise RuntimeError("Propulsion Sizing must be created first")

        prop = self._props

        last_mass = 0
        for _ in range(self.n_max):

            lipos: list[LiPo] = []
            for i in range(prop.LiPos_req):
                lipo_to_add = prop.lipo.copy(
                    name=f"LiPo {i+1}",
                    offset=Position(x=-prop.lipo.dx * i)
                )

                lipos.append(lipo_to_add)
            self._set("lipos", lipos)

            # end condition
            mass = self.mass.mass
            error = abs(last_mass - mass)

            if self.verbose:
                print(
                    f"[prop] iter: {i}, battery mass error: {error*1e3:.2f}g"
                )

            if abs(error) < self.abs_tol:
                return
            last_mass = mass
        else:
            raise RuntimeError("Battery Sizing did not converge")

    def add_tail(self, tail: TailSizing) -> None:
        self._set("_tail", tail)
        self.__size_tail()

    def __size_tail(self) -> None:
        """recursively resize/reposition tail until the target tail arm is achieved"""

        if not hasattr(self, "_fuselage"):
            raise RuntimeError("Fuselage must be added first")

        if not hasattr(self, "_tail"):
            raise RuntimeError("Tail must be created first")

        tail = self._tail
        target_tail_arm = tail.arm

        for i in range(self.n_max):
            wing = tail.optimise_wing()

            x = - tail.arm + self.mass.cg.x - wing.airfoil3D.aerodynamic_center.x
            port_wing = wing.copy("Port Tail", Position(
                x=x, y=-(self._fuselage.width + wing.spar.length) / 2)
            )
            starboard_wing = wing.copy("Starboard Tail", Position(
                x=x, y=(self._fuselage.width + wing.spar.length) / 2)
            )
            self._set("port_tail_wing", port_wing)
            self._set("starboard_tail_wing", starboard_wing)
            self.__size_fuselage()

            # end_condition
            actual_tail_arm = abs(
                self.port_tail_wing.airfoil3D.aerodynamic_center.x - self.mass.cg.x)
            error = abs(target_tail_arm - actual_tail_arm)

            if self.verbose:
                print(f"[tail] iter: {i}, tail arm error: {error*1e3:.2f}mm")

            if error < self.abs_tol:
                return
        else:
            raise RuntimeError("Aircraft Tail Arm location did not converge")

    def add_fin(self, fin: FinSizing) -> None:
        self._set("_fin", fin)
        self.__size_fin()

    def __size_fin(self) -> None:
        """recursively resize/reposition fin until the target tail arm is achieved"""

        if not hasattr(self, "_fuselage"):
            raise RuntimeError("Fuselage must be added first")

        if not hasattr(self, "_fin"):
            raise RuntimeError("Fin must be created first")

        fin = self._fin
        target_fin_arm = fin.arm

        for i in range(self.n_max):
            wing = fin.optimise_wing()

            x = - fin.arm + self.mass.cg.x - wing.airfoil3D.aerodynamic_center.x

            fin_wing = wing.copy(
                name="Fin",
                offset=Position(
                    x=x,
                    y=0,
                    z=(
                        self.fuselage.center.z +
                        self.fuselage.external_height / 2 +
                        wing.spar.length / 2
                    )
                )
            )
            self._set("fin_wing", fin_wing)
            self.__size_fuselage()

            # end_condition
            actual_tail_arm = abs(
                self.fin_wing.airfoil3D.aerodynamic_center.x - self.mass.cg.x)
            error = abs(target_fin_arm - actual_tail_arm)

            if self.verbose:
                print(f"[fin] iter: {i}, fin arm error: {error*1e3:.2f}mm")

            if error < self.abs_tol:
                return
        else:
            raise RuntimeError("Aircraft Fin Arm location did not converge")

    def add_fuselage(self, fuselage: FuselageSizing) -> None:
        self._set("_fuselage", fuselage)

    def __size_fuselage(self) -> None:
        """Size fuselage based on main, tail and fin wing geometry and location"""

        if not hasattr(self, "_fuselage"):
            raise RuntimeError("Fuselage must be created first")

        self._set("fuselage", self._fuselage.size_fuselage())

    def add_payload(self, payload: PayloadSizing) -> None:
        self._set("_payload", payload)
        self.__size_payload()

    def __size_payload(self) -> None:
        """Position payload based on static margin requirement"""
        if not hasattr(self, "_payload"):
            raise RuntimeError("Payload must be created first")

        payload = self._payload
        m_payload = payload.payload.mass.mass
        target_SM = payload.SM

        # place at origin (initial position)
        self._set("payload", payload.payload)

        for i in range(self.n_max):

            m = self.mass.mass
            c = self.port_main_wing.airfoil3D.chord

            x_current = self.payload.center.x

            error = target_SM - self.SM

            if abs(error) < self.static_margin_abs_tol:
                return

            # correction
            dx = error * (m / m_payload) * c
            x_new = x_current + dx

            self._set("payload", payload.payload.copy(
                name=payload.payload.name,
                offset=Position(x=x_new)
            ))

            if hasattr(self, "_prop"):
                self.__size_propulsion_system()
            if hasattr(self, "_fin"):
                self.__size_fin()
            if hasattr(self, "_tail"):
                self.__size_tail()
            if hasattr(self, "_fuselage"):
                self.__size_fuselage()
            if hasattr(self, "_gear"):
                self.__size_gear()

            if self.verbose:
                print(
                    f"[payload] iter: {i}, static margin error: {error:.3f}")

        else:
            raise RuntimeError("Aircraft Payload location did not converge")

    def resize(self) -> None:
        """resize all dynamically determined components to meet the defined parameters, e.g. tail arm"""

        last_cg = 0  # cg [m] in the x-axis
        for i in range(self.n_max):

            if hasattr(self, "_prop"):
                self.__size_propulsion_system()
            if hasattr(self, "_fin"):
                self.__size_fin()
            if hasattr(self, "_tail"):
                self.__size_tail()
            if hasattr(self, "_fuselage"):
                self.__size_fuselage()
            if hasattr(self, "_gear"):
                self.__size_gear()
            if hasattr(self, "_payload"):
                self.__size_payload()

            # end condition
            cg = self.mass.cg.x
            error = abs(last_cg - cg)

            if self.verbose:
                print(
                    f"[aircraft] iter: {i}, cg error: {error*1e3:.2f}mm"
                )

            if error < self.abs_tol:
                return
            last_cg = cg
        else:
            raise RuntimeError("Battery Sizing did not converge")

    #
    # Internal Methods
    #

    def _draw_planform(self, ax: Axes):
        for child in self.children:
            child._draw_planform(ax)

        ax.plot(self.np, 0, "*", markersize=10, markeredgewidth=2, label="NP")

    def _draw_side_elevation(self, ax: Axes):
        for child in self.children:
            child._draw_side_elevation(ax)

        ax.plot(self.np, 0, "*", markersize=10, markeredgewidth=2, label="NP")


class AircraftPlots(ComponentPlots[Aircraft]):

    @scientific_plot(auto_limits=None)
    def CL(self, ax: Axes):
        model = self.model

        main_airfoil = model.port_main_wing.airfoil3D.airfoil2d
        tail_airfoil = model.port_tail_wing.airfoil3D.airfoil2d
        fin_airfoil = model.fin_wing.airfoil3D.airfoil2d

        min_alpha = max(
            main_airfoil.min_alpha, tail_airfoil.min_alpha, fin_airfoil.min_alpha
        )
        max_alpha = min(
            main_airfoil.max_alpha, tail_airfoil.max_alpha, fin_airfoil.max_alpha
        )
        alphas = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(alphas, [model.CL(alpha) for alpha in alphas])

        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Lift Coefficient")

    @scientific_plot(auto_limits=None)
    def CD(self, ax: Axes):
        model = self.model

        main_airfoil = model.port_main_wing.airfoil3D.airfoil2d
        tail_airfoil = model.port_tail_wing.airfoil3D.airfoil2d
        fin_airfoil = model.fin_wing.airfoil3D.airfoil2d

        min_alpha = max(
            main_airfoil.min_alpha, tail_airfoil.min_alpha, fin_airfoil.min_alpha
        )
        max_alpha = min(
            main_airfoil.max_alpha, tail_airfoil.max_alpha, fin_airfoil.max_alpha
        )
        alphas = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(alphas, [model.CD(alpha) for alpha in alphas])

        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Drag Coefficient")

    @scientific_plot(auto_limits=None)
    def LD(self, ax: Axes):
        model = self.model

        main_airfoil = model.port_main_wing.airfoil3D.airfoil2d
        tail_airfoil = model.port_tail_wing.airfoil3D.airfoil2d
        fin_airfoil = model.fin_wing.airfoil3D.airfoil2d

        min_alpha = max(
            main_airfoil.min_alpha, tail_airfoil.min_alpha, fin_airfoil.min_alpha
        )
        max_alpha = min(
            main_airfoil.max_alpha, tail_airfoil.max_alpha, fin_airfoil.max_alpha
        )
        alphas = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(alphas, [model.CL(alpha) / model.CD(alpha)
                for alpha in alphas])

        ax.set_xlabel(r"Angle of attack [$^\circ$]")
        ax.set_ylabel(r"Lift-to-Drag Ratio")

    @scientific_plot(auto_limits=None)
    def drag_polar(self, ax: Axes):
        model = self.model

        main_airfoil = model.port_main_wing.airfoil3D.airfoil2d
        tail_airfoil = model.port_tail_wing.airfoil3D.airfoil2d
        fin_airfoil = model.fin_wing.airfoil3D.airfoil2d

        min_alpha = max(
            main_airfoil.min_alpha, tail_airfoil.min_alpha, fin_airfoil.min_alpha
        )
        max_alpha = min(
            main_airfoil.max_alpha, tail_airfoil.max_alpha, fin_airfoil.max_alpha
        )
        alphas = np.linspace(min_alpha, max_alpha, 50)

        ax.plot(
            [model.CL(alpha) for alpha in alphas],
            [model.CD(alpha)for alpha in alphas]
        )

        ax.set_xlabel(r"Lift Coefficient")
        ax.set_ylabel(r"Drag Coefficient")


class AircraftInfo(ComponentInfo[Aircraft]):

    @scientific_table()
    def overview(self):
        model = self.model

        return pd.DataFrame({
            "Mass": model.mass.mass,
            "CL_max": model.CL_max,
            "CD_0": model.CD_0(0),
            "Lift-to-Drag Ratio": model.CL(0) / model.CD(0),
            "Stall Airspeed [m/s]": model.estimate_stall_speed,
            "Optimal Range Airspeed [m/s]": model.optimal_range_speed,
            "Optimal Endurance Airspeed [m/s]": model.optimal_endurance_speed,
            "Static Margin": model.SM
        }, index=[0])
