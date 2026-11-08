from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    """material name"""
    ultimate_tensile_strength: float
    """ultimate tensile strength, UTC [Pa]"""
    ultimate_compressive_strength: float
    """ultimate compressive strength, UCS [Pa]"""
    youngs_modulus: float
    """elastic/youngs modulus, E [Pa]"""
    shear_modulus: float
    """shear modulus, G [Pa]"""
    density: float
    """density, rho [kg/m^3]"""

    nominal_thickness: float = 0.0
    """nominal thickness [m]"""

    @property
    def ultimate_strength(self) -> float:
        """minimum of ultimate tensile strength and ultimate compressive strength [Pa]"""
        return min(self.ultimate_tensile_strength, self.ultimate_compressive_strength)

    @property
    def area_density(self) -> float:
        """area density [kg/m^2]"""
        return self.density * self.nominal_thickness
