import decimal
from decimal import Decimal, getcontext
import math

# Set precision to maintain numerical stability during capacitor calculations
getcontext().prec = 28

class NumericallyStableCalculator:
    """Provides numerically stable calculations for physical quantities using Decimal."""
    
    EPSILON_0 = Decimal('8.8541878128e-12') # F/m

    @classmethod
    def calculate_capacitance(cls, permittivity_r: float, area_m2: float, separation_m: float) -> Decimal:
        """
        Calculate capacitance C = (ε_r * ε_0 * A) / d with stable decimal precision.
        Returns capacitance in Farads.
        """
        eps_r = Decimal(str(permittivity_r))
        a_m2 = Decimal(str(area_m2))
        d_m = Decimal(str(separation_m))
        
        if d_m <= 0:
            raise ValueError("Separation distance must be > 0")
            
        capacitance = (eps_r * cls.EPSILON_0 * a_m2) / d_m
        return capacitance
        
    @classmethod
    def threshold_compare(cls, val1: float, val2: float, tolerance: float = 1e-9) -> bool:
        """Safe comparison of floating point numbers with tolerance."""
        return abs(Decimal(str(val1)) - Decimal(str(val2))) <= Decimal(str(tolerance))
