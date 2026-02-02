"""
Greeks Engine
=============

Black-Scholes model for calculating option Greeks.
Provides Delta, Gamma, Theta, Vega, and Probability of Profit.

Now with optional GPU acceleration via gpu_engine.
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


@dataclass
class Greeks:
    """Container for option Greeks."""
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float   # Per 1% IV change
    rho: float


@dataclass
class SpreadGreeks:
    """Container for spread Greeks (net of short and long legs)."""
    net_delta: float
    net_gamma: float
    net_theta: float  # Per day (positive = time decay helps us)
    net_vega: float   # Per 1% IV change (negative = IV drop helps us)


class GreeksEngine:
    """
    Black-Scholes Greeks calculator.
    
    Formulas use standard Black-Scholes model with:
    - S: Spot price
    - K: Strike price
    - T: Time to expiration (years)
    - r: Risk-free rate
    - sigma: Implied volatility
    """
    
    # Default risk-free rate (current ~4.5% as of 2026)
    DEFAULT_RISK_FREE_RATE = 0.045
    
    def __init__(self, risk_free_rate: float = None):
        """
        Initialize Greeks engine.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4.5%)
        """
        self.r = risk_free_rate or self.DEFAULT_RISK_FREE_RATE
        logger.debug(f"Greeks Engine initialized with r={self.r}")
    
    def _d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return self._d1(S, K, T, sigma) - sigma * math.sqrt(T)
    
    def calculate_delta(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> float:
        """
        Calculate option delta.
        
        Delta measures the rate of change of option price with respect
        to changes in the underlying price.
        
        For puts: Delta is negative (price goes up, put value goes down).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (decimal)
            option_type: 'call' or 'put'
            
        Returns:
            Delta value (-1 to 0 for puts, 0 to 1 for calls)
        """
        if T <= 0:
            # At expiration
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = self._d1(S, K, T, sigma)
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1  # Put delta is N(d1) - 1
    
    def calculate_gamma(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float
    ) -> float:
        """
        Calculate option gamma.
        
        Gamma measures the rate of change of delta. Same for calls and puts.
        
        Returns:
            Gamma value (always positive)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    def calculate_theta(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> float:
        """
        Calculate option theta (per day).
        
        Theta measures time decay. Negative means option loses value over time.
        
        Returns:
            Theta per day (negative for long options)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        
        # First term (same for calls and puts)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        
        if option_type == 'call':
            theta_annual = term1 - self.r * K * math.exp(-self.r * T) * norm.cdf(d2)
        else:
            theta_annual = term1 + self.r * K * math.exp(-self.r * T) * norm.cdf(-d2)
        
        # Convert to per-day
        return theta_annual / 365
    
    def calculate_vega(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float
    ) -> float:
        """
        Calculate option vega (per 1% IV change).
        
        Vega measures sensitivity to volatility changes.
        Same for calls and puts.
        
        Returns:
            Vega per 1% IV change
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, sigma)
        vega_full = S * norm.pdf(d1) * math.sqrt(T)
        
        # Convert to per 1% change
        return vega_full / 100
    
    def calculate_rho(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> float:
        """
        Calculate option rho (sensitivity to interest rates).
        
        Returns:
            Rho per 1% rate change
        """
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, sigma)
        
        if option_type == 'call':
            return K * T * math.exp(-self.r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * math.exp(-self.r * T) * norm.cdf(-d2) / 100
    
    def calculate_all_greeks(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> Greeks:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (decimal, e.g., 0.40 for 40%)
            option_type: 'call' or 'put'
            
        Returns:
            Greeks dataclass with all values
        """
        return Greeks(
            delta=self.calculate_delta(S, K, T, sigma, option_type),
            gamma=self.calculate_gamma(S, K, T, sigma),
            theta=self.calculate_theta(S, K, T, sigma, option_type),
            vega=self.calculate_vega(S, K, T, sigma),
            rho=self.calculate_rho(S, K, T, sigma, option_type),
        )
    
    def calculate_spread_greeks(
        self,
        S: float,
        short_strike: float,
        long_strike: float,
        T: float,
        short_iv: float,
        long_iv: float = None,
    ) -> SpreadGreeks:
        """
        Calculate net Greeks for a put credit spread.
        
        For a put credit spread:
        - We SELL the higher strike put (short)
        - We BUY the lower strike put (long)
        
        Net Greeks = Short Greeks (inverted sign) + Long Greeks
        
        Args:
            S: Spot price
            short_strike: Strike we sell (higher)
            long_strike: Strike we buy (lower)
            T: Time to expiration in years
            short_iv: IV of short put
            long_iv: IV of long put (defaults to short_iv)
            
        Returns:
            SpreadGreeks with net values
        """
        if long_iv is None:
            long_iv = short_iv
        
        # Short put (we sold it, so invert the sign)
        short_greeks = self.calculate_all_greeks(S, short_strike, T, short_iv, 'put')
        
        # Long put (we own it)
        long_greeks = self.calculate_all_greeks(S, long_strike, T, long_iv, 'put')
        
        # Net = -Short + Long (because we're short the first put)
        return SpreadGreeks(
            net_delta=-short_greeks.delta + long_greeks.delta,
            net_gamma=-short_greeks.gamma + long_greeks.gamma,
            net_theta=-short_greeks.theta + long_greeks.theta,  # Positive = good for us
            net_vega=-short_greeks.vega + long_greeks.vega,     # Negative = IV drop helps
        )
    
    def probability_of_profit(
        self,
        S: float,
        short_strike: float,
        credit: float,
        T: float,
        sigma: float,
        long_strike: Optional[float] = None,
        use_gpu: bool = True,
        use_heston: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate probability of profit and touch.
        
        Args:
            S: Current spot price
            short_strike: Strike we sold
            credit: Net credit received
            T: Time to expiration in years
            sigma: Implied volatility
            long_strike: Strike we bought
            use_gpu: If True, use GPU
            use_heston: If True, use stochastic volatility
            
        Returns:
            Dict with 'pop' (0-100) and 'touch_stop' (0-100)
        """
        break_even = short_strike - credit
        
        if T <= 0:
            return {'pop': 100.0 if S > break_even else 0.0, 'touch_stop': 0.0}
        
        if use_gpu:
            try:
                from .gpu_engine import get_gpu_engine
                engine = get_gpu_engine()
                
                ls = long_strike if long_strike else short_strike - (S * 0.1) # Default 10% wide
                
                if use_heston:
                    return engine.monte_carlo_pop_heston(
                        S=S, short_strike=short_strike, long_strike=ls,
                        credit=credit, T=T, v0=sigma**2
                    )
                else:
                    pop = engine.monte_carlo_pop(
                        S=S, short_strike=short_strike, long_strike=ls,
                        credit=credit, T=T, sigma=sigma
                    )
                    return {'pop': round(pop, 1), 'touch_stop': 0.0}
            except Exception as e:
                logger.debug(f"GPU unavailable, using analytical: {e}")
        
        # Fallback: Analytical
        d2 = self._d2(S, break_even, T, sigma)
        return {'pop': round(norm.cdf(d2) * 100, 1), 'touch_stop': 0.0}
    
    def expected_move(
        self,
        S: float,
        sigma: float,
        T: float,
    ) -> Tuple[float, float]:
        """
        Calculate expected move based on implied volatility.
        
        The "expected move" is the 1 standard deviation range.
        ~68% of outcomes are expected within this range.
        
        Args:
            S: Current spot price
            sigma: Implied volatility (annualized)
            T: Time to expiration in years
            
        Returns:
            Tuple of (low_price, high_price)
        """
        if T <= 0 or sigma <= 0:
            return (S, S)
        
        # Expected move = S * sigma * sqrt(T)
        move = S * sigma * math.sqrt(T)
        
        return (round(S - move, 2), round(S + move, 2))
    
    @staticmethod
    def dte_to_years(dte: int) -> float:
        """Convert days to expiration to years."""
        return dte / 365


# Test function
def test_greeks():
    """Test the Greeks engine."""
    engine = GreeksEngine()
    
    print("=" * 60)
    print("GREEKS ENGINE TEST")
    print("=" * 60)
    
    # Example: IBIT at $42, selling $38 put, 10 DTE, 50% IV
    S = 42.0
    K = 38.0
    T = 10 / 365
    sigma = 0.50
    
    print(f"\nExample: IBIT ${S}, ${K} Put, 10 DTE, {sigma*100:.0f}% IV")
    
    greeks = engine.calculate_all_greeks(S, K, T, sigma, 'put')
    print(f"\nSingle Put Greeks:")
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"  Gamma: {greeks.gamma:.4f}")
    print(f"  Theta: ${greeks.theta:.4f}/day")
    print(f"  Vega:  ${greeks.vega:.4f}/1% IV")
    
    # Spread: Sell $38, Buy $36
    short_strike = 38.0
    long_strike = 36.0
    spread_greeks = engine.calculate_spread_greeks(S, short_strike, long_strike, T, sigma)
    
    print(f"\nPut Spread Greeks (Sell ${short_strike}, Buy ${long_strike}):")
    print(f"  Net Delta: {spread_greeks.net_delta:.4f}")
    print(f"  Net Gamma: {spread_greeks.net_gamma:.4f}")
    print(f"  Net Theta: ${spread_greeks.net_theta:.4f}/day (+ is good)")
    print(f"  Net Vega:  ${spread_greeks.net_vega:.4f}/1% IV (- is good)")
    
    # PoP
    credit = 0.50
    pop = engine.probability_of_profit(S, short_strike, credit, T, sigma)
    print(f"\nProbability of Profit: {pop:.1f}%")
    
    # Expected move
    low, high = engine.expected_move(S, sigma, T)
    print(f"Expected Move (1 SD): ${low:.2f} - ${high:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_greeks()
