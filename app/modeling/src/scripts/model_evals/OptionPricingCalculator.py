"""
option_pricing.py - Option pricing models and utilities for options backtesting
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Constants for option pricing
MIN_PREMIUM = 3.00  # Minimum option premium
STRIKE_STEP = 1     # SPY strike increments
SPREAD_ADJUSTMENT = 1.05  # 5% bid-ask spread

def calculate_black_scholes(S, K, T, sigma, r, option_type='call'):
    """
    Calculate Black-Scholes option price with robust error handling
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        sigma: Volatility (annualized)
        r: Risk-free rate
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0
            
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return max(price, 0.0)
    except:
        return 0.0

def get_option_prices(spy_price, vix_prev_close, us10y_prev_close, direction="call"):
    """
    Generate option chain with Black-Scholes prices
    
    Args:
        spy_price: Current SPY price
        vix_prev_close: Previous day's VIX close
        us10y_prev_close: Previous day's US10Y close
        direction: 'call' or 'put'
        
    Returns:
        Dictionary of {strike: premium}
    """
    
    # Convert to float if needed
    try:
        spy_price = float(spy_price)
        vix_prev_close = float(vix_prev_close)
        us10y_prev_close = float(us10y_prev_close)
    except (TypeError, ValueError) as e:
        print(f"ERROR: Failed to convert inputs to float: {e}")
        return {}
    
    if pd.isna(vix_prev_close) or vix_prev_close <= 0:
        return {}
    
    sigma = vix_prev_close / 100
    r = us10y_prev_close / 100
    T = 1 / 252  # 1 trading day
    
    # Generate realistic ITM strikes
    if direction == "call":
        base_strike = np.floor(spy_price / STRIKE_STEP) * STRIKE_STEP
        strikes = [round(base_strike - i*STRIKE_STEP, 2) for i in range(0, 20)]
    else:
        base_strike = np.ceil(spy_price / STRIKE_STEP) * STRIKE_STEP
        strikes = [round(base_strike + i*STRIKE_STEP, 2) for i in range(0, 20)]
    
    options = {}
    for strike in strikes:
        premium = calculate_black_scholes(
            S=spy_price,
            K=strike,
            T=T,
            sigma=sigma,
            r=r,
            option_type=direction
        )
        
        # Apply spread adjustment and minimum premium
        premium = round(premium * SPREAD_ADJUSTMENT, 2)
        options[strike] = premium
    
    return options

def select_best_option(options, spot_price, direction="call", max_affordability=None):
    """
    Select optimal ITM option based on criteria
    
    Args:
        options: Dictionary of {strike: premium}
        spot_price: Current spot price
        direction: 'call' or 'put'
        
    Returns:
        Tuple of (selected strike, premium, reason)
    """
    best_strike = None
    best_premium = None
    best_diff = float('inf')
    
    # If call, sort by strike descending
    # If put, sort by strike ascending
    if direction == "call":
        options = dict(sorted(options.items(), key=lambda x: x[0], reverse=True))
    else:
        options = dict(sorted(options.items(), key=lambda x: x[0]))
    
    for strike, premium in options.items():
        # Apply filters
        if premium < MIN_PREMIUM:
            continue
        
        if max_affordability is not None and premium > max_affordability:
            continue
        
        if direction == "call":
            strike_pct_diff = abs(strike + premium - spot_price) / premium
        else:
            strike_pct_diff = abs(strike - premium - spot_price) / premium
        if strike_pct_diff > 0.15:
            continue
            
        if direction == "call":
            if strike >= spot_price:  # Must be ITM
                continue
            diff = abs((strike + premium) - spot_price)
        else:
            if strike <= spot_price:  # Must be ITM
                continue
            diff = abs((strike - premium) - spot_price)
        
        best_diff = diff
        best_strike = strike
        best_premium = premium
        break
    
    if best_strike is None:
        return None, None, "No valid ITM option found meeting criteria"
    
    reason = f"Selected {direction.upper()} {best_strike} | Premium: ${best_premium:.2f} | "
    reason += f"Strike {'+' if direction == 'call' else '-'} Premium = "
    reason += f"{best_strike + best_premium if direction == 'call' else best_strike - best_premium:.2f}"
    
    return best_strike, best_premium, reason

def simulate_option_performance(exit_ticker_price, entry_premium, strike, us10y, vix, direction="call", expir_time_override=None):
    """
    Simulate option performance
    
    Args:
        exit_ticker_price: Exit price of the underlying asset
        entry_premium: Entry premium of the option
        strike: Strike price of the option
        us10y: US10Y yield
        vix: VIX index value
        direction: 'call' or 'put'
        
    Returns:
        Percentage return on option
    """

    if expir_time_override == 0:
        if direction == "call":
            close_price = max(exit_ticker_price - strike, 0)
        else:
            close_price = max(strike - exit_ticker_price, 0)
    else:
        # Calculate option price
        close_price = calculate_black_scholes(
            S=exit_ticker_price,
            K=strike,
            T= expir_time_override if expir_time_override else 1 / 252,
            sigma=vix / 100,
            r=us10y / 100,
            option_type=direction,
        )
    
    return close_price / entry_premium - 1.0


# prices = get_option_prices(
#     spy_price=546.65,
#     vix_prev_close=26.47,
#     us10y_prev_close=4.30,
#     direction="call"
# )

# for strike, premium in prices.items():
#     print(f"Strike: {strike}, Premium: {premium}")
    
# print("Best Option:")
# best_strike, best_premium, reason = select_best_option(prices, 550.64, direction="call", max_affordability=10.00)
# print(f"Strike: {best_strike}, Premium: {best_premium}, Reason: {reason}")