"""
trading_simulation.py - Trading simulation functions for options backtesting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from pathlib import Path

# Local imports
from OptionPricingCalculator import get_option_prices, select_best_option, simulate_option_performance

# Constants
EASTERN_TZ = pytz.timezone('US/Eastern')
TRADE_START_TIME = datetime.strptime("09:30", "%H:%M").time()
TRADE_END_TIME = datetime.strptime("16:00", "%H:%M").time()

def simulate_options_trading(model, metadata, daily_data):
    """
    Core trading simulation with Black-Scholes pricing
    
    Args:
        model: TensorFlow model
        metadata: Model metadata dictionary
        daily_data: DataFrame with daily market data
        
    Returns:
        Dictionary with trading results
    """
    # Extract model parameters
    features = metadata['features']
    scaler = metadata['scaler']
    
    # Prepare data and get predictions
    X = daily_data[features]
    X_scaled = scaler.transform(X)
    y_pred_proba = model.predict(X_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'date': daily_data.index,
        'actual': daily_data['move_status'],
        'prediction': y_pred,
        'open': daily_data['open'],
        'close': daily_data['close'],  # Make sure we have close price from daily data
        'vix_close': daily_data['vix_close'],
        'us10y_close': daily_data['us10y_close'],
        'vix_open': daily_data['vix_open'],
        'us10y_open': daily_data['us10y_open'],
        'thirty_to_close_price': daily_data['thirty_to_close_price'],
    })
    
    # Initialize trading variables
    trades = []
    initial_balance = 1000.0
    balance = initial_balance
    balances = [balance]
    trade_dates = []
    
    # Simulate trading each day
    for _, row in predictions_df.iterrows():
        date = row['date']
        prediction = row['prediction']
        actual = row['actual']
        
        # Skip neutral predictions
        if prediction == 1:
            continue
            
        # Get market prices
        market_open = row['open']
        market_close = row['close']
        vix_close = row['vix_close']
        us10y_close = row['us10y_close']
        vix_open = row['vix_open']
        us10y_open = row['us10y_open']
        thirty_to_close_price = row['thirty_to_close_price']
        
        # Create entry time (market open)
        entry_time = datetime.combine(date, TRADE_START_TIME)
        entry_time = EASTERN_TZ.localize(entry_time) if entry_time.tzinfo is None else entry_time
        
        # Create exit time (market close or user-defined exit time)
        exit_time = datetime.combine(date, TRADE_END_TIME)
        exit_time = EASTERN_TZ.localize(exit_time) if exit_time.tzinfo is None else exit_time
        
        # Determine option type
        option_type = "call" if prediction == 2 else "put"
        
        # Get option chain
        option_chain = get_option_prices(
            spy_price=market_open,
            vix_prev_close=vix_close,
            us10y_prev_close=us10y_close,
            direction=option_type
        )

        # Select best option
        strike, premium, reason = select_best_option(
            option_chain,
            market_open,
            option_type,
            max_affordability=balance / 100 # Balance divided by 100 to account for 100x for premium costs
        )
        
        if strike is None:
            continue  # Skip if no valid option found
        
        # (1 / 252) / 13 is 30 minutes to close for time to expire (1/13th of a trading day is 30 minutes)
        if actual == 0:
            # Simulate option performance
            performance_pct = simulate_option_performance(
                exit_ticker_price=thirty_to_close_price,
                entry_premium=premium,
                strike=strike,
                us10y=us10y_open,
                vix=vix_open,
                direction=option_type,
                expir_time_override = (1 / 252) / 13
            )
        elif actual == 2:
            # Simulate option performance
            performance_pct = simulate_option_performance(
                exit_ticker_price=thirty_to_close_price,
                entry_premium=premium,
                strike=strike,
                us10y=us10y_open,
                vix=vix_open,
                direction=option_type,
                expir_time_override = (1 / 252) / 13
            )
        else:
            performance_pct = simulate_option_performance(
                exit_ticker_price=thirty_to_close_price,
                entry_premium=premium,
                strike=strike,
                us10y=us10y_open,
                vix=vix_open,
                direction=option_type,
                expir_time_override = (1 / 252) / 13
            )
        
        def get_position_size(balance, premium):
            """
            Calculate position size based on balance and premium selected.
            
            Take greater value between 25% of balance and premium.
            If 25% is greater, reduce down to premium * 100 where less than 25% of balance.
            
            Args:
                balance: Current account balance
                premium: Option premium selected
            """
            max_safe_bet = balance * 0.25
            premium_cost = premium * 100  # Cost of one contract (100 shares)
            
            if premium_cost > max_safe_bet:
                return premium_cost
            else:
                return np.floor(max_safe_bet / premium_cost) * premium_cost
        
        # Calculate position and update balance
        position_size = get_position_size(balance, premium)
        dollar_return = position_size * performance_pct
        new_balance = balance + dollar_return
        
        # Record trade
        trades.append({
            'date': date,
            'entry_time': entry_time,
            'exit_time': "15:30:00",
            'prediction': prediction,
            'actual': actual,
            'option_type': option_type,
            'strike': strike,
            'vix_close': vix_close,
            'us10y_close': us10y_close,
            'premium': premium,
            'entry_price': market_open,
            'rough_exit_price': thirty_to_close_price,
            'close_price': market_close,
            'position_size': position_size,
            'performance_pct': performance_pct * 100,
            'pnl_dollar': dollar_return,
            'balance': new_balance,
            'result': "WIN" if dollar_return > 0 else "LOSS",
            'reasoning': reason,
        })
        
        # Update balance history
        balance = new_balance
        balances.append(balance)
        trade_dates.append(date)
    
    # Calculate final statistics
    win_count = sum(1 for trade in trades if trade['result'] == 'WIN')
    total_trades = len(trades)
    total_return = (balance / initial_balance - 1) * 100
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    return {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': total_return,
        'trades': trades,
        'balances': balances,
        'trade_dates': trade_dates,
        'win_count': win_count,
        'loss_count': total_trades - win_count,
        'total_trades': total_trades,
        'win_rate': win_rate
    }

def plot_options_trading_results(trading_results, save_path=None):
    """
    Plot the options trading simulation results
    
    Args:
        trading_results: Dictionary containing trading simulation results
        save_path: Path to save the plot (optional)
    """
    balances = trading_results['balances']
    trades = trading_results['trades']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot balance over time
    ax1.plot(range(len(balances)), balances, 'b-', linewidth=2)
    ax1.set_title('Options Trading Performance Simulation', fontsize=16)
    ax1.set_ylabel('Account Balance ($)', fontsize=12)
    ax1.set_xlim(0, len(balances)-1)
    ax1.grid(True)
    
    # Annotate start and end balance
    ax1.annotate(f"Start: ${trading_results['initial_balance']:.2f}", 
                xy=(0, trading_results['initial_balance']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='green')
    
    ax1.annotate(f"End: ${trading_results['final_balance']:.2f}", 
                xy=(len(balances)-1, trading_results['final_balance']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='blue')
    
    # Add horizontal line at initial balance
    ax1.axhline(y=trading_results['initial_balance'], color='r', linestyle='--', alpha=0.5)
    
    # Plot individual trade results
    pnl_values = [trade['pnl_dollar'] for trade in trades]
    colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
    
    ax2.bar(range(len(pnl_values)), pnl_values, color=colors, width=0.8)
    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.set_ylabel('Trade P&L ($)', fontsize=12)
    ax2.set_xlim(-0.5, len(trades)-0.5)
    ax2.grid(True, axis='y')
    
    # Add overall stats to the plot
    stats_text = (
        f"Total Return: {trading_results['total_return']:.2f}%\n"
        f"Win Rate: {trading_results['win_rate']:.2%}\n"
        f"Total Trades: {trading_results['total_trades']}\n"
        f"Wins: {trading_results['win_count']}\n"
        f"Losses: {trading_results['loss_count']}"
    )
    
    # Position the text box in figure coords
    plt.figtext(0.15, 0.01, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def display_options_trading_results(trading_results):
    """
    Display the options trading simulation results
    
    Args:
        trading_results: Dictionary containing trading simulation results
    """
    print("\n" + "=" * 60)
    print(f"OPTIONS TRADING SIMULATION RESULTS")
    print("=" * 60)
    
    print(f"\nInitial Balance: ${trading_results['initial_balance']:.2f}")
    print(f"Final Balance: ${trading_results['final_balance']:.2f}")
    print(f"Total Return: {trading_results['total_return']:.2f}%")
    print(f"Total Trades: {trading_results['total_trades']}")
    print(f"Win Rate: {trading_results['win_rate']:.2%}")
    print(f"Wins: {trading_results['win_count']}")
    print(f"Losses: {trading_results['loss_count']}")
    
    # Show sample of trades
    print("\nSample of Recent Trades:")
    print("Date       | Time  | Type | Strike | Premium | Entry  | Close  | P&L %    | P&L $    | Balance")
    print("-" * 100)
    
    for trade in trading_results['trades'][-10:]:  # Show the last 10 trades
        print(f"{trade['date']} | {trade['entry_time'].time()} | {trade['option_type']:<4} | {trade['strike']:<6} | ${trade['premium']:<7.2f} | {trade['entry_price']:<6.2f} | {trade['close_price']:<6.2f} | {trade['performance_pct']:8.2f}% | ${trade['pnl_dollar']:8.2f} | ${trade['balance']:8.2f}")
    
    print("\nDetailed trade reasoning:")
    for i, trade in enumerate(trading_results['trades'][-5:]):
        print(f"{i+1}. {trade['date']} - {trade['reasoning']}")

def save_options_trading_results_to_table(trading_results, table_name="options_trading_results"):
    """
    Save the options trading results to a postgres table
    
    Args:
        trading_results: Dictionary containing trading simulation results
        table_name: Name of the table to save results to
    """
    from db.database import get_session
    from sqlalchemy import text
    session = get_session()
    
    # Create table if it doesn't exist
    session.execute(text(f"""DROP TABLE IF EXISTS {table_name}"""))
    create_table_query = f"""
    CREATE TABLE {table_name} (
        date DATE,
        entry_time TIMESTAMP,
        exit_time TIME,
        prediction INT,
        actual INT,
        option_type VARCHAR(5),
        strike FLOAT,
        vix_close FLOAT,
        us10y_close FLOAT,
        premium FLOAT, 
        entry_price FLOAT,
        rough_exit_price FLOAT,
        close_price FLOAT,
        position_size FLOAT,
        performance_pct FLOAT,
        pnl_dollar FLOAT,
        balance FLOAT,
        result VARCHAR(10),
        reasoning TEXT
    )
    """
    session.execute(text(create_table_query))
    
    # Insert data into the table with pandas
    import pandas as pd
    
    trades_df = pd.DataFrame(trading_results['trades'])
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df.to_sql(table_name, session.connection(), if_exists='append', index=False)
    
    session.connection().commit()
    session.close()