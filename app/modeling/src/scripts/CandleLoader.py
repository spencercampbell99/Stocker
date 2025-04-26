#!/usr/bin/env python
"""
Candle Loader Script

Command-line interface to load candle/price data from Alpaca API
into the PostgreSQL database.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parents[2]))

from src.data.CandleData import CandleDataManager
from src.data.SymbolData import SymbolDataManager


def validate_date(date_str):
    """Validate date format YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_time(time_str):
    """Validate time format HH:MM."""
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False


def load_candles(ticker, timeframe, start_date, end_date, start_time=None, end_time=None, verbose=False):
    """
    Load candle data for a specific ticker and timeframe.
    
    Args:
        ticker (str): The stock ticker to get data for
        timeframe (str): One of 'daily', 'hourly', '30min', '5min'
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        start_time (str, optional): Start time in HH:MM format (24h)
        end_time (str, optional): End time in HH:MM format (24h)
        verbose (bool): Whether to display detailed output
        
    Returns:
        dict: Result of the operation
    """
    # Initialize managers
    candle_manager = CandleDataManager()
    symbol_manager = SymbolDataManager()
    
    # Verify the symbol exists in the database
    all_symbols = symbol_manager.get_all_symbols()
    symbol_exists = any(s.symbol == ticker for s in all_symbols)
    
    if not symbol_exists:
        return {
            'success': False,
            'message': f"Symbol {ticker} not found in database. Please load it first."
        }
    
    # Validate the timeframe
    valid_timeframes = ['daily', 'hourly', '30min', '5min']
    if timeframe not in valid_timeframes:
        return {
            'success': False,
            'message': f"Invalid timeframe: {timeframe}. Must be one of {', '.join(valid_timeframes)}"
        }
    
    # Validate dates
    if not validate_date(start_date) or not validate_date(end_date):
        return {
            'success': False,
            'message': "Invalid date format. Use YYYY-MM-DD."
        }
    
    # Validate times if provided
    if (start_time and not validate_time(start_time)) or (end_time and not validate_time(end_time)):
        return {
            'success': False,
            'message': "Invalid time format. Use HH:MM (24h)."
        }
    
    # Get data from Alpaca
    if verbose:
        print(f"Fetching {timeframe} candles for {ticker} from {start_date} to {end_date}")
        if start_time and end_time:
            print(f"Time filtering: {start_time} to {end_time}")
    
    candle_data = candle_manager.get_candle_data(
        ticker=ticker,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time
    )
    
    if candle_data.empty:
        return {
            'success': False,
            'message': f"No data found for {ticker} with the specified parameters."
        }
    
    # Save to database
    if verbose:
        print(f"Found {len(candle_data)} candles. Saving to database...")
    
    result = candle_manager.save_candle_data(
        ticker=ticker,
        timeframe=timeframe,
        data=candle_data
    )
    
    return result


def main():
    """
    Main function to handle command-line arguments and load candle data.
    Uses the CandleDataManager for the core loading functionality.
    """
    # Shift end date back by 20 minutes to avoid SIP data restrictions
    now_minus_20_minutes = datetime.now() - timedelta(minutes=20)
    today = now_minus_20_minutes.strftime("%Y-%m-%d")
    today_time = now_minus_20_minutes.strftime("%H:%M")
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    parser = argparse.ArgumentParser(description='Load candle/price data into database')
    parser.add_argument('ticker', help='Stock ticker symbol to load data for')
    parser.add_argument('--timeframe', '-t', choices=['daily', 'hourly', '30min', '5min'], 
                        default='daily', help='Timeframe for the candle data')
    parser.add_argument('--start-date', '-s', default=one_year_ago, 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', '-e', default=today, 
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--start-time', help='Start time in HH:MM format (24h)')
    parser.add_argument('--end-time', default=today_time if not today.endswith("00:00") else None,
                        help='End time in HH:MM format (24h)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    timeframe = args.timeframe
    start_date = args.start_date
    end_date = args.end_date
    start_time = args.start_time
    end_time = args.end_time
    verbose = args.verbose
    
    # Load the data
    print(f"Loading {timeframe} candle data for {ticker} from {start_date} to {end_date}")
    if start_time:
        print(f"Start time: {start_time}")
    if end_time:
        print(f"End time: {end_time}")
    
    result = load_candles(
        ticker=ticker,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose
    )
    
    # Display results
    if result['success']:
        print(f"\nSuccess: {result['message']}")
    else:
        print(f"\nError: {result['message']}")


if __name__ == "__main__":
    main()