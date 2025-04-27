#!/usr/bin/env python
"""
Stocker App - Main Entry Point

This module provides a simple text-based UI for the Stocker application,
allowing users to access various features like loading stock symbols and price data.
"""

import os
import sys
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Callable, Any

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data.SymbolData import SymbolDataManager
from scripts.CandleLoader import load_candles, validate_date, validate_time


def clear_screen():
    """Clear the console screen based on the operating system."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title="STOCKER APP"):
    """Print the application header."""
    clear_screen()
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    print("A stock market analysis and trading application")
    print("=" * 60)
    print()


def get_valid_date(prompt, default=None):
    """Get a valid date from user input."""
    while True:
        default_display = f" [{default}]" if default else ""
        user_input = input(f"{prompt}{default_display}: ").strip()
        
        if not user_input and default:
            return default
        
        if validate_date(user_input):
            return user_input
        
        print("Invalid date format. Please use YYYY-MM-DD format.")


def get_valid_time(prompt, default=None):
    """Get a valid time from user input."""
    while True:
        default_display = f" [{default}]" if default else ""
        user_input = input(f"{prompt}{default_display}: ").strip()
        
        if not user_input and default:
            return default
        
        if validate_time(user_input):
            return user_input
        
        print("Invalid time format. Please use HH:MM format (24h).")


def load_symbols_menu():
    """Menu for loading stock symbols into the database."""
    while True:
        print_header("SYMBOL LOADER")
        print("\nEnter stock symbols to load (comma or space separated)")
        print("Type 'back' to return to the main menu")
        
        user_input = input("\nEnter symbols: ").strip()
        
        if user_input.lower() == 'back':
            return
        
        # Split by comma or space and remove any empty strings
        symbols = [s.strip().upper() for s in user_input.replace(',', ' ').split()]
        symbols = [s for s in symbols if s]
        
        if not symbols:
            print("No valid symbols entered.")
            time.sleep(2)
            continue
        
        print(f"\nLoading {len(symbols)} symbols: {', '.join(symbols)}")
        print("Please wait...")
        
        # Use the SymbolDataManager to load the symbols
        symbol_manager = SymbolDataManager()
        result = symbol_manager.load_multiple_symbols(symbols)
        
        # Display results
        print("\nResults:")
        print(f"- Successfully loaded: {result['success_count']} out of {result['total']}")
        
        if result['failed_symbols']:
            print(f"- Failed symbols: {', '.join(result['failed_symbols'])}")
        
        input("\nPress Enter to continue...")


def view_symbols_menu():
    """Menu for viewing loaded stock symbols from the database using SQLAlchemy."""
    print_header("VIEW SYMBOLS")
    
    # Use the SymbolDataManager to retrieve symbols
    symbol_manager = SymbolDataManager()
    symbols = symbol_manager.get_all_symbols()
    
    if symbols:
        print(f"\nFound {len(symbols)} symbols in the database:")
        print("\n{:<10} {:<40} {:<15} {:<20}".format("SYMBOL", "NAME", "SECTOR", "UPDATED AT"))
        print("-" * 85)
        
        for symbol in symbols:
            print("{:<10} {:<40} {:<15} {:<20}".format(
                symbol.symbol,
                symbol.name[:38] + '..' if len(symbol.name) > 40 else symbol.name,
                symbol.sector or "Unknown",
                symbol.updated_at.strftime('%Y-%m-%d %H:%M:%S')
            ))
    else:
        print("\nNo symbols found in the database.")
    
    input("\nPress Enter to continue...")


def load_candles_menu():
    """Menu for loading candle/price data for stocks."""
    
    # Define timeframe options
    timeframe_options = [
        {'id': '1', 'name': 'Daily', 'key': 'daily'},
        {'id': '2', 'name': 'Hourly', 'key': 'hourly'},
        {'id': '3', 'name': '30 Minutes', 'key': '30min'},
        {'id': '4', 'name': '5 Minutes', 'key': '5min'}
    ]
    
    while True:
        print_header("CANDLE DATA LOADER")
        
        # First, select a symbol
        print("\nEnter a stock symbol to load candle data for")
        print("Type 'back' to return to the main menu")
        
        ticker = input("\nEnter symbol: ").strip().upper()
        
        if ticker.lower() == 'back':
            return
        
        # Select timeframe
        print("\nSelect timeframe:")
        for option in timeframe_options:
            print(f"{option['id']}. {option['name']}")
            
        timeframe_choice = input("\nEnter choice (1-4): ").strip()
        if not timeframe_choice or timeframe_choice not in [o['id'] for o in timeframe_options]:
            print("\nInvalid timeframe selection.")
            time.sleep(2)
            continue
        
        selected_timeframe = next(o['key'] for o in timeframe_options if o['id'] == timeframe_choice)
        
        # Get date range
        today = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        print("\nEnter date range:")
        start_date = get_valid_date("Start date (YYYY-MM-DD)", one_year_ago)
        end_date = get_valid_date("End date (YYYY-MM-DD)", today)
        
        # Ask for time range if not daily candles
        start_time = None
        end_time = None
        if selected_timeframe != 'daily':
            print("\nWould you like to filter by specific time of day? (y/n)")
            if input().strip().lower() == 'y':
                start_time = get_valid_time("Start time (HH:MM)", "09:30")
                end_time = get_valid_time("End time (HH:MM)", "16:00")
        
        # Fetch and save the candle data
        print(f"\nFetching {selected_timeframe} candles for {ticker} from {start_date} to {end_date}")
        if start_time and end_time:
            print(f"Time filtering: {start_time} to {end_time}")
        
        print("Please wait...")
        
        # Use the abstracted load_candles function from the CandleLoader script
        result = load_candles(
            ticker=ticker,
            timeframe=selected_timeframe,
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,
            end_time=end_time,
            verbose=True
        )
        
        # Display results
        if result['success']:
            print(f"\nSuccess: {result['message']}")
        else:
            print(f"\nError: {result['message']}")
        
        input("\nPress Enter to continue...")


def main_menu():
    """Display and handle the main application menu."""
    # Define menu options
    menu_options = [
        {'id': '1', 'name': 'Load Stock Symbols', 'function': load_symbols_menu},
        {'id': '2', 'name': 'View Loaded Symbols', 'function': view_symbols_menu},
        {'id': '3', 'name': 'Load Candle Data', 'function': load_candles_menu},
        {'id': '0', 'name': 'Exit', 'function': lambda: sys.exit(0)}
    ]
    
    while True:
        print_header()
        print("MAIN MENU")
        print("\nPlease select an option:")
        
        # Print menu options from the list
        for option in menu_options:
            print(f"{option['id']}. {option['name']}")
        
        # Get valid range for user input
        valid_choices = [o['id'] for o in menu_options]
        max_choice = max(int(c) for c in valid_choices if c.isdigit())
        
        choice = input(f"\nEnter your choice (0-{max_choice}): ").strip()
        
        # Find the selected option
        selected_option = next((o for o in menu_options if o['id'] == choice), None)
        
        if selected_option:
            if selected_option['id'] == '0':
                print("\nExiting Stocker App. Goodbye!")
                sys.exit(0)
            else:
                # Execute the function associated with this option
                selected_option['function']()
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye!")
        sys.exit(0)