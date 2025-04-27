#!/usr/bin/env python
"""
Symbol Loader Script

Command-line interface to load stock symbol information from Alpaca API
into the PostgreSQL database.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parents[2]))

from data.SymbolData import SymbolDataManager


def main():
    """
    Main function to handle command-line arguments and load symbols.
    Uses the SymbolDataManager for the core loading functionality.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Load stock symbols into database')
    parser.add_argument('tickers', nargs='+', help='Stock ticker symbols to load')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    verbose = args.verbose
    
    print(f"Loading symbol data for: {', '.join(tickers)}")
    
    # Use the abstracted data manager for the actual loading logic
    symbol_manager = SymbolDataManager()
    result = symbol_manager.load_multiple_symbols(tickers)
    
    # Display results to the user
    success_count = result['success_count']
    total = result['total']
    failed_symbols = result['failed_symbols']
    
    print(f"\nSummary: Successfully loaded {success_count} out of {total} symbols")
    
    if failed_symbols and (verbose or len(failed_symbols) <= 5):
        print(f"Failed symbols: {', '.join(failed_symbols)}")
    elif failed_symbols:
        print(f"{len(failed_symbols)} symbols failed. Use --verbose for details.")


if __name__ == "__main__":
    main()