"""
OptionsBacktest.py - Options trading backtest system

This module runs backtests of options trading strategies using machine learning models
to predict market movements and simulates trading performance with Black-Scholes pricing.
"""

import sys
from pathlib import Path
import pandas as pd
import tensorflow as tf
import joblib
import pandas as pd
from sqlalchemy import text
from db.database import get_session
from models.DataHandler import get_up_down_percent_model_data, get_percent_move_model_data
from scripts.model_evals.TradingSimulation import (
    simulate_options_trading,
    plot_options_trading_results,
    display_options_trading_results,
    save_options_trading_results_to_table
)

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parents[3]))

# Constants
START_DATE = "2025-04-23"
TICKER = "SPY"
MODEL_VERSION = "TfStraightUpDownModel_v0.1"
# MODEL_VERSION = "TfUpDownModel_v0.1"

def get_market_data_straight_model(start_date=START_DATE, ticker=TICKER):
    """
    Fetch and merge all required market data for the straight model
    
    Args:
        start_date: Start date for the data
        ticker: Ticker symbol
        
    Returns:
        DataFrame with market data
    """
    print(f"Fetching market data from database for {ticker} starting {start_date}...")
    
    original_start_date = start_date
    
    # move start_date back 2 months
    start_date = pd.to_datetime(start_date) - pd.DateOffset(months=2)
    start_date = start_date.strftime('%Y-%m-%d')
    
    # Get core price data
    data = get_percent_move_model_data(
        start_date=start_date,
        ticker=ticker,
    )

    # Get open price for 4:15 every day
    db = get_session()
    query = text(f"""
        SELECT
            "timestamp"::date as date,
            close
        FROM
            stocks_fivemincandle
        WHERE
            ticker = '{ticker}'
            AND "timestamp" >= '{start_date}'
            AND "timestamp"::time = '16:10:00'
    """)
    
    four_fifteen_close_data = db.execute(query).fetchall()
    db.close()
    
    # convert to DataFrame and concat
    four_fifteen_close_data = pd.DataFrame(four_fifteen_close_data, columns=['date', 'four_fifteen_price'])
    four_fifteen_close_data['date'] = pd.to_datetime(four_fifteen_close_data['date'])
    four_fifteen_close_data.set_index('date', inplace=True)
    
    data = pd.concat([data, four_fifteen_close_data], axis=1)
    
    # make sure all data columns are float
    for col in data.columns:
        if col not in ['date', 'ticker']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    
    # filter to after original_start_date
    data = data[data.index >= pd.to_datetime(original_start_date).date()]
    
    return data

def get_market_data(start_date=START_DATE, ticker=TICKER, up_threshold=1.005, down_threshold=0.995):
    """
    Fetch and merge all required market data
    
    Args:
        start_date: Start date for the data
        ticker: Ticker symbol
        up_threshold: Threshold for up move classification
        down_threshold: Threshold for down move classification
        
    Returns:
        DataFrame with market data
    """
    print(f"Fetching market data from database for {ticker} starting {start_date}...")
    
    # Get core price data
    data = get_up_down_percent_model_data(
        start_date=start_date,
        ticker=ticker,
        up_threshold=up_threshold,
        down_threshold=down_threshold
    )
    
    # Get VIX and US10Y data
    db = get_session()
    query = text(f"""
        WITH AllData AS (
            SELECT 
                ticker,
                "timestamp",
                close,
                open
            FROM 
                stocks_dailycandle
            WHERE 
                ticker IN ('VIX', 'US10Y')
                AND "timestamp" >= '{start_date}'
        ),
        PivotedData AS (
            SELECT 
                "timestamp" AS date,
                MAX(CASE WHEN ticker = 'VIX' THEN close END) AS vix_close,
                MAX(CASE WHEN ticker = 'US10Y' THEN close END) AS us10y_close,
                MAX(CASE WHEN ticker = 'VIX' THEN open END) AS vix_open,
                MAX(CASE WHEN ticker = 'US10Y' THEN open END) AS us10y_open
            FROM 
                AllData
            GROUP BY 
                "timestamp"
        )
        SELECT
            *
        from
            PivotedData
        ORDER BY
            date ASC
    """)
    
    macro_data = db.execute(query).fetchall()
    
    # Convert to DataFrame
    macro_data = pd.DataFrame(macro_data, columns=['date', 'vix_close', 'us10y_close', 'vix_open', 'us10y_open'])
    
    # Shift close so date of 2025-04-20 has close of 2025-04-19
    # macro_data['vix_close'] = macro_data['vix_close'].shift(1)
    # macro_data['us10y_close'] = macro_data['us10y_close'].shift(1)
    
    # index by date
    macro_data['date'] = pd.to_datetime(macro_data['date'])
    macro_data.set_index('date', inplace=True)
    
    macro_data = macro_data[macro_data.index >= start_date]
    macro_data = macro_data.dropna()
    
    # merge with core data
    data = pd.merge(data, macro_data, left_index=True, right_index=True, how='left')
    
    # Get open price for 4:15 every day
    query = text(f"""
        SELECT
            "timestamp"::date as date,
            close
        FROM
            stocks_fivemincandle
        WHERE
            ticker = '{ticker}'
            AND "timestamp" >= '{start_date}'
            AND "timestamp"::time = '16:10:00'
    """)
    
    four_fifteen_close_data = db.execute(query).fetchall()
    db.close()
    
    # convert to DataFrame and concat
    four_fifteen_close_data = pd.DataFrame(four_fifteen_close_data, columns=['date', 'four_fifteen_price'])
    four_fifteen_close_data['date'] = pd.to_datetime(four_fifteen_close_data['date'])
    four_fifteen_close_data.set_index('date', inplace=True)
    
    data = pd.concat([data, four_fifteen_close_data], axis=1)
    
    # make sure all data columns are float
    for col in data.columns:
        if col not in ['date', 'ticker']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    
    return data

def load_model_and_metadata(model_version=MODEL_VERSION):
    """
    Load the saved TensorFlow model and its metadata
    
    Args:
        model_version: Version of model to load from saved_models directory
        
    Returns:
        tuple: (model, metadata_dict)
    """
    print(f"Loading {model_version} and metadata...")
    
    # Define paths to model and metadata files
    model_dir = Path(__file__).parents[3] / "saved_models"
    model_path = model_dir / f"{model_version}.h5"
    fallback_model_path = model_dir / f"{model_version}.tflite"
    metadata_path = model_dir / f"{model_version}_metadata.pkl"
    
    # Check if files exist
    if model_path.exists():
        model = tf.keras.models.load_model(str(model_path))   
    elif fallback_model_path.exists():
        model = tf.lite.Interpreter(model_path=str(fallback_model_path))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load the metadata
    metadata = joblib.load(str(metadata_path))
    
    return model, metadata

MODEL_FUNCTION_MAPPINGS = {
    "TfStraightUpDownModel": {
        "v0.1": {
            "model": load_model_and_metadata,
            "get_market_data": get_market_data_straight_model,
        },
    },
    "TfUpDownModel": {
        "v0.1": {
            "model": load_model_and_metadata,
            "get_market_data": get_market_data,
        },
        "v0.2": {
            "model": load_model_and_metadata,
            "get_market_data": get_market_data,
        }
    },
}

def main():
    """Main function"""
    try:
        # Load model and metadata
        model, metadata = load_model_and_metadata(model_version=MODEL_VERSION)
        
        # Extract key information from metadata
        features = metadata['features']
        ticker = metadata.get('metadata', {}).get('ticker', TICKER)
        model_version = metadata.get('metadata', {}).get('model_version', MODEL_VERSION)
        up_threshold = metadata.get('thresholds', {}).get('up_threshold', 1.005)
        down_threshold = metadata.get('thresholds', {}).get('down_threshold', 0.995)
        
        print(f"\nModel information:")
        print(f"- Version: {model_version}")
        print(f"- Ticker: {ticker}")
        # print(f"- Threshold settings: Up > {up_threshold:.4f}, Down < {down_threshold:.4f}")
        print(f"- Features: {', '.join(features)}")
        
        market_data_func = MODEL_FUNCTION_MAPPINGS.get(
            metadata['metadata']['model_type'], {}).get(model_version, {}).get("get_market_data")
        if not market_data_func:
            raise ValueError(f"Model version {model_version} not supported for {metadata['metadata']['model_type']}")
        
        # if metadata['metadata']['trained_through_date'] > START_DATE, update start date
        start_date = START_DATE
        if 'trained_through_date' in metadata.get('metadata', {}) and metadata['metadata']['trained_through_date'] > START_DATE:
            start_date = metadata['metadata']['trained_through_date']
        
        # Get market data from the database
        daily_data = market_data_func(
            start_date=start_date
        )
        print(f"Loaded {len(daily_data)} days of market data")
        
        # Simulate options trading
        trading_results = simulate_options_trading(model, metadata, daily_data)
    
        # Display results
        display_options_trading_results(trading_results)
        
        # Plot trading results
        plot_path = Path(__file__).parent / "options_trading_simulation.png"
        plot_options_trading_results(trading_results, save_path=str(plot_path))
        print(f"\nOptions trading simulation plot saved to: {plot_path}")
        
        # If user wants to save to table, have them enter table name
        save_to_table = input("Do you want to save the trading results to a table? (y/n): ").strip().lower()
        if save_to_table == 'y':
            table_name = input("Enter table name (default: options_trading_results): ").strip() or "options_trading_results"
            save_options_trading_results_to_table(trading_results, table_name=table_name)
            print(f"Options trading results saved to table: {table_name}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()