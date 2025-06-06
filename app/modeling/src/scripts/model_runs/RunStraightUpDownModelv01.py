#!/usr/bin/env python
"""
RunModelv0-1ForDay.py

Script to run the TfStraightUpDownModel_v0.1 on the current day's data.
This script:
1. Loads the saved v0.1 TensorFlow model
2. Gets the latest market data (SPY, VIX, US10Y)
3. Prepares features for prediction
4. Makes a prediction about market direction
5. Selects the best option trade based on the prediction
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parents[3]))

# Import required modules
from data.CandleData import CandleDataManager
from models.DataHandler import get_percent_move_model_data
from scripts.model_evals.OptionPricingCalculator import get_option_prices, select_best_option


def load_model_and_metadata(model_version="v0.1"):
    """
    Load the saved TensorFlow model and its metadata
    
    Args:
        model_version: Version tag of the model to load
        
    Returns:
        tuple: (model, metadata_dict)
    """
    print(f"Loading TfStraightUpDownModel_{model_version} and metadata...")
    
    # Define paths to model and metadata files
    model_dir = Path(__file__).parents[3] / "saved_models"
    model_path = model_dir / f"TfStraightUpDownModel_{model_version}.h5"
    metadata_path = model_dir / f"TfStraightUpDownModel_{model_version}_metadata.pkl"
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load the TensorFlow model
    model = tf.keras.models.load_model(str(model_path))
    
    # Load the metadata
    metadata = joblib.load(str(metadata_path))
    
    return model, metadata


def get_latest_market_data():
    """
    Get the latest market data for SPY, VIX, and US10Y
    
    Returns:
        dict: Dictionary containing the latest market data
    """
    print("Fetching latest market data...")
    
    # Initialize candle data manager
    candle_manager = CandleDataManager()
    
    # Calculate date ranges
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get SPY daily candles
    spy_daily = candle_manager.get_candle_data(
        ticker="SPY",
        timeframe="daily",
        start_date=yesterday,
        end_date=today
    )
    
    # Get SPY 5-minute candles for pre-market (4:00 AM to 9:29 AM)
    spy_5min = candle_manager.get_candle_data(
        ticker="SPY",
        timeframe="5min",
        start_date=yesterday,
        end_date=today,
    )
    
    # Use CandleDataManager to insert the spy data
    candle_manager.save_candle_data(
        ticker="SPY",
        data=spy_daily,
        timeframe="daily"
    )
    
    candle_manager.save_candle_data(
        ticker="SPY",
        data=spy_5min,
        timeframe="5min"
    )
    
    # Get VIX daily candles
    # Since VIX is manually loaded, we'll load it all and filter
    candle_manager.get_vix_candle_data()
    
    # Get US10Y daily candles
    # Since US10Y is manually loaded, we'll load it all and filter
    candle_manager.get_10_year_treasury_candle_data(last_week_only=True)
    
    return {
        'spy_daily': spy_daily,
        'spy_5min': spy_5min,
    }


def prepare_model_features(model_metadata, override_date=None, open_override=None):
    """
    Prepare features for the model based on the latest market data
    
    Args:
        model_metadata: Dictionary containing model metadata
        override_date: Optional date to override the default (today's date)
        
    Returns:
        DataFrame: DataFrame containing features for model prediction
    """
    print("Preparing features for model prediction...")

    target_date = datetime.strptime(override_date, '%Y-%m-%d') if override_date else datetime.now()
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    # Get the data from our DataHandler
    start_date = (target_date - timedelta(days=60)).strftime('%Y-%m-%d')
    
    # Use the data handler to get properly formatted data
    data = get_percent_move_model_data(
        start_date=start_date,
        ticker="SPY",
        open_override=open_override,
    )
    
    # Get the data at date - using string comparison of dates
    # Convert index to datetime first if it's not already, then filter by date string
    data.index = pd.to_datetime(data.index)
    data = data[data.index.strftime('%Y-%m-%d') == target_date_str]
    
    # for item in data.iloc[0].items():
    #     print(f"{item[0]}: {item[1]}")
    
    return data


def make_prediction(model, data, metadata):
    """
    Make a prediction using the model
    
    Args:
        model: TensorFlow model
        data: DataFrame containing features
        metadata: Dictionary containing model metadata
        
    Returns:
        tuple: (prediction_class, prediction_probabilities, features_used)
    """
    print("Making prediction...")
    
    # Extract features and scaler from metadata
    features = metadata['features']
    scaler = metadata['scaler']
    
    # Check if all required features are present
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # Extract features and scale them
    X = data[features].values
    X_scaled = scaler.transform(X)
    
    # Make prediction
    pred_proba = model.predict(X_scaled, verbose=0)
    pred_class = 2 if pred_proba[0][0] > 0.5 else 0
    
    # Class mapping: 0 = Down, 1 = Flat, 2 = Up
    class_names = {
        0: "Down",
        1: "Flat/Neutral",
        2: "Up"
    }
    
    print(f"Prediction: {class_names[pred_class]} (Class {pred_class})")
    print(f"Probabilities: UP: {pred_proba[0][0]:.4f}")
    
    return pred_class, pred_proba[0][0], features


def select_option_trade(prediction, spy_price, vix_value, us10y_value, max_affordability=None):
    """
    Select the best option trade based on the prediction
    
    Args:
        prediction: Predicted market direction (0 = Down, 1 = Flat, 2 = Up)
        spy_price: Current SPY price
        vix_value: Current VIX value
        us10y_value: Current US10Y value
        max_affordability: Maximum option premium affordability (optional)
        
    Returns:
        dict: Dictionary containing selected option information
    """
    if prediction == 1:  # Flat/Neutral
        print("Model predicts flat/neutral market. No option trade recommended.")
        return None
    
    # Determine option type
    option_type = "call" if prediction == 2 else "put"
    
    print(f"Selecting best {option_type.upper()} option based on prediction...")
    
    # Get option chain
    option_chain = get_option_prices(
        spy_price=spy_price,
        vix_prev_close=vix_value,
        us10y_prev_close=us10y_value,
        direction=option_type
    )
    
    # Select best option
    best_strike, best_premium, reason = select_best_option(
        option_chain,
        spy_price,
        direction=option_type,
        max_affordability=max_affordability
    )
    
    if best_strike is None:
        print("No suitable option found.")
        return None
    
    # Create result dictionary
    option_info = {
        'direction': option_type,
        'strike': best_strike,
        'premium': best_premium,
        'reason': reason,
        'spot_price': spy_price
    }
    
    return option_info


def get_latest_values(at_date=None):
    """
    Get the latest values for SPY price, VIX, and US10Y yield
    from the database for the specified date
    
    Args:
        at_date: Optional date to override the default (today's date)
    
    Returns:
        tuple: (spy_price, vix_value, us10y_value)
    """
    from db.database import get_session
    from sqlalchemy import text
    from datetime import datetime

    # Use provided date or default to today
    target_date = at_date if at_date else datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching values from database for date: {target_date}...")
    
    session = get_session()
    
    # Query for SPY price on target date (open or previous close)
    spy_query = text("""
        WITH date_prices AS (
            SELECT ticker, timestamp, open, close
            FROM stocks_dailycandle
            WHERE ticker = 'SPY' AND timestamp <= :target_date
            ORDER BY timestamp DESC
            LIMIT 2
        )
        SELECT 
            CASE 
                -- Open price from target date if available
                WHEN EXISTS (
                    SELECT 1 FROM date_prices 
                    WHERE DATE(timestamp) = DATE(:target_date) AND open IS NOT NULL
                ) THEN (
                    SELECT open FROM date_prices 
                    WHERE DATE(timestamp) = DATE(:target_date)
                )
                -- Otherwise previous day's close
                ELSE (
                    SELECT close FROM date_prices
                    LIMIT 1
                )
            END as price
    """)
    
    # Similar query for VIX value
    vix_query = text("""
        WITH date_prices AS (
            SELECT ticker, timestamp, open, close
            FROM stocks_dailycandle
            WHERE ticker = 'VIX' AND timestamp <= :target_date
            ORDER BY timestamp DESC
            LIMIT 2
        )
        SELECT 
            CASE 
                WHEN EXISTS (
                    SELECT 1 FROM date_prices 
                    WHERE DATE(timestamp) = DATE(:target_date) AND open IS NOT NULL
                ) THEN (
                    SELECT open FROM date_prices 
                    WHERE DATE(timestamp) = DATE(:target_date)
                )
                ELSE (
                    SELECT close FROM date_prices
                    LIMIT 1
                )
            END as price
    """)
    
    # Similar query for US10Y yield
    us10y_query = text("""
        WITH date_prices AS (
            SELECT ticker, timestamp, open, close
            FROM stocks_dailycandle
            WHERE ticker = 'US10Y' AND timestamp <= :target_date
            ORDER BY timestamp DESC
            LIMIT 2
        )
        SELECT 
            CASE 
                WHEN EXISTS (
                    SELECT 1 FROM date_prices 
                    WHERE DATE(timestamp) = DATE(:target_date) AND open IS NOT NULL
                ) THEN (
                    SELECT open FROM date_prices 
                    WHERE DATE(timestamp) = DATE(:target_date)
                )
                ELSE (
                    SELECT close FROM date_prices
                    LIMIT 1
                )
            END as price
    """)
    
    try:
        # Explicitly convert all values to float to avoid decimal.Decimal issues
        spy_price = float(session.execute(spy_query, {"target_date": target_date}).scalar())
        vix_value = float(session.execute(vix_query, {"target_date": target_date}).scalar())
        us10y_value = float(session.execute(us10y_query, {"target_date": target_date}).scalar())
        
        print(f"SPY price: ${spy_price:.2f}")
        print(f"VIX value: {vix_value:.2f}")
        print(f"US10Y yield: {us10y_value:.2f}%")
        
        return spy_price, vix_value, us10y_value
    
    except Exception as e:
        print(f"Error fetching values: {e}")
        return None, None, None
    finally:
        session.close()

def main():
    """Main function to run the model for the current day"""
    try:
        print("\n" + "=" * 60)
        print("Running TfStraightUpDownModel_v0.1 for Today")
        print("=" * 60)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        date_to_get = input(f"Enter date to run model for (default is today {today}): ").strip()
        if date_to_get:
            try:
                datetime.strptime(date_to_get, '%Y-%m-%d')
            except ValueError:
                print("Invalid date format. Using today's date.")
                date_to_get = today
        else:
            date_to_get = today
        
        # Step 1: Load the model and metadata
        model, metadata = load_model_and_metadata()
        
        # Step 2: Get latest market data
        market_data = get_latest_market_data()
        
        # Step 3: Prepare features for the model
        model_features = prepare_model_features(metadata, override_date=date_to_get)
        
        # Step 4: Make prediction
        prediction, probabilities, features_used = make_prediction(model, model_features, metadata)
        
        # Step 5: Get latest values for option pricing
        spy_price, vix_value, us10y_value = get_latest_values(at_date=date_to_get)
        
        if spy_price is None or vix_value is None or us10y_value is None:
            print("Could not get latest market values. Exiting.")
            return
        
        # Step 6: Select option trade
        option_trade = select_option_trade(
            prediction=prediction,
            spy_price=spy_price,
            vix_value=vix_value,
            us10y_value=us10y_value,
            max_affordability=10.00  # Maximum option premium of $10
        )
        
        # Step 7: Display results
        print("\n" + "=" * 60)
        print("MODEL PREDICTION SUMMARY")
        print("=" * 60)
        
        prediction_map = {0: "DOWN", 1: "FLAT/NEUTRAL", 2: "UP"}
        print(f"Date: {date_to_get}")
        print(f"Prediction: {prediction_map[prediction]}")
        print(f"Confidence: UP: {probabilities:.4f}")
        
        print("\n" + "=" * 60)
        print("RECOMMENDED OPTION TRADE")
        print("=" * 60)
        
        if option_trade:
            print(f"Direction: {option_trade['direction'].upper()}")
            print(f"Strike: ${option_trade['strike']:.2f}")
            print(f"Premium: ${option_trade['premium']:.2f}")
            print(f"SPY Price: ${option_trade['spot_price']:.2f}")
            print(f"\nReasoning: {option_trade['reason']}")
        else:
            print("No option trade recommended for today.")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error running model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()