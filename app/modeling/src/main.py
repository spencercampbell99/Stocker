#!/usr/bin/env python
"""
Stocker API Server
"""

import dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uvicorn
import json

# Import required modules
from data.CandleData import CandleDataManager
from models.DataHandler import get_up_down_percent_model_data
from scripts.model_runs.RunModelv01 import load_model_and_metadata, prepare_model_features, make_prediction, select_option_trade

# Load dotenv
dotenv.load_dotenv()

METADATA = {
    'title': 'Stocker API',
    'description': 'API for stock market data and model predictions',
    'version': '0.1',
}

app = FastAPI(
    title=METADATA['title'],
    description=METADATA['description'],
    version=METADATA['version'],
)

class CandleResponse(BaseModel):
    success: bool
    message: str
    count: int
    data: Optional[List[Dict[str, Any]]] = None

class ModelDataResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    spy_price: float
    vix_value: float
    us10y_value: float
    max_affordability: Optional[float] = 10.0

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    prediction_class: int
    probabilities: Dict[str, float]
    option_trade: Optional[Dict[str, Any]] = None
    message: str = ""

@app.get("/")
def root():
    """Root endpoint to verify the API is working."""
    return {
        "status": "online",
        "service": "Stocker Modeling API",
        "version": METADATA['version'],
    }

@app.get("/api/candles", response_model=CandleResponse)
def load_candles(
    ticker: str = Query(..., description="The stock ticker symbol"),
    timeframe: str = Query("daily", description="Candle timeframe (daily, hourly, 30min, 5min)"),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    start_time: Optional[str] = Query(None, description="Start time in HH:MM format (24h)"),
    end_time: Optional[str] = Query(None, description="End time in HH:MM format (24h)"),
):
    """
    Load candle data for a specific ticker and timeframe.
    If start_date is not provided, it will fetch the last day of data.
    """
    # Initialize candle manager
    candle_manager = CandleDataManager()
    
    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if not start_date:
        # Default to last day
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        match ticker:
            case "VIX":
                # Get the VIX data
                candle_manager.get_vix_candle_data()
                
                return {
                    "success": True,
                    "message": "VIX data loaded successfully",
                    "count": 0,
                    "data": None
                }
            case "US10Y":
                # Get the US10Y data
                candle_manager.get_10_year_treasury_candle_data(last_week_only=True)
                
                return {
                    "success": True,
                    "message": "US10Y data loaded successfully",
                    "count": 0,
                    "data": None
                }
            case _:
                # Get the candle data
                candles_df = candle_manager.get_candle_data(
                    ticker=ticker,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Save to database
                result = candle_manager.save_candle_data(
                    ticker=ticker,
                    timeframe=timeframe,
                    data=candles_df
                )
        
        # Add data to the response
        if result['success'] and not candles_df.empty:
            # Reset index to get timestamp as column
            df_with_timestamp = candles_df.reset_index()
            # Convert to dict for JSON serialization
            result['data'] = json.loads(df_with_timestamp.to_json(orient='records', date_format='iso'))
        
        return result
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error loading candles: {str(e)}",
            "count": 0
        }

@app.get("/api/model-data", response_model=ModelDataResponse)
def get_model_data(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    ticker: str = Query("SPY", description="The stock ticker symbol"),
    up_threshold: float = Query(1.0075, description="Threshold for upward price movement"),
    down_threshold: float = Query(0.9925, description="Threshold for downward price movement"),
):
    """
    Get model data for a specific date or date range.
    If date is not provided, it will use the current date.
    """
    return_data = {}
    
    # If time is not between 9:00am and 10:0am, add warning to return_data
    current_time = datetime.now().time()
    if current_time < datetime.strptime("09:00", "%H:%M").time() or current_time > datetime.strptime("10:00", "%H:%M").time():
        return_data['warning'] = "Data may not be accurate as it is outside the expected time range (9am-10am)."
    
    try:
        # Set default date if not provided
        if not date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            # If specific date is provided, get data from 30 days before that date
            target_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Get model data
        model_data = get_up_down_percent_model_data(
            start_date=start_date,
            ticker=ticker,
            up_threshold=up_threshold,
            down_threshold=down_threshold
        )
        
        # If a specific date was requested, filter to just that date
        if date:
            target_date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            if target_date_obj in model_data.index:
                model_data = model_data.loc[[target_date_obj]]
            else:
                # Build rest of return data
                return_data['success'] = False
                return_data['message'] = f"No data found for {date}"
                return_data['data'] = None
                return return_data
        
        # Convert to dict for JSON serialization
        model_data_dict = model_data.reset_index().to_dict(orient='records')
        
        for record in model_data_dict:
            record['date'] = record['date'].isoformat()
        
        return_data['success'] = True
        return_data['message'] = f"Successfully retrieved model data"
        return_data['data'] = model_data_dict
        return return_data
    
    except Exception as e:
        return_data['success'] = False
        return_data['message'] = f"Error retrieving model data: {str(e)}"
        return_data['data'] = None
        return return_data

@app.post("/api/predict", response_model=PredictionResponse)
def predict_market(
        request: PredictionRequest,
    ):
    """
    Get market prediction and suggested option trade based on 
    current SPY price, VIX, and US10Y values.
    """
    try:
        # Load model and metadata
        model, metadata = load_model_and_metadata()
        
        # Prepare features for the model
        model_features = prepare_model_features(metadata)
        
        # Make prediction
        pred_class, probabilities, features_used = make_prediction(model, model_features, metadata)
        
        # Map prediction class to label
        prediction_map = {0: "DOWN", 1: "FLAT/NEUTRAL", 2: "UP"}
        prediction = prediction_map[pred_class]
        
        # Get option trade suggestion
        option_trade = select_option_trade(
            prediction=pred_class,
            spy_price=request.spy_price,
            vix_value=request.vix_value,
            us10y_value=request.us10y_value,
            max_affordability=request.max_affordability if request.max_affordability else None,
        )
        
        return {
            "success": True,
            "prediction": prediction,
            "prediction_class": int(pred_class),
            "probabilities": {
                "down": float(probabilities[0]),
                "flat": float(probabilities[1]),
                "up": float(probabilities[2])
            },
            "option_trade": option_trade,
            "message": "Prediction successful"
        }
        
    except Exception as e:
        return {
            "success": False,
            "prediction": "ERROR",
            "prediction_class": -1,
            "probabilities": {"down": 0.0, "flat": 0.0, "up": 0.0},
            "option_trade": None,
            "message": f"Error making prediction: {str(e)}"
        }

if __name__ == "__main__":
    PORT = 9000
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)