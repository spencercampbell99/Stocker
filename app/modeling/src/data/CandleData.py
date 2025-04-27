"""
Module for handling candle/price data using Alpaca API.

This module provides functionality to retrieve historical price data
in various timeframes (daily, hourly, 30min, 5min) and store it in the database.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo
from typing import Dict, Optional
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import pandas as pd


from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from alpaca.data.timeframe import TimeFrameUnit

from data.AlpacaConnector import AlpacaConnector
from db.database import StockSymbol, init_db, get_session, get_engine, DailyCandle, HourlyCandle, FiveMinuteCandle, ThirtyMinuteCandle
from datetime import timedelta


class CandleDataManager:
    """
    Class to manage candle/price data operations using Alpaca API.
    Handles fetching, processing, and storing price information.
    """
    
    TIMEFRAMES = {
        'daily': TimeFrame.Day,
        'hourly': TimeFrame.Hour,
        '30min': TimeFrame(30, TimeFrameUnit.Minute),
        '5min': TimeFrame(5, TimeFrameUnit.Minute)
    }
    
    # Define timeframe hierarchy for aggregation
    TIMEFRAME_HIERARCHY = {
        'daily': ['hourly', '30min', '5min'],
        'hourly': ['30min', '5min'],
        '30min': ['5min'],
        '5min': []
    }
    
    def __init__(self):
        """Initialize the candle data manager with API connections."""
        # Use the centralized Alpaca connector
        self.alpaca = AlpacaConnector()
        self.eastern_tz = ZoneInfo("America/New_York")
        
    def ensure_database_schema(self):
        """Ensure that the database tables exist."""
        try:
            # Use SQLAlchemy's metadata to create tables if they don't exist
            init_db()
            return True
        except Exception as e:
            print(f"Error ensuring database schema: {e}")
            return False
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string in YYYY-MM-DD format to datetime object."""
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=self.eastern_tz)
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string in HH:MM format to time object."""
        return datetime.strptime(time_str, "%H:%M").time()
    
    def get_candle_data(self, 
                       ticker: str, 
                       timeframe: str, 
                       start_date: str, 
                       end_date: str,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None) -> pd.DataFrame:
        """
        Get candle data for a specific ticker and timeframe.
        
        Args:
            ticker: The stock ticker to get data for
            timeframe: One of 'daily', 'hourly', '30min', '5min'
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            start_time: Optional start time in HH:MM format (24h)
            end_time: Optional end time in HH:MM format (24h)
            
        Returns:
            DataFrame containing candle data
        """
        try:
            # Convert string dates to datetime objects
            start = self._parse_date(start_date)
            end = self._parse_date(end_date)
            end = end.replace(hour=23, minute=59, second=59)
            
            # if end is after now, set to now - 20 minutes
            if end > datetime.now(self.eastern_tz):
                end = datetime.now(self.eastern_tz) - timedelta(minutes=20)
            
            # Use the appropriate timeframe
            if timeframe not in self.TIMEFRAMES:
                raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(self.TIMEFRAMES.keys())}")
            
            tf = self.TIMEFRAMES[timeframe]
            
            # Create the request
            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                start=start,
                end=end,
                adjustment=Adjustment.SPLIT
            )
            
            # Get bars from Alpaca
            bars = self.alpaca.data_client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df if hasattr(bars, 'df') else pd.DataFrame()
            
            # Return empty DataFrame if no data
            if df.empty:
                print(f"No data found for {ticker} from {start_date} to {end_date}")
                return df
            
            # Time adjust (accounting for daylight savings UTC->EST)
            time_adjust = timedelta(hours=4) if self.eastern_tz.utcoffset(datetime.now()) == timedelta(hours=-4) else timedelta(hours=5)
            
            # Make sure timezone is set to Eastern Time for multi-index DataFrame
            # For multi-index DataFrame, get the timestamp level and convert timezone
            if isinstance(df.index, pd.MultiIndex):
                # Get the timestamp level (second level) and convert timezone
                timestamps = df.index.get_level_values(1).tz_convert(self.eastern_tz)
                
                # Adjust time back 4 hours and 1 timeframe (no idea why this is happening, seems to return in wrong tz)
                timestamps = timestamps - time_adjust
                
                # Recreate the multi-index with the converted timestamps
                df.index = pd.MultiIndex.from_arrays([df.index.get_level_values(0), timestamps])
            else:
                # For single-level index, convert directly
                df.index = df.index.tz_convert(self.eastern_tz) - time_adjust
            
            # Filter by time of day if specified
            if start_time and end_time and timeframe != 'daily':
                start_t = self._parse_time(start_time)
                end_t = self._parse_time(end_time)
                
                # Create time objects for each timestamp for comparison
                df['time_of_day'] = df.index.map(lambda x: x.time())
                df = df[(df['time_of_day'] >= start_t) & (df['time_of_day'] <= end_t)]
                df = df.drop('time_of_day', axis=1)
            
            return df
            
        except Exception as e:
            print(f"Error fetching candle data for {ticker}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def save_candle_data(self, ticker: str, timeframe: str, data: pd.DataFrame) -> Dict:
        """
        Save candle data to the database using a bulk insert with temp table approach.
        Works with existing tables created by Django app.
        
        Args:
            ticker: The stock ticker
            timeframe: The timeframe of the data
            data: DataFrame containing candle data
            
        Returns:
            Dict with success information
        """
        if data.empty:
            return {'success': False, 'message': 'No data to save', 'count': 0}
        
        engine = get_engine()
        
        try:
            # Verify symbol exists
            with get_session() as session:
                symbol_obj = session.query(StockSymbol).filter_by(symbol=ticker).first()
                if not symbol_obj:
                    raise ValueError(f"Symbol {ticker} not found in database. Please load it first.")
            
            # Select the appropriate candle model based on timeframe
            candle_models = {
                'daily': DailyCandle,
                'hourly': HourlyCandle,
                '30min': ThirtyMinuteCandle,
                '5min': FiveMinuteCandle
            }
            
            if timeframe not in candle_models:
                raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(candle_models.keys())}")
            
            CandleModel = candle_models[timeframe]
            target_table = CandleModel.__tablename__
            
            # Add symbol column to the DataFrame
            df_with_symbol = data.copy()
            df_with_symbol['ticker'] = ticker
            
            # Reset the index to get timestamp as a column
            df_with_symbol = df_with_symbol.reset_index().rename(columns={'index': 'timestamp'})
            
            # reduce df to match cols in target table
            df_with_symbol = df_with_symbol[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

            with engine.begin() as connection:
                # Batch insert using SQLAlchemy ORM
                connection.execute(
                    text(f"DELETE FROM {target_table} WHERE ticker = :ticker AND timestamp BETWEEN :start AND :end"),
                    {'ticker': ticker, 'start': df_with_symbol['timestamp'].min(), 'end': df_with_symbol['timestamp'].max()}
                )
                
                connection.execute(
                    CandleModel.__table__.insert(),
                    df_with_symbol.to_dict(orient='records')
                )
                
                count = df_with_symbol.shape[0]
                
                return {'success': True, 'message': f'Successfully saved {count} candles using bulk insert', 'count': count}
            
        except SQLAlchemyError as e:
            # Transaction is automatically rolled back by the context manager
            return {'success': False, 'message': f'Database error: {str(e)}', 'count': 0}
        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}', 'count': 0}
        finally:
            pass