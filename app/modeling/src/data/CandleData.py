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

def adjust_timezone_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    eastern_tz = ZoneInfo("America/New_York")
    if isinstance(df.index, pd.MultiIndex):
        timestamps = df.index.get_level_values(1)
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')
        
        # Vectorized adjustment
        adjustments = timestamps.map(
            lambda x: timedelta(hours=-4) 
            if x.tz_convert(eastern_tz).dst() 
            else timedelta(hours=-5)
        )
        
        df.index = pd.MultiIndex.from_arrays([
            df.index.get_level_values(0),
            timestamps + adjustments
        ])
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        adjustments = df.index.map(
            lambda x: timedelta(hours=-4) 
            if x.tz_convert(eastern_tz).dst() 
            else timedelta(hours=-5)
        )
        df.index += adjustments
    
    return df
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
    
    def get_vix_candle_data(self):
        """
        Load daily VIX data from 1990 to present from CSV.
        URL: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
        
        Returns:
            Results of the operation as a dictionary.
        """
        try:
            vix_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
            vix_data = pd.read_csv(vix_url, parse_dates=['DATE'])
            vix_data = vix_data.rename(columns={'CLOSE': 'close', 'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'DATE': 'timestamp'})
            
            # Fill volume with 0s
            vix_data['volume'] = 0
            
            # Update timestamp by appending 00:00:00 to the date
            vix_data['timestamp'] = pd.to_datetime(vix_data['timestamp'].dt.strftime('%Y-%m-%d 00:00:00'))
            
            # Add ticker
            vix_data['ticker'] = 'VIX'
        except Exception as e:
            return {'success': False, 'message': f"Error loading VIX data: {e}"}
        
        try:
            # Load data into stocks_dailycandle table
            with get_session() as session:
                # Delete existing VIX data
                session.query(DailyCandle).filter_by(ticker='VIX').delete()
                
                # Insert vix data into the database
                vix_data = vix_data[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                session.bulk_insert_mappings(DailyCandle, vix_data.to_dict(orient='records'))
                session.commit()
        except SQLAlchemyError as e:
            return {'success': False, 'message': f"Database error: {str(e)}"}
        except Exception as e:
            return {'success': False, 'message': f"Error saving VIX data: {e}"}
        
        return {'success': True, 'message': f'Successfully saved all daily VIX candles since 1990', 'count': len(vix_data)}
    
    def get_xsp_candle_data(self, last_week_only: bool = False):
        """
        Load XSP daily data back to 2017-01-01 or just back 1 week if last_week_only is True.
        Uses yfinance.
        
        Args:
            last_week_only: If True, only load data from the last week using yfinance.
        Returns:
            Results of the operation as a dictionary.
        """
        try:
            import yfinance as yf
            import pandas as pd
            from datetime import datetime, timedelta
            
            if last_week_only:
                # For last week only, use yfinance since it's more up-to-date than FRED
                start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                end = datetime.now().strftime("%Y-%m-%d")
            else:
                # For historical data, still use yfinance
                start = "2017-01-01"
                end = datetime.now().strftime("%Y-%m-%d")
                
            # Load XSP data using yfinance
            xsp_data = yf.download("^XSP", start=start, end=end, interval="1d")
            
            # Handle multi-index columns if present
            if isinstance(xsp_data.columns, pd.MultiIndex):
                xsp_data.columns = xsp_data.columns.get_level_values(0)
            
            xsp_data = xsp_data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'})
            
            # if empty, error
            if xsp_data.empty:
                return {'success': False, 'message': 'No data found for XSP'}
                
            # Reset index to get timestamp as a column
            xsp_data.reset_index(inplace=True)
            
            # Add 00:00:00 to the date
            xsp_data['Date'] = pd.to_datetime(xsp_data['Date'].dt.strftime('%Y-%m-%d 00:00:00'))
            xsp_data = xsp_data.rename(columns={'Date': 'timestamp'})
            
            # Add ticker
            xsp_data['ticker'] = 'XSP'
            
            # Ensure volume column exists
            if 'Volume' in xsp_data.columns:
                xsp_data.rename(columns={'Volume': 'volume'}, inplace=True)
            elif 'volume' not in xsp_data.columns:
                xsp_data['volume'] = 0

        except Exception as e:
            return {'success': False, 'message': f"Error loading XSP data: {e}"}
    
        print(xsp_data)
    
        try:
            # Load data into stocks_dailycandle table
            with get_session() as session:
                if last_week_only:
                    # Delete existing XSP data for the last week
                    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    session.query(DailyCandle).filter_by(ticker='XSP').filter(DailyCandle.timestamp >= start_date).delete()
                else:
                    # Delete existing XSP data
                    session.query(DailyCandle).filter_by(ticker='XSP').delete()
                
                # Insert XSP data into the database
                xsp_data = xsp_data[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                session.bulk_insert_mappings(DailyCandle, xsp_data.to_dict(orient='records'))
                session.commit()
                
            return {'success': True, 'message': f'Successfully saved {"weekly" if last_week_only else "all"} daily XSP candles', 'count': len(xsp_data)}
        except SQLAlchemyError as e:
            return {'success': False, 'message': f"Database error: {str(e)}"}
        except Exception as e:
            return {'success': False, 'message': f"Error saving XSP data: {e}"}
    
    def get_10_year_treasury_candle_data(self, last_week_only: bool = False):
        """
        Load daily 10-year treasury data using FRED for historical data and yfinance for recent data.
        
        Args:
            last_week_only: If True, only load data from the last week using yfinance.
        
        Returns:
            Results of the operation as a dictionary.
        """
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            
            if last_week_only:
                # For last week only, use yfinance since it's more up-to-date than FRED
                import yfinance as yf
                import time
                import random
                
                # Get the last week's date range
                start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                end = datetime.now().strftime("%Y-%m-%d")
                
                # Add retry logic with exponential backoff for yfinance
                max_retries = 5
                retry_count = 0
                treasury_data = None
                
                while retry_count < max_retries:
                    try:
                        # Load 10-year treasury data
                        treasury_data = yf.download("^TNX", start=start, end=end, interval="1d")
                        
                        # If successful, break out of the retry loop
                        if not treasury_data.empty:
                            break
                            
                    except Exception as download_error:
                        print(f"Attempt {retry_count + 1} failed: {download_error}")
                        
                    # Exponential backoff with jitter
                    sleep_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    retry_count += 1
                
                if treasury_data is None or treasury_data.empty:
                    return {'success': False, 'message': 'No data found for 10-year treasury after multiple attempts.'}
                
                # Handle multi-index columns if present
                if isinstance(treasury_data.columns, pd.MultiIndex):
                    treasury_data.columns = treasury_data.columns.get_level_values(0)
                
                # Standard column renaming for yfinance data
                treasury_data = treasury_data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'})
                
                # Reset index to get timestamp as a column
                treasury_data.reset_index(inplace=True)
                
                # Add 00:00:00 to the date
                treasury_data['Date'] = pd.to_datetime(treasury_data['Date'].dt.strftime('%Y-%m-%d 00:00:00'))
                treasury_data = treasury_data.rename(columns={'Date': 'timestamp'})
                
                # Add ticker
                treasury_data['ticker'] = 'US10Y'
                
                # Ensure volume column exists
                if 'Volume' in treasury_data.columns:
                    treasury_data.rename(columns={'Volume': 'volume'}, inplace=True)
                elif 'volume' not in treasury_data.columns:
                    treasury_data['volume'] = 0
                
            else:
                # For historical data, use FRED which is more reliable for long-term data
                import pandas_datareader.data as web
                
                start = "2000-01-01"
                
                # Load 10-year treasury data from FRED
                # DGS10 is the 10-Year Treasury Constant Maturity Rate
                treasury_data = web.DataReader('DGS10', 'fred', start=start)
                
                # Check if data is empty
                if treasury_data.empty:
                    return {'success': False, 'message': 'No data found for 10-year treasury.'}
                
                # FRED data only includes the close price, so we'll set all OHLC values to this value
                treasury_data['open'] = treasury_data['DGS10']
                treasury_data['high'] = treasury_data['DGS10']
                treasury_data['low'] = treasury_data['DGS10']
                treasury_data['close'] = treasury_data['DGS10']
                treasury_data['volume'] = 0  # No volume data for Treasury
                
                # Drop the original column
                treasury_data = treasury_data.drop('DGS10', axis=1)
                
                # Reset index to get timestamp as a column
                treasury_data.reset_index(inplace=True)
                
                # Format the date
                treasury_data['DATE'] = pd.to_datetime(treasury_data['DATE'].dt.strftime('%Y-%m-%d 00:00:00'))
                treasury_data = treasury_data.rename(columns={'DATE': 'timestamp'})
                
                # Add ticker
                treasury_data['ticker'] = 'US10Y'
                
                # Handle NaN values (weekends/holidays)
                treasury_data = treasury_data.dropna()
        except Exception as e:
            return {'success': False, 'message': f"Error loading 10-year treasury data: {e}"}
        
        try:
            # Load data into stocks_dailycandle table
            with get_session() as session:
                if last_week_only:
                    # Delete existing treasury data for the last week
                    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    session.query(DailyCandle).filter_by(ticker='US10Y').filter(DailyCandle.timestamp >= start_date).delete()
                else:
                    # Delete existing treasury data
                    session.query(DailyCandle).filter_by(ticker='US10Y').delete()
                
                # Insert treasury data into the database with consistent column ordering
                treasury_data = treasury_data[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                session.bulk_insert_mappings(DailyCandle, treasury_data.to_dict(orient='records'))
                session.commit()
        except SQLAlchemyError as e:
            return {'success': False, 'message': f"Database error: {str(e)}"}
        except Exception as e:
            return {'success': False, 'message': f"Error saving 10-year treasury data: {e}"}
        
        return {'success': True, 'message': f'Successfully saved {"weekly" if last_week_only else "all"} daily 10-year treasury candles', 'count': len(treasury_data)}
    
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
            
            df = adjust_timezone_vectorized(df)
            
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