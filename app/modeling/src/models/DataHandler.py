from db.database import get_session, DailyCandle
import pandas as pd
from models.SqlQueries import get_daily_move_status_query, five_min_data_for_ticker_query
import numpy as np

db = get_session()

def get_daily_data_for_ticker(ticker, start_date):
    """Get daily data for a specific ticker from the database."""
    daily_query = db.query(DailyCandle).filter(
        DailyCandle.ticker == ticker,
        DailyCandle.timestamp >= start_date
    ).order_by(DailyCandle.timestamp.asc())
    
    data = pd.read_sql(daily_query.statement, db.connection())
    data = data.set_index("timestamp")
    data = data.drop(columns=["id", "ticker"])
    data["date"] = data.index.date
    return data

def get_five_min_data_for_ticker(ticker, start_date):
    """Get 5-minute data for a specific ticker from the database."""
    five_min_query = five_min_data_for_ticker_query(
        ticker=ticker, 
        start_date=start_date, 
        start_time='04:00', 
        end_time='09:29'
    )
    data_5min = pd.read_sql(five_min_query, db.connection())
    data_5min = data_5min.set_index("timestamp")
    data_5min["date"] = data_5min.index.date
    return data_5min

def calculate_slope(series):
    """Calculate slope of % changes, handling NaNs."""
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return np.nan
    pct_changes = series_clean.pct_change().dropna()
    if len(pct_changes) < 2:
        return np.nan
    x = np.arange(len(pct_changes))
    y = pct_changes.values * 100  # Convert to percentage
    return np.polyfit(x, y, 1)[0]  # Return only slope

def calculate_slope_vectorized(series):
    """Vectorized calculation of slope of % changes, handling NaNs."""
    # Drop NaNs for rolling window, but keep index alignment
    pct_changes = series.pct_change() * 100
    n = len(pct_changes)
    if n < 2:
        return np.nan
    x = np.arange(n)
    mask = ~np.isnan(pct_changes)
    if mask.sum() < 2:
        return np.nan
    # Only use valid (non-NaN) values
    x_valid = x[mask]
    y_valid = pct_changes[mask]
    slope = np.polyfit(x_valid, y_valid, 1)[0]
    return slope

def calculate_realized_volatility(series, n):
    """Calculate realized volatility of a series
    
    Formula: 100 * sqrt((252 / n) * summation 1 -> n of R^2)
    
    where R = ln(Pt / Pt-1)
    and Pt is the price at time t.
    
    Args:
        series (pd.Series): The series to calculate volatility for.
        n (int): The number of periods to use for the calculation.
    """
    log_returns = np.log(series / series.shift(1))
    squared_returns = log_returns ** 2
    realized_volatility = np.sqrt((252 / n) * squared_returns.rolling(window=n).sum())
    return realized_volatility * 100

def calculate_bollinger_band_position(series, n, clip_values=True):
    """Calculate the position of price within Bollinger Bands.
    
    This returns a normalized value where:
    - Values below -1 indicate price is below the lower band
    - Values between -1 and 1 indicate price is between bands
    - Values above 1 indicate price is above the upper band
    
    Args:
        series (pd.Series): Price series (typically close prices)
        n (int): The number of periods for the moving average
        clip_values (bool): Whether to clip values to [-1, 1] range
        
    Returns:
        pd.Series: Bollinger Band position values
    """
    # Calculate bands as before
    middle_band = series.rolling(window=n).mean()
    std_dev = series.rolling(window=n).std()
    upper_band = middle_band + (2 * std_dev)
    lower_band = middle_band - (2 * std_dev)
    
    band_width = upper_band - lower_band
    band_width = band_width.replace(0, np.nan)
    
    position = (series - middle_band) / (0.5 * band_width)
    
    if clip_values:
        position = position.clip(-1, 1)
    
    return position

def calculate_bollinger_band_position_vectorized(series, n, clip_values=True):
    """Vectorized calculation of the position of price within Bollinger Bands."""
    middle_band = series.rolling(window=n).mean()
    std_dev = series.rolling(window=n).std()
    upper_band = middle_band + (2 * std_dev)
    lower_band = middle_band - (2 * std_dev)
    band_width = upper_band - lower_band
    band_width = band_width.replace(0, np.nan)
    position = (series - middle_band) / (0.5 * band_width)
    if clip_values:
        position = position.clip(-1, 1)
    return position

def calculate_realized_volatility_vectorized(series, n):
    """Vectorized calculation of realized volatility."""
    log_returns = np.log(series / series.shift(1))
    squared_returns = log_returns ** 2
    rolling_sum = squared_returns.rolling(window=n).sum()
    realized_volatility = np.sqrt((252 / n) * rolling_sum) * 100
    return realized_volatility

def calculate_average_true_range(data, n):
    """Calculate the Average True Range (ATR) for a given data series.
    
    Formula: ATR = (ATR(t-1) * (n-1) + TR(t)) / n
    where TR = max(high - low, abs(high - close(t-1)), abs(low - close(t-1)))
    
    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        n (int): The number of periods to use for the calculation.
        
    Returns:
        pd.Series: The ATR values.
    """
    raise NotImplementedError("This function is not implemented yet.")
    
    high_low = data['high'] - data['low']
    high_prev_close = np.abs(data['high'] - data['close'].shift(1))
    low_prev_close = np.abs(data['low'] - data['close'].shift(1))
    
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    atr = true_range.rolling(window=n).mean()
    return atr

def get_up_down_percent_model_data(start_date="2018-01-01", ticker="SPY", up_threshold=1.0075, down_threshold=0.9925, skip_move_status=False, open_override=None):
    """Get the up/down model data from the database."""

    # --- Daily Data ---
    data = get_daily_data_for_ticker(ticker, start_date)

    # --- 5-Minute Data ---
    data_5min = get_five_min_data_for_ticker(ticker, start_date)

    # --- Moving Averages ---
    # Daily MAs
    data["MA9"] = data["close"].rolling(window=9, min_periods=1).mean()
    data["MA20"] = data["close"].rolling(window=20, min_periods=1).mean()
    
    # 5-Minute MAs (grouped by date)
    data_5min["pm_MA9"] = data_5min.groupby('date')['close'].transform(
        lambda x: x.rolling(window=9, min_periods=1).mean()
    )
    data_5min["pm_MA20"] = data_5min.groupby('date')['close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    # --- Daily Slopes ---
    data['daily_ma9_slope'] = data['MA9'].rolling(window=3).apply(calculate_slope, raw=False)
    data['daily_ma20_slope'] = data['MA20'].rolling(window=5).apply(calculate_slope, raw=False)

    # --- 5-Minute Slopes ---
    # Calculate MA slopes for the last window of each day's pre-market session
    last_points = data_5min.groupby('date').tail(3)
    
    # Group by date and calculate slope for each day's last window
    slope_df = last_points.groupby('date').apply(
        lambda group: pd.Series({
            '5min_premarket_9ma_slope': group['pm_MA9'].iloc[-3:].pct_change().mean() * 100,
            '5min_premarket_20ma_slope': group['pm_MA20'].iloc[-3:].pct_change().mean() * 100,
            'last_pm_candle_open': group['open'].iloc[-1],
            'last_pm_9ma': group['pm_MA9'].iloc[-1],
            'last_pm_20ma': group['pm_MA20'].iloc[-1],
        }),
        include_groups=False
    )
    
    # Merge slope data back to original dataframe
    data_5min = data_5min.join(slope_df, on='date', how='left')

    # --- Cleanup ---
    data = data.drop(columns=["volume"])
    data_5min = data_5min.drop(columns=["open", "high", "low", "close", "volume"])
    
    # --- Merge Data ---
    # Merge 5-minute slopes into daily data
    data_5min_agg = data_5min.groupby('date').last()  # Take last pre-market slope
    data = data.merge(data_5min_agg, on='date', how='left', suffixes=('', '_5min'))
    
    if not skip_move_status:
        # --- HL Data ---
        hl_query = get_daily_move_status_query(
            start_date=start_date,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
            ticker=ticker
        )
        hl_data = pd.read_sql(hl_query, db.connection())
        hl_data['date'] = pd.to_datetime(hl_data['date']).dt.date  # Ensure date format matches
        
        # Merge HL data
        data = data.merge(hl_data, on='date', how='left')
    
    # Override open value for the last row if open_override is provided
    if open_override is not None:
        last_idx = data.index[-1]
        data.at[last_idx, 'open'] = open_override if not data.at[last_idx, 'open'] else data.at[last_idx, 'open']
    
    # --- Premarket % Change ---
    data['premarket_pct_change'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1) * 100
    
    # --- Additonal Features ---
    # Calculate % diff of last pm candle open and last pm 9ma and 20ma as well as daily 9ma and 20ma
    data['last_pm_9ma_diff'] = (data['last_pm_9ma'] / data['last_pm_candle_open'] - 1) * 100
    data['last_pm_20ma_diff'] = (data['last_pm_20ma'] / data['last_pm_candle_open'] - 1) * 100
    data['daily_9ma_diff'] = (data['MA9'] / data['last_pm_candle_open'] - 1) * 100
    data['daily_20ma_diff'] = (data['MA20'] / data['last_pm_candle_open'] - 1) * 100
    
    # --- Final Cleanup ---
    data = data.dropna(subset=[
        'daily_ma9_slope', 
        'daily_ma20_slope', 
        '5min_premarket_9ma_slope',
        '5min_premarket_20ma_slope',
        'premarket_pct_change',
        'last_pm_9ma_diff',
        'last_pm_20ma_diff',
        'daily_9ma_diff',
        'daily_20ma_diff',
    ])
    
    if not skip_move_status:
        data = data.dropna(subset=['move_status'])
    
    # Set index to date
    data = data.set_index('date')
    data = data.sort_index()
    
    # insert data into a temp table
    # from sqlalchemy import text
    # db.execute(text("DROP TABLE IF EXISTS temp_table"))
    # db.execute(text('CREATE TABLE temp_table (date DATE, open FLOAT, daily_ma9_slope FLOAT, daily_ma20_slope FLOAT, "5min_premarket_9ma_slope" FLOAT, "5min_premarket_20ma_slope" FLOAT, move_status INT, premarket_pct_change FLOAT, last_pm_9ma_diff FLOAT, last_pm_20ma_diff FLOAT, daily_9ma_diff FLOAT, daily_20ma_diff FLOAT)'))
    # data[['open', 'daily_ma9_slope', 'daily_ma20_slope', '5min_premarket_9ma_slope', '5min_premarket_20ma_slope', 'move_status', 'premarket_pct_change', 'last_pm_9ma_diff', 'last_pm_20ma_diff', 'daily_9ma_diff', 'daily_20ma_diff']] \
    #     .to_sql('temp_table', db.connection(), if_exists='append', index=True, index_label='date')
    # db.connection().commit()
    
    
    return data

def get_percent_move_model_data(start_date="2018-01-01", ticker="SPY", open_override=None):
    """Get the up/down model data from the database."""
    
    data = get_daily_data_for_ticker(ticker, start_date)
    data_5min = get_five_min_data_for_ticker(ticker, start_date)
    
    vix_data = get_daily_data_for_ticker("VIX", start_date)
    us10y_data = get_daily_data_for_ticker("US10Y", start_date)
    
    # For both VIX and US10Y df, leave only date and open/close renamed to vix_open/vix_close and us10y_open/us10y_close
    vix_data = vix_data[['open', 'close', 'date']].rename(columns={'open': 'vix_open', 'close': 'vix_close'})
    us10y_data = us10y_data[['open', 'close', 'date']].rename(columns={'open': 'us10y_open', 'close': 'us10y_close'})
    
    # Merge VIX and US10Y data with daily data
    data = data.merge(vix_data, on='date', how='left')
    data = data.merge(us10y_data, on='date', how='left')
    
    # set index to date
    data = data.set_index('date')
    data_5min = data_5min.set_index('date')
    
    # SPY Calculate MA9/MA20 for daily and 5min data
    data["MA9"] = data["close"].rolling(window=9, min_periods=1).mean()
    data["MA20"] = data["close"].rolling(window=20, min_periods=1).mean()
    data_5min["pm_MA9"] = data_5min.groupby('date')['close'].transform(
        lambda x: x.rolling(window=9, min_periods=1).mean()
    )
    data_5min["pm_MA20"] = data_5min.groupby('date')['close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    data_5min["pm_MA9"] = data_5min["pm_MA9"]
    data_5min["pm_MA20"] = data_5min["pm_MA20"]
    
    # Calculate slopes for daily and 5min data (only last calculation per day for premarket data)
    data['daily_ma9_slope'] = data['MA9'].rolling(window=3).apply(calculate_slope_vectorized, raw=False)
    data['daily_ma20_slope'] = data['MA20'].rolling(window=5).apply(calculate_slope_vectorized, raw=False)
    
    # Only calculate slope on the last n (e.g., 3) candles of each premarket session
    n = 3
    def last_n_slope(x):
        if len(x) < n:
            return np.nan
        return calculate_slope_vectorized(x.iloc[-n:])
    data_5min['5min_premarket_9ma_slope'] = data_5min.groupby('date')['pm_MA9'].transform(last_n_slope)
    data_5min['5min_premarket_20ma_slope'] = data_5min.groupby('date')['pm_MA20'].transform(last_n_slope)
    data_5min = data_5min.groupby('date').last()  # Take last pre-market slope
    
    # Calculate bollinger band position for daily data (optimized, reusable)
    data['bb_position'] = calculate_bollinger_band_position_vectorized(data['close'], n=20, clip_values=False)
    data['realized_volatility'] = calculate_realized_volatility_vectorized(data['close'], n=5)

    # Join data
    data = data.merge(data_5min[['5min_premarket_9ma_slope', '5min_premarket_20ma_slope', 'pm_MA9', 'pm_MA20']], on='date', how='left')
    data = data.drop(columns=["volume"])
    
    # Calculate % change from previous close to open (pm % change)
    if open_override is not None:
        last_idx = data.index[-1]
        data.at[last_idx, 'open'] = open_override if not data.at[last_idx, 'open'] else data.at[last_idx, 'open']
    
    data['premarket_pct_change'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1) * 100
    
    # Clean
    data = data.dropna(subset=[
        'daily_ma9_slope', 
        'daily_ma20_slope', 
        '5min_premarket_9ma_slope',
        '5min_premarket_20ma_slope',
        'bb_position',
        'realized_volatility',
        'vix_open', 'vix_close',
        'us10y_open', 'us10y_close',
        'pm_MA9', 'pm_MA20'
    ])
    
    return data

# data = get_up_down_percent_model_data(start_date="2025-04-01", ticker="SPY", up_threshold=1.005, down_threshold=0.995)
# print(data.tail())

# get_percent_move_model_data(start_date="2017-04-01", ticker="SPY", up_threshold=1.005, down_threshold=0.995)