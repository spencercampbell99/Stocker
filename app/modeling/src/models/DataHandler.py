from db.database import get_session, DailyCandle
import pandas as pd
from models.SqlQueries import get_daily_move_status_query, five_min_data_for_ticker_query
import numpy as np

def get_up_down_percent_model_data(start_date="2018-01-01", ticker="SPY", up_threshold=1.0075, down_threshold=0.9925):
    """Get the up/down model data from the database."""
    db = get_session()
    
    # --- Daily Data ---
    daily_query = db.query(DailyCandle).filter(
        DailyCandle.ticker == ticker,
        DailyCandle.timestamp >= start_date
    ).order_by(DailyCandle.timestamp.asc())
    
    data = pd.read_sql(daily_query.statement, db.connection())
    data = data.set_index("timestamp")
    data = data.drop(columns=["id", "ticker"])
    data["date"] = data.index.date

    # --- 5-Minute Data ---
    five_min_query = five_min_data_for_ticker_query(
        ticker=ticker, 
        start_date=start_date, 
        start_time='04:00', 
        end_time='09:29'
    )
    data_5min = pd.read_sql(five_min_query, db.connection())
    data_5min = data_5min.set_index("timestamp")
    data_5min["date"] = data_5min.index.date

    # --- HL Data ---
    hl_query = get_daily_move_status_query(
        start_date=start_date,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        ticker=ticker
    )
    hl_data = pd.read_sql(hl_query, db.connection())
    hl_data['date'] = pd.to_datetime(hl_data['date']).dt.date  # Ensure date format matches

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

    # --- Slope Calculation Function ---
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

    # --- Daily Slopes ---
    data['daily_ma9_slope'] = data['MA9'].rolling(window=3).apply(calculate_slope, raw=False)
    data['daily_ma20_slope'] = data['MA20'].rolling(window=5).apply(calculate_slope, raw=False)

    # --- 5-Minute Slopes ---
    # Filter to ensure only pre-market data is included
    data_5min = data_5min.between_time('04:00', '09:29')
    
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
    data = data.drop(columns=['open', "volume"])
    data_5min = data_5min.drop(columns=["open", "high", "low", "close", "volume"])
    
    # --- Merge Data ---
    # Merge 5-minute slopes into daily data
    data_5min_agg = data_5min.groupby('date').last()  # Take last pre-market slope
    data = data.merge(data_5min_agg, on='date', how='left', suffixes=('', '_5min'))
    
    # Merge HL data
    data = data.merge(hl_data, on='date', how='left')
    
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
        'move_status',
        'premarket_pct_change',
        'last_pm_9ma_diff',
        'last_pm_20ma_diff',
        'daily_9ma_diff',
        'daily_20ma_diff',
    ])
    
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

# get_up_down_percent_model_data(start_date="2018-01-01", ticker="SPY", up_threshold=1.005, down_threshold=0.995)