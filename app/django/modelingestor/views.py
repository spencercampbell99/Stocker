from django.views.decorators.http import require_http_methods
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
from stocks.models import DailyCandle, FiveMinCandle

@require_http_methods(["GET"])
def get_v01_model_predictions(request):
    """
    API view to fetch model predictions for a specific ticker and timeframe.

    This endpoint returns model predictions based on the specified parameters.

    Args:
        request (HttpRequest): The HTTP request object.
                            Should include the following query parameters:
                            - ma9_five_min: 9-period moving average on 5-minute timeframe
                            - ma20_five_min: 20-period moving average on 5-minute timeframe
                            - ma9_daily: 9-period moving average on daily timeframe
                            - ma20_daily: 20-period moving average on daily timeframe
                            - vix_price: Current VIX index price
                            - us10y_rate: Current 10-year US Treasury yield rate

    Returns:
        JsonResponse: JSON response containing model predictions.
                    On success, returns status code 200 with prediction data.
                    On error, returns appropriate error status code and message.
    """
    # Extract parameters from the request
    ma9_five_min = request.GET.get('ma9_five_min')
    ma20_five_min = request.GET.get('ma20_five_min')
    ma9_daily = request.GET.get('ma9_daily')
    ma20_daily = request.GET.get('ma20_daily')
    vix_price = request.GET.get('vix_price')
    us10y_rate = request.GET.get('us10y_rate')

    # load last 30 five minute candles and last 30 daily candles
    last_30_five_min_candles = FiveMinCandle.where(
        ticker='SPY',
    ).order_by('-timestamp')[:30]
    last_30_daily_candles = DailyCandle.where(
        ticker='SPY',
    ).order_by('-timestamp')[:30]
    
    # Conver to pandas dataframe
    last_30_five_min_candles_df = pd.DataFrame(list(last_30_five_min_candles.values()))
    last_30_daily_candles_df = pd.DataFrame(list(last_30_daily_candles.values()))
    
    # Invert sorts
    last_30_five_min_candles_df = last_30_five_min_candles_df.sort_values(by='timestamp')
    last_30_daily_candles_df = last_30_daily_candles_df.sort_values(by='timestamp')
    
    # --- Moving Averages ---
    # Daily MAs
    last_30_daily_candles_df["MA9"] = last_30_daily_candles_df["close"].rolling(window=9, min_periods=1).mean()
    last_30_daily_candles_df["MA20"] = last_30_daily_candles_df["close"].rolling(window=20, min_periods=1).mean()
    
    # 5-Minute MAs (grouped by date)
    last_30_five_min_candles_df["pm_MA9"] = last_30_five_min_candles_df['close'].transform(
        lambda x: x.rolling(window=9, min_periods=1).mean()
    )
    last_30_five_min_candles_df["pm_MA20"] = last_30_five_min_candles_df['close'].transform(
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

    # --- Slope Calculation ---
    daily_ma9_slope = calculate_slope(last_30_daily_candles_df["MA9"].tail(3))
    daily_ma20_slope = calculate_slope(last_30_daily_candles_df["MA20"].tail(5))
    pm_ma9_slope = calculate_slope(last_30_five_min_candles_df["pm_MA9"].tail(3))
    pm_ma20_slope = calculate_slope(last_30_five_min_candles_df["pm_MA20"].tail(3))
    
    