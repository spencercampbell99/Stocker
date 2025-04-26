from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from datetime import timedelta
from .models import Ticker, DailyCandle, FiveMinCandle, ThirtyMinCandle, HourCandle
import pandas as pd
import json

def stock_dashboard(request):
    """Main dashboard view for stock visualization"""
    tickers = Ticker.objects.filter(is_active=True).order_by('symbol').values('symbol', 'name')
    
    # Get submitted ticker if any
    ticker_symbol = request.GET.get('ticker', '')
    
    context = {
        'tickers': tickers,
        'selected_ticker': ticker_symbol,
    }
    
    return render(request, 'stocks/dashboard.html', context)

@require_http_methods(["GET"])
def get_candle_data(request):
    """API view to fetch candle data for a specific ticker and timeframe"""
    ticker_symbol = request.GET.get('ticker', '')
    timeframe = request.GET.get('timeframe', 'daily')  # daily, hour, thirty_min, five_min
    
    if not ticker_symbol:
        return JsonResponse({'error': 'Ticker symbol is required'}, status=400)
    
    # Validate ticker exists
    try:
        ticker = Ticker.objects.get(symbol=ticker_symbol)
    except Ticker.DoesNotExist:
        return JsonResponse({'error': f'Ticker {ticker_symbol} not found'}, status=404)
    
    # Set date range (last year)
    end_date = timezone.now()
    start_date = end_date - timedelta(days=365)
    
    # Select appropriate model based on timeframe
    if timeframe == 'daily':
        model = DailyCandle
    elif timeframe == 'hour':
        model = HourCandle
    elif timeframe == 'thirty_min':
        model = ThirtyMinCandle
    elif timeframe == 'five_min':
        model = FiveMinCandle
    else:
        return JsonResponse({'error': 'Invalid timeframe'}, status=400)
    
    # Query data
    candles = model.objects.filter(
        ticker=ticker_symbol,
        timestamp__gte=start_date,
        timestamp__lte=end_date
    ).order_by('timestamp')
    
    # Check if we have data
    if not candles.exists():
        return JsonResponse({
            'ticker': ticker_symbol,
            'timeframe': timeframe,
            'candles': [],
            'dates': [],
            'moving_averages': {}
        })
    
    # Process data for chart display with moving averages
    data = []
    dates = []
    
    # Convert QuerySet to lists for JSON serialization
    for candle in candles:
        data.append({
            'timestamp': candle.timestamp.isoformat(),
            'open': float(candle.open),
            'high': float(candle.high),
            'low': float(candle.low),
            'close': float(candle.close),
            'volume': candle.volume
        })
        dates.append(candle.timestamp.isoformat())
    
    # Calculate moving averages if we have enough data
    ma_data = {}
    periods = [20, 50, 200]  # Common moving average periods
    
    if data:
        # Convert to pandas DataFrame for easier calculation
        df = pd.DataFrame(data)
        for period in periods:
            if len(df) >= period:
                ma_series = df['close'].rolling(window=period).mean()
                # Convert to list and handle NaN values
                ma_data[f'MA{period}'] = [float(x) if pd.notna(x) else None for x in ma_series]
    
    result = {
        'ticker': ticker_symbol,
        'timeframe': timeframe,
        'candles': data,
        'dates': dates,
        'moving_averages': ma_data
    }
    
    return JsonResponse(result)
