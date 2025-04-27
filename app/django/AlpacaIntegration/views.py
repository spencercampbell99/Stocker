from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import redirect

def api_docs(request):
    """View showing documentation for the Alpaca WebSocket API."""
    # Open in new tab
    return redirect('https://docs.alpaca.markets/docs/getting-started')

def check_api_status(request):
    """Simple endpoint to check if Alpaca API credentials are configured."""
    api_key = settings.ALPACA_API_KEY
    api_secret = settings.ALPACA_API_SECRET
    
    if not api_key or not api_secret:
        return JsonResponse({
            'status': 'error',
            'message': 'Alpaca API credentials not configured'
        }, status=500)
    
    return JsonResponse({
        'status': 'success',
        'message': 'Alpaca API credentials are configured',
        'data_feed': settings.ALPACA_DATA_FEED
    })
