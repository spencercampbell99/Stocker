from django.views.decorators.http import require_http_methods
from modelingestor.ModelingApiConnector import ModelingApiConnector
from django.http import JsonResponse
import logging

# Configure logging
logger = logging.getLogger('modelingestor')

@require_http_methods(["GET"])
def get_v01_model_predictions(request):
    """
    API view to fetch model predictions for a specific ticker and timeframe.

    This endpoint returns model predictions based on the specified parameters.

    Args:
        request (HttpRequest): The HTTP request object.
                            Should include the following query parameters:
                            - spy_price: Current SPY price
                            - vix_price: Current VIX index price
                            - us10y_rate: Current 10-year US Treasury yield rate

    Returns:
        JsonResponse: JSON response containing model predictions.
                    On success, returns status code 200 with prediction data.
                    On error, returns appropriate error status code and message.
    """
    
    modeling_api = ModelingApiConnector()
    spy_price = request.GET.get('spy_price', None)
    vix_price = request.GET.get('vix_price', None)
    us10y_rate = request.GET.get('us10y_rate', None)
    buying_power = request.GET.get('buying_power', None)
    
    if not spy_price or not vix_price or not us10y_rate:
        logger.error("Missing required parameters: spy_price=%s, vix_price=%s, us10y_rate=%s", spy_price, vix_price, us10y_rate)
        return JsonResponse({'error': 'spy_price, vix_price, and us10y_rate are required'}, status=400)
    
    try:
        spy_price = float(spy_price)
        vix_price = float(vix_price)
        us10y_rate = float(us10y_rate)
        if buying_power:
            buying_power = float(buying_power)
            
        results = modeling_api.get_v01_model_predictions(
            spy_price=spy_price,
            vix_price=vix_price,
            us10y_rate=us10y_rate,
            buying_power=buying_power
        )
        
        print(f"Modeling API response: {results}")
        
        return JsonResponse({
            'success': results['success'],
            'prediction': results['prediction'],
            'prediction_class': results['prediction_class'],
            'probabilities': results['probabilities'],
            'option_trade': results.get('option_trade', None),
            'message': results.get('message', '')
        })
    except ValueError:
        logger.error("Invalid input values: spy_price=%s, vix_price=%s, us10y_rate=%s", spy_price, vix_price, us10y_rate)
        return JsonResponse({'error': 'Invalid input values'}, status=400)