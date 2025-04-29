import os
import json
import requests
from typing import Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    prediction_class: int
    probabilities: Dict[str, float]
    option_trade: Optional[Dict[str, Any]] = None
    message: str = ""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelingApiConnector:
    """
    Connector class for interacting with the modeling FastAPI service.
    Handles authentication and provides methods for common HTTP operations.
    """
    
    def __init__(self):
        """Initialize the connector with configuration from environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Get API credentials from environment
        self.base_url = os.environ.get("FAST_API_BASE_URL")
        self.api_key = os.environ.get("FAST_API_SECRET_KEY")
        
        # Validate required configuration
        if not self.base_url:
            raise ValueError("FAST_API_BASE_URL environment variable is required")
        if not self.api_key:
            raise ValueError("FAST_API_SECRET_KEY environment variable is required")
            
        # Remove trailing slash from base URL if present
        self.base_url = self.base_url.rstrip('/')
        
        # Prepare default headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": self.api_key
        }
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Process API response and handle errors."""
        try:
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}, Response: {response.text}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response: {response.text}")
            raise ValueError("Invalid JSON response received from the API")
            
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send GET request to the API."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"GET request to: {url}")
        
        response = requests.get(url, headers=self.headers, params=params)
        return self._handle_response(response)
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send POST request to the API."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"POST request to: {url}")
        
        response = requests.post(url, headers=self.headers, json=data)
        return self._handle_response(response)
    
    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send PUT request to the API."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"PUT request to: {url}")
        
        response = requests.put(url, headers=self.headers, json=data)
        return self._handle_response(response)
    
    def patch(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send PATCH request to the API."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"PATCH request to: {url}")
        
        response = requests.patch(url, headers=self.headers, json=data)
        return self._handle_response(response)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Send DELETE request to the API."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"DELETE request to: {url}")
        
        response = requests.delete(url, headers=self.headers)
        return self._handle_response(response)

    def get_v01_model_predictions(
            self,
            spy_price: float,
            vix_price: float,
            us10y_rate: float,
            buying_power: float|None = None
        ) -> PredictionResponse:
        """
        Fetch model predictions from the FastAPI service.

        Args:
            params (Dict[str, Any]): Dictionary of parameters to send in the request.

        Returns:
            Dict[str, Any]: Model predictions from the API.
        """
        endpoint = "api/predict"
        params = {
            "spy_price": spy_price,
            "vix_value": vix_price,
            "us10y_value": us10y_rate,
        }
        
        if buying_power is not None:
            params["buying_power"] = buying_power
        
        response = self.post(endpoint, params)
        
        return response