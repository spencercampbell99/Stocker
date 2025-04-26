"""
Alpaca API Connector

This module provides a central connection point for Alpaca API services.
"""

import os
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient


class AlpacaConnector:
    """
    Singleton class to manage connections to the Alpaca API.
    Provides centralized access to various Alpaca client objects.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(AlpacaConnector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the Alpaca API connector with credentials from .env file."""
        if self._initialized:
            return
            
        # Load environment variables
        load_dotenv()
        
        # Get Alpaca API credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables. "
                           "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file.")
        
        # Initialize client instances as None, they will be created on demand
        self._trading_client = None
        self._data_client = None
        
        self._initialized = True
    
    @property
    def trading_client(self):
        """
        Get or create the Alpaca Trading client.
        
        Returns:
            alpaca.trading.client.TradingClient: Initialized trading client
        """
        if not self._trading_client:
            self._trading_client = TradingClient(self.api_key, self.api_secret)
        return self._trading_client
    
    @property
    def data_client(self):
        """
        Get or create the Alpaca Stock Historical Data client.
        
        Returns:
            alpaca.data.historical.StockHistoricalDataClient: Initialized data client
        """
        if not self._data_client:
            self._data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        return self._data_client