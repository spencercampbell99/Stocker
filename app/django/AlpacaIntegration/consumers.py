import json
import logging
import asyncio
import websockets
import uuid
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings
from stocks.models import DailyCandle, FiveMinCandle

logger = logging.getLogger(__name__)

class AlpacaStreamConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for streaming real-time stock data directly from Alpaca."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpaca_ws = None
        self.ticker = None
        self.connection_id = str(uuid.uuid4())
        self.keepalive_task = None
        self.yfinance_task = None
        # Update to use the configured data feed from settings
        self.ALPACA_WS_URL = f"wss://stream.data.alpaca.markets/v2/{settings.ALPACA_DATA_FEED}"
        
        # Market indices from yfinance
        self.market_indices = {"^TNX": None, "^VIX": None}  # Store the latest quotes for US10Y and VIX
        self.indices_update_interval = 5 * 60  # Update market indices every 5 minutes (in seconds)
        
        # For throttling quotes
        self.last_quote_time = datetime.now() - timedelta(seconds=10)  # Initialize to send first quote immediately
        self.quote_throttle_seconds = 10  # Send quotes every 10 seconds
        self.latest_quotes = {}  # Store latest quotes between throttle periods
    
    async def connect(self):
        """Handle WebSocket connection."""        
        logger.info(f"WebSocket connection attempt started (ID: {self.connection_id})")
        
        try:
            await self.accept()
            
            API_KEY = settings.ALPACA_API_KEY
            API_SECRET = settings.ALPACA_API_SECRET
            
            if not API_KEY or not API_SECRET:
                logger.error("Alpaca API credentials not configured")
                await self.send_error("Alpaca API credentials not configured")
                await self.close(code=4001)
                return

            query_string = self.scope.get('query_string', b'').decode()
            query_params = dict(param.split('=') for param in query_string.split('&') if param)
            self.ticker = query_params.get('ticker')
            
            if not self.ticker:
                logger.error("Ticker not provided in WebSocket query parameters")
                await self.send_error("Ticker not provided")
                await self.close(code=4002)
                return
            
            # Connect to Alpaca WebSocket
            self.alpaca_ws = await websockets.connect(
                self.ALPACA_WS_URL,
                extra_headers={
                    "APCA-API-KEY-ID": API_KEY,
                    "APCA-API-SECRET-KEY": API_SECRET
                }
            )
            
            # Send authentication message
            auth_msg = {
                "action": "auth",
                "key": API_KEY,
                "secret": API_SECRET
            }
            await self.alpaca_ws.send(json.dumps(auth_msg))
            
            # Wait for auth response
            auth_response = await self.alpaca_ws.recv()
            
            # Subscribe to quotes for our ticker only (not indices)
            subscribe_msg = {
                "action": "subscribe",
                "quotes": [self.ticker]
            }
            await self.alpaca_ws.send(json.dumps(subscribe_msg))
            
            # Start background tasks
            self.keepalive_task = asyncio.create_task(self.keepalive())
            asyncio.create_task(self.stream_data())
            
            # Start yfinance market indices fetching task
            self.yfinance_task = asyncio.create_task(self.fetch_market_indices_periodically())
            
            # Send historical data and fetch initial indices data
            asyncio.create_task(self.send_historical_data())
            # Fetch market indices data immediately
            asyncio.create_task(self.fetch_market_indices())
            
        except Exception as e:
            logger.exception(f"Error during WebSocket connection: {e}")
            await self.send_error(f"Connection error: {str(e)}")
            await self.close_alpaca_connection()
            await self.close(code=4500)
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""        
        logger.info(f"WebSocket disconnection (ID: {self.connection_id}) with code: {close_code}")
        
        # Cancel the yfinance task
        if self.yfinance_task:
            self.yfinance_task.cancel()
            try:
                await self.yfinance_task
            except asyncio.CancelledError:
                pass
        
        # Close Alpaca connection
        await self.close_alpaca_connection()
    
    async def receive(self, text_data):
        """Handle incoming messages from WebSocket."""        
        try:
            logger.debug(f"Received WebSocket message (ID: {self.connection_id}): {text_data[:100]}...")
            data = json.loads(text_data)
            command = data.get('command')
            
            if command == 'ping':
                logger.debug("Received ping, sending pong")
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
            elif command == 'update_indices':
                logger.info(f"Manual update of market indices requested (ID: {self.connection_id})")
                # Trigger immediate fetch of market indices
                await self.fetch_market_indices()
            else:
                logger.warning(f"Unknown command received (ID: {self.connection_id}): {command}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received (ID: {self.connection_id}): {e}")
            await self.send_error(f"Invalid JSON: {str(e)}")
        except Exception as e:
            logger.exception(f"Error processing WebSocket message (ID: {self.connection_id}): {e}")
            await self.send_error(f"Failed to process command: {str(e)}")

    async def stream_data(self):
        """Stream data from Alpaca WebSocket to client."""        
        try:
            async for message in self.alpaca_ws:
                try:
                    data = json.loads(message)
                    # Alpaca can send data as a list or as a dictionary
                    if isinstance(data, list):
                        for item in data:
                            await self.process_alpaca_message(item)
                    else:
                        await self.process_alpaca_message(data)
                except Exception as e:
                    logger.error(f"Error processing Alpaca message: {e}")
                    await self.send_error(f"Error processing market data: {str(e)}")
        except Exception as e:
            logger.error(f"Alpaca WebSocket stream error: {e}")
            await self.send_error(f"Market data stream disconnected: {str(e)}")
            await self.close_alpaca_connection()

    async def process_alpaca_message(self, data):
        """Process individual messages from Alpaca."""        
        message_type = data.get('T')
        ticker = data.get('S')
        
        # Handle different message types
        if message_type == 'error':
            # This is an error response
            logger.error(f"Alpaca error: {data}")
            
        elif message_type == 'q':  # Quote message
            # Always store the latest quotes
            if ticker in self.market_indices:
                self.market_indices[ticker] = {
                    'ticker': ticker,
                    'price': (float(data.get('bp', 0)) + float(data.get('ap', 0))) / 2,
                    'bidPrice': float(data.get('bp', 0)),
                    'askPrice': float(data.get('ap', 0)),
                    'timestamp': data.get('t')
                }
            elif ticker == self.ticker:
                # Store the latest quote for the requested ticker
                self.latest_quotes[ticker] = {
                    'type': 'quote',
                    'ticker': ticker,
                    'bidPrice': float(data.get('bp', 0)),
                    'askPrice': float(data.get('ap', 0)),
                    'bidSize': int(data.get('bs', 0)),
                    'askSize': int(data.get('as', 0)),
                    'timestamp': data.get('t')
                }
            
            # Only send updates every 10 seconds
            now = datetime.now()
            if (now - self.last_quote_time).total_seconds() >= self.quote_throttle_seconds:
                self.last_quote_time = now
                
                # Send any stored market indices data
                if self.market_indices["^TNX"] and self.market_indices["^VIX"]:
                    await self.send_market_indices()
                
                # Send the latest quote for the main ticker
                if self.ticker in self.latest_quotes:
                    await self.send(text_data=json.dumps(self.latest_quotes[self.ticker]))
                    
        elif message_type == 't':  # Trade message
            pass  # We don't need to do anything with trades right now
            
        else:
            # Only log unknown message types
            if message_type not in ['success', 'subscription']:
                logger.warning(f"Unknown message type: {message_type}")

    async def send_market_indices(self):
        """Send the current market indices data to the client."""        
        try:
            # Only send if we have data for both indices
            if self.market_indices["^TNX"] and self.market_indices["^VIX"]:
                await self.send(text_data=json.dumps({
                    'type': 'market_indices',
                    'us10y': {
                        'ticker': '^TNX',
                        'price': self.market_indices["^TNX"]["price"],
                        'timestamp': self.market_indices["^TNX"]["timestamp"]
                    },
                    'vix': {
                        'ticker': '^VIX',
                        'price': self.market_indices["^VIX"]["price"],
                        'timestamp': self.market_indices["^VIX"]["timestamp"]
                    }
                }))
        except Exception as e:
            logger.error(f"Error sending market indices data: {e}")
            # Don't send error to client as this is not critical

    async def keepalive(self):
        """Send periodic keepalive messages to Alpaca."""        
        try:
            while True:
                # Use the same 10-second interval as the quote throttling
                await asyncio.sleep(self.quote_throttle_seconds)
                if self.alpaca_ws and self.alpaca_ws.open:
                    await self.alpaca_ws.send(json.dumps({"action": "ping"}))
        except Exception as e:
            logger.error(f"Keepalive task error: {e}")
            await self.close_alpaca_connection()

    async def send_error(self, message):
        """Send error message to client."""        
        logger.error(f"Sending error to client (ID: {self.connection_id}): {message}")
        try:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': message,
                'timestamp': datetime.now().isoformat()
            }))
        except Exception as e:
            logger.exception(f"Error sending error message to client (ID: {self.connection_id}): {e}")
    
    async def close_alpaca_connection(self):
        """Close the Alpaca WebSocket connection."""        
        if self.alpaca_ws:
            try:
                # Unsubscribe from only the ticker (not indices since they come from yfinance)
                unsubscribe_msg = {
                    "action": "unsubscribe",
                    "quotes": [self.ticker]
                }
                await self.alpaca_ws.send(json.dumps(unsubscribe_msg))
                
                # Cancel keepalive task
                if self.keepalive_task:
                    self.keepalive_task.cancel()
                    try:
                        await self.keepalive_task
                    except asyncio.CancelledError:
                        pass
                
                # Close WebSocket
                await self.alpaca_ws.close()
                logger.info(f"Alpaca connection closed (ID: {self.connection_id})")
            except Exception as e:
                logger.exception(f"Error closing Alpaca connection (ID: {self.connection_id}): {e}")
            finally:
                self.alpaca_ws = None
                self.keepalive_task = None
    
    @database_sync_to_async
    def get_historical_candles(self, timeframe, period, limit=1000):
        """Get historical candle data from database for moving average calculation."""        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period)
            
            if timeframe == "5min":
                candles = FiveMinCandle.objects.filter(
                    ticker=self.ticker,
                    timestamp__gte=start_date,
                    timestamp__lte=end_date
                ).order_by('-timestamp').values('timestamp', 'close')[:limit]
                logger.debug(f"Retrieved {len(candles)} 5-minute candles for {self.ticker}")
            else:  # daily
                candles = DailyCandle.objects.filter(
                    ticker=self.ticker,
                    timestamp__gte=start_date,
                    timestamp__lte=end_date
                ).order_by('-timestamp').values('timestamp', 'close')[:limit]
                logger.debug(f"Retrieved {len(candles)} daily candles for {self.ticker}")
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(list(candles))
            
            # invert the sorting order
            df = df[::-1].reset_index(drop=True)
            
            return df if not df.empty else None
        except Exception as e:
            logger.exception(f"Error getting historical candles: {e}")
            return None
    
    async def send_historical_data(self):
        """
        Get historical data from our local database.
        Calculate initial moving averages and send to client.
        """
        try:
            logger.info(f"Retrieving historical data for {self.ticker}")
            # Get Alpaca API credentials
            api_key = settings.ALPACA_API_KEY
            api_secret = settings.ALPACA_API_SECRET
            
            if not api_key or not api_secret:
                logger.error("Alpaca API credentials not configured")
                await self.send_error("Alpaca API credentials not configured")
                return
            
            # Get data from our database first
            daily_df = await self.get_historical_candles("daily", 60, limit=20)
            fivemin_df = await self.get_historical_candles("5min", 4, limit=20)
            
            # Calculate moving averages
            logger.info(f"Calculating moving averages for {self.ticker}")
            ma_data = {
                'type': 'ma_data'
            }
            
            # Calculate 5-minute moving averages
            if fivemin_df is not None and len(fivemin_df) > 0:
                fivemin_df['ma9'] = fivemin_df['close'].rolling(window=9).mean()
                fivemin_df['ma20'] = fivemin_df['close'].rolling(window=20).mean()
                
                # Get latest values
                latest = fivemin_df.iloc[-1]
                ma_data['ma5min9'] = float(latest['ma9']) if not np.isnan(latest['ma9']) else None
                ma_data['ma5min20'] = float(latest['ma20']) if not np.isnan(latest['ma20']) else None
                logger.info(f"5-minute MAs calculated: 9MA={ma_data.get('ma5min9')}, 20MA={ma_data.get('ma5min20')}")
            else:
                logger.warning(f"Could not calculate 5-minute MAs for {self.ticker} due to insufficient data")
            
            # Calculate daily moving averages
            if daily_df is not None and len(daily_df) > 0:
                daily_df['ma9'] = daily_df['close'].rolling(window=9).mean()
                daily_df['ma20'] = daily_df['close'].rolling(window=20).mean()
                
                # Get latest values
                latest = daily_df.iloc[-1]
                ma_data['maDaily9'] = float(latest['ma9']) if not np.isnan(latest['ma9']) else None
                ma_data['maDaily20'] = float(latest['ma20']) if not np.isnan(latest['ma20']) else None
                logger.info(f"Daily MAs calculated: 9MA={ma_data.get('maDaily9')}, 20MA={ma_data.get('maDaily20')}")
            else:
                logger.warning(f"Could not calculate daily MAs for {self.ticker} due to insufficient data")
            
            # Send moving averages to client
            await self.send(text_data=json.dumps(ma_data))
            logger.info(f"Moving average data sent to client for {self.ticker}")
            
        except Exception as e:
            logger.exception(f"Error fetching historical data: {e}")
            await self.send_error(f"Error fetching historical data: {str(e)}")
    
    async def fetch_market_indices_periodically(self):
        """Fetch market indices data from yfinance periodically."""
        try:
            while True:
                await self.fetch_market_indices()
                # Wait for the specified interval before fetching again
                await asyncio.sleep(self.indices_update_interval)
        except asyncio.CancelledError:
            logger.info("Market indices fetch task cancelled")
        except Exception as e:
            logger.error(f"Error in market indices periodic fetch: {e}")
            
    async def fetch_market_indices(self):
        """Fetch market indices data from yfinance."""
        try:
            # Run yfinance operations in a thread pool since they are blocking
            indices_data = await asyncio.to_thread(self._fetch_yfinance_data, ["^TNX", "^VIX"])
            
            if indices_data:
                # Store the fetched data
                for symbol, data in indices_data.items():
                    if data is not None:
                        self.market_indices[symbol] = {
                            'ticker': symbol,
                            'price': data['price'],
                            'timestamp': data['timestamp'].isoformat()
                        }
                
                # Send updated indices to client
                await self.send_market_indices()
                logger.debug(f"Updated market indices from yfinance: TNX={self.market_indices.get('^TNX', {}).get('price')}, VIX={self.market_indices.get('^VIX', {}).get('price')}")
        except Exception as e:
            logger.error(f"Error fetching market indices data: {e}")

    def _fetch_yfinance_data(self, symbols):
        """Fetch data from yfinance (runs in a separate thread)."""
        result = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get the latest price data
                history = ticker.history(period="1d")
                
                if not history.empty:
                    latest = history.iloc[-1]
                    result[symbol] = {
                        'price': float(latest['Close']),
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return result