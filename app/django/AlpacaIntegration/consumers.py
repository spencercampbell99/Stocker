import json
import os
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from urllib.parse import parse_qs
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import REST
from stocks.models import DailyCandle, FiveMinCandle
from django.conf import settings

logger = logging.getLogger(__name__)

class AlpacaStreamConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for streaming real-time stock data from Alpaca."""    
    
    async def connect(self):
        """Handle WebSocket connection."""
        logger.info("WebSocket connection attempt started")
        
        try:
            # Get ticker from URL path parameters
            self.ticker = self.scope['url_route']['kwargs'].get('ticker')
            
            # If not in path params, try to get from query string
            if not self.ticker:
                logger.debug("No ticker in URL path, checking query parameters")
                query_string = self.scope['query_string'].decode()
                
                if query_string:
                    try:
                        query_params = parse_qs(query_string)
                        self.ticker = query_params.get('ticker', [''])[0]
                        logger.debug(f"Found ticker in query parameters: {self.ticker}")
                    except Exception as e:
                        logger.error(f"Error parsing query string: {e}")
            
            if not self.ticker:
                logger.error("No ticker specified in WebSocket connection")
                await self.close(code=4000)
                return
                
            # Normalize ticker symbol
            self.ticker = self.ticker.upper()
            logger.info(f"WebSocket connection for ticker: {self.ticker}")
            
            # Create the group name for this ticker
            self.ticker_group_name = f'alpaca_stream_{self.ticker}'
            
            # Initialize Alpaca clients
            self.alpaca_stream = None
            self.is_connected = False
            
            # Accept the connection
            await self.accept()
            logger.info(f"WebSocket connection accepted for {self.ticker}")
            
            # Initialize variables for data tracking
            self.last_quote = None
            
            # Get historical data for moving average calculations
            await self.send_historical_data()
            
            # Start the Alpaca stream connection
            await self.connect_to_alpaca()
            
        except Exception as e:
            logger.exception(f"Error during WebSocket connection: {e}")
            # Try to send error to client if connection was already accepted
            try:
                if hasattr(self, 'accept') and self.accept:
                    await self.send_error(f"Connection error: {str(e)}")
            except:
                pass
            # Close the connection with an error code
            await self.close(code=4500)
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        logger.info(f"WebSocket disconnection with code: {close_code}")
        # Close Alpaca connection if active
        await self.close_alpaca_connection()
    
    async def receive(self, text_data):
        """Handle incoming messages from WebSocket."""
        try:
            logger.debug(f"Received WebSocket message: {text_data[:100]}...")  # Log first 100 chars
            data = json.loads(text_data)
            command = data.get('command')
            
            if command == 'subscribe':
                # Handle subscription to additional tickers
                symbols = data.get('symbols', [])
                if symbols and self.alpaca_stream:
                    logger.info(f"Subscribing to additional symbols: {symbols}")
                    await self.subscribe_to_tickers(symbols)
            elif command == 'unsubscribe':
                # Handle unsubscription from tickers
                symbols = data.get('symbols', [])
                if symbols and self.alpaca_stream:
                    logger.info(f"Unsubscribing from symbols: {symbols}")
                    await self.unsubscribe_from_tickers(symbols)
            elif command == 'ping':
                # Simple ping-pong to keep connection alive
                logger.debug("Received ping, sending pong")
                await self.send(text_data=json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}))
            else:
                logger.warning(f"Unknown command received: {command}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            await self.send_error(f"Invalid JSON: {str(e)}")
        except Exception as e:
            logger.exception(f"Error processing WebSocket message: {e}")
            await self.send_error(f"Failed to process command: {str(e)}")
    
    async def connect_to_alpaca(self):
        """Connect to Alpaca WebSocket API."""        
        try:
            # Get Alpaca API credentials from environment
            api_key = settings.ALPACA_API_KEY
            api_secret = settings.ALPACA_API_SECRET
            
            if not api_key or not api_secret:
                logger.error("Alpaca API credentials not configured")
                await self.send_error("Alpaca API credentials not configured")
                return
            
            logger.info(f"Initializing Alpaca stream client for {self.ticker}")
            
            # Initialize the alpaca stream connection
            self.alpaca_stream = Stream(
                key_id=api_key,
                secret_key=api_secret,
                base_url='https://paper-api.alpaca.markets',
                data_feed='iex',  # Use 'sip' for production with proper subscriptions
                raw_data=False
            )
            
            # Define callback handlers for quotes
            async def quote_handler(data):
                try:
                    logger.debug(f"Received quote update for {self.ticker}")
                    
                    # Make sure we're sending data in the format expected by the frontend
                    # Extract the required data using proper error handling
                    try:
                        bid_price = float(data.bid_price)
                        bid_size = int(data.bid_size)
                        ask_price = float(data.ask_price)
                        ask_size = int(data.ask_size)
                        timestamp = data.timestamp
                    except (AttributeError, ValueError) as e:
                        logger.error(f"Error extracting quote data: {e}")
                        return
                    
                    # Prepare quote data for frontend using the exact keys expected in the JS code
                    quote_data = {
                        'type': 'quote',
                        'bidPrice': bid_price,
                        'bidSize': bid_size,
                        'askPrice': ask_price,
                        'askSize': ask_size,
                        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                    }
                    
                    # Send to WebSocket client
                    await self.send(text_data=json.dumps(quote_data))
                    logger.debug(f"Sent quote data: bid={bid_price}, ask={ask_price}")
                    
                except Exception as e:
                    logger.exception(f"Error handling quote update: {e}")
            
            # Register the handlers with the Alpaca stream
            self.alpaca_stream.subscribe_quotes(quote_handler, self.ticker)

            def run_stream():
                try:
                    self.alpaca_stream._run_forever()
                except Exception as e:
                    logger.exception(f"Error in Alpaca stream: {e}")
            
            # Start the stream in a separate thread
            import threading
            self.stream_thread = threading.Thread(target=run_stream, daemon=True)
            self.stream_thread.start()
            
            self.is_connected = True
            logger.info(f"Connected to Alpaca Stream for {self.ticker}")
            
            # Notify client that we're connected
            await self.send(text_data=json.dumps({
                'type': 'connection_status',
                'status': 'connected',
                'message': f'Successfully connected to Alpaca stream for {self.ticker}'
            }))
            
        except Exception as e:
            logger.exception(f"Failed to connect to Alpaca: {e}")
            await self.send_error(f"Failed to connect to Alpaca: {str(e)}")
    
    async def close_alpaca_connection(self):
        """Close connection to Alpaca stream."""
        if self.alpaca_stream and self.is_connected:
            try:
                logger.info(f"Closing Alpaca stream for {self.ticker}")
                await self.alpaca_stream.stop_ws()
                self.is_connected = False
                logger.info(f"Closed Alpaca stream for {self.ticker}")
            except Exception as e:
                logger.exception(f"Error closing Alpaca stream: {e}")
    
    async def send_error(self, message):
        """Send error message to client."""
        logger.error(f"Sending error to client: {message}")
        try:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': message,
                'timestamp': datetime.now().isoformat()
            }))
        except Exception as e:
            logger.exception(f"Error sending error message to client: {e}")
    
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
        Get historical data from Alpaca API if not available in local database.
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
            
            # If data not in database, fetch from Alpaca API
            if daily_df is None or fivemin_df is None:
                logger.info(f"Some historical data not found in database, fetching from Alpaca API")
                # Create API client
                api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')
                
                end = datetime.now()
                start_daily = end - timedelta(days=60)
                start_fivemin = end - timedelta(days=3)
                
                # Get daily data
                if daily_df is None:
                    logger.info(f"Fetching daily data from Alpaca for {self.ticker}")
                    daily_bars = api.get_barset(
                        symbols=[self.ticker],
                        timeframe='1D',
                        start=start_daily.isoformat(),
                        end=end.isoformat()
                    )
                    if self.ticker in daily_bars:
                        daily_df = daily_bars[self.ticker].df
                        logger.info(f"Received {len(daily_df)} daily bars from Alpaca")
                    else:
                        logger.warning(f"No daily data returned from Alpaca for {self.ticker}")
                
                # Get 5-minute data
                if fivemin_df is None:
                    logger.info(f"Fetching 5-minute data from Alpaca for {self.ticker}")
                    fivemin_bars = api.get_barset(
                        symbols=[self.ticker],
                        timeframe='5Min',
                        start=start_fivemin.isoformat(),
                        end=end.isoformat()
                    )
                    if self.ticker in fivemin_bars:
                        fivemin_df = fivemin_bars[self.ticker].df
                        logger.info(f"Received {len(fivemin_df)} 5-minute bars from Alpaca")
                    else:
                        logger.warning(f"No 5-minute data returned from Alpaca for {self.ticker}")
            
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
    
    async def calculate_and_send_moving_averages(self):
        """Calculate and send updated moving averages periodically."""        
        # This is a placeholder - in a real implementation we would keep track of
        # received trades and recalculate moving averages only when appropriate
        # For now, we'll rely on the initial calculations from historical data
        pass
    
    async def subscribe_to_tickers(self, symbols):
        """Subscribe to additional ticker symbols."""        
        if not self.alpaca_stream:
            logger.error("Cannot subscribe: Alpaca stream not initialized")
            await self.send_error("Alpaca stream not initialized")
            return
        
        try:
            async def quote_callback(quote):
                try:
                    symbol = quote.symbol
                    logger.debug(f"Received quote update for {symbol}")
                    
                    # Prepare quote data for frontend
                    quote_data = {
                        'type': 'quote',
                        'symbol': symbol,
                        'bidPrice': float(quote.bid_price),
                        'bidSize': int(quote.bid_size),
                        'askPrice': float(quote.ask_price),
                        'askSize': int(quote.ask_size),
                        'timestamp': quote.timestamp.isoformat()
                    }
                    
                    # Send to WebSocket client
                    await self.send(text_data=json.dumps(quote_data))
                except Exception as e:
                    logger.exception(f"Error handling quote update: {e}")
                    
            # Subscribe to trades and quotes for additional symbols
            for symbol in symbols:
                symbol = symbol.upper()
                logger.info(f"Subscribing to {symbol}")
                self.alpaca_stream.subscribe_quotes(quote_callback, symbol)
            
            logger.info(f"Subscribed to additional symbols: {symbols}")
            await self.send(text_data=json.dumps({
                'type': 'subscription_update',
                'message': f"Successfully subscribed to: {', '.join(symbols)}",
                'symbols': symbols
            }))
        except Exception as e:
            logger.exception(f"Failed to subscribe to symbols: {e}")
            await self.send_error(f"Failed to subscribe: {str(e)}")
    
    async def unsubscribe_from_tickers(self, symbols):
        """Unsubscribe from ticker symbols."""        
        if not self.alpaca_stream:
            logger.error("Cannot unsubscribe: Alpaca stream not initialized")
            await self.send_error("Alpaca stream not initialized")
            return
        
        try:
            # Unsubscribe from trades and quotes
            for symbol in symbols:
                symbol = symbol.upper()
                logger.info(f"Unsubscribing from {symbol}")
                try:
                    self.alpaca_stream.unsubscribe_quotes(symbol)
                except Exception as e:
                    logger.warning(f"Error unsubscribing from {symbol}: {e}")
            
            logger.info(f"Unsubscribed from symbols: {symbols}")
            await self.send(text_data=json.dumps({
                'type': 'subscription_update',
                'message': f"Successfully unsubscribed from: {', '.join(symbols)}",
                'symbols': symbols
            }))
        except Exception as e:
            logger.exception(f"Failed to unsubscribe from symbols: {e}")
            await self.send_error(f"Failed to unsubscribe: {str(e)}")