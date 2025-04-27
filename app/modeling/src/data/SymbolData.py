"""
Module for handling stock symbol data using Alpaca API and SQLAlchemy ORM.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus

from db.PostgresConnector import PostgresConnector
from data.AlpacaConnector import AlpacaConnector
from db.database import StockSymbol, init_db, get_session


class SymbolDataManager:
    """
    Class to manage stock symbol data operations using Alpaca API.
    Handles fetching, processing, and storing symbol information.
    """
    
    def __init__(self):
        """Initialize the symbol data manager with API connections."""
        # Use the centralized Alpaca connector
        self.alpaca = AlpacaConnector()
        
        # Initialize database
        self.db = PostgresConnector()
        
    def ensure_database_schema(self):
        """Ensure that the database tables exist."""
        try:
            # Use SQLAlchemy's metadata to create tables if they don't exist
            init_db()
            return True
        except Exception as e:
            print(f"Error ensuring database schema: {e}")
            return False
    
    def get_symbol_info(self, ticker):
        """
        Get information for a specific ticker symbol from Alpaca API.
        
        Args:
            ticker (str): The stock ticker symbol to query
            
        Returns:
            dict: Dictionary containing symbol information or None if not found
        """
        try:
            # Get asset information
            search_params = GetAssetsRequest(status=AssetStatus.ACTIVE)
            assets = self.alpaca.trading_client.get_all_assets(search_params)
            
            # Find the specific ticker
            asset_info = next((asset for asset in assets if asset.symbol.upper() == ticker.upper()), None)
            
            if asset_info:
                # Get sector safely (this may need to be fetched from a different source as Alpaca may not provide it)
                sector = "Unknown"  # Default sector
                
                # Create a dictionary with the asset information conforming to our database schema
                symbol_info = {
                    'symbol': asset_info.symbol,
                    'name': asset_info.name,
                    'sector': sector,
                    'is_active': asset_info.status == AssetStatus.ACTIVE,
                }
                return symbol_info
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching symbol information for {ticker}: {e}")
            return None
    
    def insert_symbol(self, ticker):
        """
        Insert or update a stock symbol in the database using SQLAlchemy ORM.
        
        Args:
            ticker (str): The stock ticker symbol to insert
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get symbol information from Alpaca
        symbol_info = self.get_symbol_info(ticker)
        
        if not symbol_info:
            return False
        
        try:
            # Use a session for this transaction
            session = get_session()
            
            # Check if symbol already exists
            existing_symbol = session.query(StockSymbol).filter_by(symbol=symbol_info['symbol']).first()
            
            current_time = datetime.now()
            
            if existing_symbol:
                # Update existing record
                existing_symbol.name = symbol_info['name']
                existing_symbol.sector = symbol_info['sector']
                existing_symbol.is_active = symbol_info['is_active']
                existing_symbol.updated_at = current_time
            else:
                # Create new record
                new_symbol = StockSymbol(
                    symbol=symbol_info['symbol'],
                    name=symbol_info['name'],
                    sector=symbol_info['sector'],
                    is_active=symbol_info['is_active'],
                    created_at=current_time,
                    updated_at=current_time
                )
                session.add(new_symbol)
            
            # Commit the transaction
            session.commit()
            print(f"Successfully inserted/updated symbol {ticker}")
            return True
            
        except SQLAlchemyError as e:
            if session:
                session.rollback()
            print(f"Database error inserting symbol {ticker}: {e}")
            return False
        except Exception as e:
            if session:
                session.rollback()
            print(f"Error inserting symbol {ticker}: {e}")
            return False
        finally:
            if session:
                session.close()
    
    def load_multiple_symbols(self, tickers):
        """
        Load multiple ticker symbols into the database.
        
        Args:
            tickers (list): List of ticker symbols to load
            
        Returns:
            dict: Dictionary with success count, total, and failed symbols
        """
        self.ensure_database_schema()
        
        success_count = 0
        failed_symbols = []
        
        for ticker in tickers:
            try:
                if self.insert_symbol(ticker):
                    success_count += 1
                else:
                    failed_symbols.append(ticker)
            except Exception as e:
                print(f"Unexpected error processing {ticker}: {e}")
                failed_symbols.append(ticker)
        
        return {
            'success_count': success_count,
            'total': len(tickers),
            'failed_symbols': failed_symbols
        }
        
    def get_all_symbols(self):
        """
        Retrieve all stock symbols from the database.
        
        Returns:
            list: List of StockSymbol objects
        """
        try:
            session = get_session()
            symbols = session.query(StockSymbol).all()
            return symbols
        except Exception as e:
            print(f"Error retrieving symbols: {e}")
            return []
        finally:
            if session:
                session.close()