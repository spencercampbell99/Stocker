"""
Database models using SQLAlchemy ORM

This module defines the SQLAlchemy models representing database tables.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import psycopg2
import psycopg2.extras
import pandas as pd
from io import StringIO
from typing import List, Dict, Any
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Create a base class for our models
Base = declarative_base()


class StockSymbol(Base):
    """SQLAlchemy model for stock symbols table."""
    
    __tablename__ = 'stocks_ticker'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255))
    sector = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<StockSymbol(symbol='{self.symbol}', name='{self.name}')>"

class BaseCandle(Base):
    """Base class for candle data models."""
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=False), nullable=False, index=True)
    open = Column(Integer, nullable=False)
    high = Column(Integer, nullable=False)
    low = Column(Integer, nullable=False)
    close = Column(Integer, nullable=False)
    volume = Column(Integer, nullable=False)
    
class DailyCandle(BaseCandle):
    """SQLAlchemy model for daily candle data."""
    
    __tablename__ = 'stocks_dailycandle'
    
    def __repr__(self):
        return f"<DailyCandle(symbol='{self.symbol}', timestamp='{self.timestamp}')>"

class HourlyCandle(BaseCandle):
    """SQLAlchemy model for hourly candle data."""
    
    __tablename__ = 'stocks_hourcandle'
    
    def __repr__(self):
        return f"<HourlyCandle(symbol='{self.symbol}', timestamp='{self.timestamp}')>"
    
class FiveMinuteCandle(BaseCandle):
    """SQLAlchemy model for 5-minute candle data."""
    
    __tablename__ = 'stocks_fivemincandle'
    
    def __repr__(self):
        return f"<FiveMinuteCandle(symbol='{self.symbol}', timestamp='{self.timestamp}')>"

class ThirtyMinuteCandle(BaseCandle):
    """SQLAlchemy model for 30-minute candle data."""
    
    __tablename__ = 'stocks_thirtymincandle'
    
    def __repr__(self):
        return f"<ThirtyMinuteCandle(symbol='{self.symbol}', timestamp='{self.timestamp}')>"
    
def get_engine():
    """Create and return a SQLAlchemy engine using environment variables."""
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_ROOT_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('POSTGRES_LOCAL_PORT')
    database = os.getenv('POSTGRES_DATABASE')
    
    connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return create_engine(connection_string, echo=False)


def get_session():
    """Create and return a new SQLAlchemy session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """
    Initialize the database connection. 
    Note: Unlike typical SQLAlchemy usage, this doesn't create tables
    as we're assuming Django has already created them.
    """
    engine = get_engine()
    # Note: We're NOT creating tables here since Django manages schema
    # Base.metadata.create_all(engine)  # This line is commented out
    return engine


@contextmanager
def get_session_context():
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()