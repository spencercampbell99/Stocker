import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from contextlib import contextmanager

from db.database import Base, get_engine, get_session

class PostgresConnector:
    """
    A class to handle PostgreSQL database connections using SQLAlchemy ORM.
    """
    
    def __init__(self, database=None):
        """
        Initialize the connector by loading environment variables.
        
        Args:
            database (str, optional): Database name to connect to. If None, uses the value
                                      from POSTGRES_DATABASE in .env file.
        """
        load_dotenv()
        
        # Get credentials from environment variables
        self.user = os.getenv('POSTGRES_USER')
        self.password = os.getenv('POSTGRES_ROOT_PASSWORD')
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('POSTGRES_LOCAL_PORT')
        self.database = database if database else os.getenv('POSTGRES_DATABASE')
        
        self.engine = None
        self._session = None
    
    def connect(self):
        """Establish a connection to the PostgreSQL database using SQLAlchemy."""
        connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        try:
            self.engine = create_engine(connection_string)
            Session = sessionmaker(bind=self.engine)
            self._session = Session()
            print("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close the database connection."""
        if self._session:
            self._session.close()
            self._session = None
            print("Database session closed")
        if self.engine:
            self.engine.dispose()
            print("Database engine disposed")
    
    @property
    def session(self):
        """Get the current SQLAlchemy session, creating one if needed."""
        if not self._session:
            self.connect()
        return self._session
    
    def execute_query(self, query, params=None):
        """
        Execute a raw SQL query.
        
        Args:
            query (str): SQL query to execute
            params (dict, optional): Parameters for the query
            
        Returns:
            Result of the query execution
        """
        try:
            if params:
                result = self.session.execute(text(query), params)
            else:
                result = self.session.execute(text(query))
            self.session.commit()
            return result
        except Exception as e:
            self.session.rollback()
            print(f"Error executing query: {e}")
            return None
    
    def query_to_dataframe(self, query, params=None):
        """
        Execute a query and return the results as a pandas DataFrame.
        
        Args:
            query (str): SQL query to execute
            params (dict, optional): Parameters for the query
            
        Returns:
            pandas.DataFrame: Query results as a DataFrame
        """
        try:
            if params:
                df = pd.read_sql_query(text(query), self.engine, params=params)
            else:
                df = pd.read_sql_query(text(query), self.engine)
            return df
        except Exception as e:
            print(f"Error executing query to DataFrame: {e}")
            return None
    
    def create_tables(self):
        """Create all tables defined in SQLAlchemy models."""
        Base.metadata.create_all(self.engine)
        print("Database tables created")
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.session
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
    
    def __enter__(self):
        """Context manager entry point."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.disconnect()