import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class PostgresConnector:
    """
    A class to handle PostgreSQL database connections using credentials from .env file.
    """
    
    def __init__(self):
        """Initialize the connector by loading environment variables."""
        load_dotenv()
        
        # Get credentials from environment variables
        self.user = os.getenv('POSTGRES_USER')
        self.password = os.getenv('POSTGRES_ROOT_PASSWORD')
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('POSTGRES_LOCAL_PORT')
        self.database = os.getenv('POSTGRES_DATABASE')
        
        self.engine = None
        self.connection = None
    
    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        try:
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            print("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed")
        if self.engine:
            self.engine.dispose()
    
    def execute_query(self, query):
        """
        Execute a raw SQL query.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            Result of the query execution
        """
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            result = self.connection.execute(text(query))
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def query_to_dataframe(self, query):
        """
        Execute a query and return the results as a pandas DataFrame.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pandas.DataFrame: Query results as a DataFrame
        """
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Error executing query to DataFrame: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry point."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.disconnect()