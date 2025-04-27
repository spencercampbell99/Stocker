#!/usr/bin/env python
"""
Test script to verify PostgresConnector functionality.
This script will:
1. Establish a connection to the PostgreSQL database
2. Run a simple test query
3. Display the results
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import PostgresConnector
sys.path.append(str(Path(__file__).parents[2]))

# Import PostgresConnector
from db.PostgresConnector import PostgresConnector

def test_connection():
    """Test basic connection to PostgreSQL database"""
    print("Testing PostgreSQL connection...")
    connector = PostgresConnector()
    
    # Test context manager approach
    print("\nTesting context manager approach:")
    try:
        with connector as db:
            if db.connection:
                print("✓ Connection successful using context manager")
            else:
                print("✗ Connection failed using context manager")
    except Exception as e:
        print(f"✗ Error using context manager: {e}")
    
    # Test manual connection
    print("\nTesting manual connection approach:")
    try:
        success = connector.connect()
        if success:
            print("✓ Manual connection successful")
        else:
            print("✗ Manual connection failed")
        connector.disconnect()
    except Exception as e:
        print(f"✗ Error in manual connection: {e}")

def test_simple_query():
    """Test executing a simple query"""
    print("\nTesting simple query execution...")
    connector = PostgresConnector()
    
    try:
        with connector as db:
            # Try to get database version - a simple query that works on PostgreSQL
            result = db.execute_query("SELECT version() as version")
            if result:
                version = result.fetchone()[0]
                print(f"✓ Query executed successfully - PostgreSQL Version: {version}")
            else:
                print("✗ Query execution failed")
    except Exception as e:
        print(f"✗ Error executing query: {e}")

def test_dataframe_query():
    """Test querying data into a pandas DataFrame"""
    print("\nTesting DataFrame query...")
    connector = PostgresConnector()
    
    try:
        with connector as db:
            # Get a list of schemas as a DataFrame (PostgreSQL equivalent of "SHOW DATABASES")
            query = "SELECT schema_name FROM information_schema.schemata"
            df = db.query_to_dataframe(query)
            
            if df is not None and not df.empty:
                print("✓ DataFrame query successful")
                print("\nAvailable schemas:")
                for idx, schema_name in enumerate(df['schema_name'].tolist(), 1):
                    print(f"  {idx}. {schema_name}")
            else:
                print("✗ DataFrame query failed or returned empty result")
    except Exception as e:
        print(f"✗ Error executing DataFrame query: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("POSTGRESQL CONNECTOR TEST SCRIPT")
    print("=" * 60)
    
    test_connection()
    test_simple_query()
    test_dataframe_query()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()