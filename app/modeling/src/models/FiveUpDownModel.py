import tensorflow as tf
from src.db.database import StockSymbol, init_db, get_session, get_engine, DailyCandle, HourlyCandle, FiveMinuteCandle, ThirtyMinuteCandle
import pandas as pd
from sqlalchemy import text

# Gather data
db = get_engine()
session = get_session(db)

# Get daily data back to 2020-01-01
query = session.query(DailyCandle).filter(DailyCandle.symbol == "SPY", DailyCandle.timestamp >= "2020-01-01").order_by(DailyCandle.date.asc())
data = pd.read_sql(query.statement, db)
data = data.set_index("timestamp")
data = data.drop(columns=["id", "symbol"])

# Get 5 minute data for premarket (7am-9:30am) back to 2020-01-01
query = text(f"""
    SELECT
        *
    FROM stocks_fivemincandle
    WHERE ticker = 'SPY'
        AND "timestamp" >= '2020-01-01'
        AND ("timestamp"::time BETWEEN '07:00:00' AND '09:30:00');
""")
data_5min = pd.read_sql(query, db)
data_5min = data_5min.set_index("timestamp")
data_5min = data_5min.drop(columns=["id", "symbol"])

print(data_5min.head())