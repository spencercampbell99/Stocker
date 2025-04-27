from sqlalchemy import text

def get_daily_move_status_query(start_date='2018-01-01', up_threshold=1.0075, down_threshold=0.9925, ticker='SPY'):
    """
    Get the daily move status for a given ticker. Returns a query that retrieves the first up and down moves for each day.
    The up and down moves are determined based on the open price at 9:30 AM and the high/low prices during the day.
    """
    return text(
        f"""
            WITH daily_open AS (
                SELECT 
                    "timestamp"::date AS date,
                    open AS open_930
                FROM stocks_fivemincandle
                WHERE ticker = '{ticker}'
                AND "timestamp"::time = '09:30:00'
                AND "timestamp" >= '{start_date}'
            ),
            time_markers AS (
                SELECT
                    f."timestamp"::date AS date,
                    f."timestamp",
                    f.high,
                    f.low,
                    o.open_930,
                    CASE WHEN f.high >= o.open_930 * {up_threshold} THEN f."timestamp" END AS up_timestamp,
                    CASE WHEN f.low <= o.open_930 * {down_threshold} THEN f."timestamp" END AS down_timestamp
                FROM stocks_fivemincandle f
                JOIN daily_open o ON f."timestamp"::date = o.date
                WHERE f.ticker = '{ticker}'
                AND f."timestamp"::time BETWEEN '09:30:00' AND '16:00:00'
            ),
            first_moves AS (
                SELECT
                    date,
                    open_930,
                    MIN(up_timestamp) AS first_up,
                    MIN(down_timestamp) AS first_down
                FROM time_markers
                GROUP BY date, open_930
            )
            SELECT
                date,
                open_930 AS open,
                CASE
                    WHEN first_up IS NOT NULL AND (first_down IS NULL OR first_up < first_down) THEN 2 -- Up move
                    WHEN first_down IS NOT NULL AND (first_up IS NULL OR first_down < first_up) THEN 0 -- Down move
                    ELSE 1 -- Neutral move
                END AS move_status
            FROM first_moves
            ORDER BY date ASC;
        """)

def five_min_data_for_ticker_query(ticker='SPY', start_date='2018-01-01', start_time='00:00', end_time='23:59'):
    """
    Get query 5-minute candle data for a given ticker within a specified date and time range.
    """
    return text(
        f"""
            SELECT
                "timestamp",
                open,
                high,
                low,
                close,
                volume
            FROM stocks_fivemincandle
            WHERE ticker = '{ticker}'
            AND "timestamp"::date >= '{start_date}'
            AND "timestamp"::time BETWEEN '{start_time}' AND '{end_time}'
            ORDER BY "timestamp" ASC;
        """)