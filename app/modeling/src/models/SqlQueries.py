from sqlalchemy import text

def get_daily_move_status_query():
    return text(
        """
            WITH daily_open AS (
                SELECT 
                    "timestamp"::date AS date,
                    open AS open_930
                FROM stocks_fivemincandle
                WHERE ticker = 'SPY'
                AND "timestamp"::time = '09:30:00'
                AND "timestamp" >= '2018-01-01'
            ),
            time_markers AS (
                SELECT
                    f."timestamp"::date AS date,
                    f."timestamp",
                    f.high,
                    f.low,
                    o.open_930,
                    CASE WHEN f.high >= o.open_930 * 1.0075 THEN f."timestamp" END AS up_timestamp,
                    CASE WHEN f.low <= o.open_930 * 0.9925 THEN f."timestamp" END AS down_timestamp
                FROM stocks_fivemincandle f
                JOIN daily_open o ON f."timestamp"::date = o.date
                WHERE f.ticker = 'SPY'
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
            ORDER BY date;
        """)