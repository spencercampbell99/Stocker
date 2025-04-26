import sys
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from stocks.models import Ticker, DailyCandle, FiveMinCandle, ThirtyMinCandle, HourCandle


class Command(BaseCommand):
    help = 'Manage PostgreSQL table partitions for candle data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--create-ticker-partitions',
            action='store_true',
            help='Create partitions for all tickers',
        )
        parser.add_argument(
            '--create-ticker-partition',
            type=str,
            help='Create partitions for a specific ticker symbol',
        )
        parser.add_argument(
            '--show-partitions',
            action='store_true',
            help='Show all existing partitions',
        )
        parser.add_argument(
            '--analyze',
            action='store_true',
            help='Run ANALYZE on partitioned tables to update statistics',
        )

    def handle(self, *args, **options):
        if options['create_ticker_partitions']:
            self.create_partitions_for_all_tickers()
        elif options['create_ticker_partition']:
            self.create_partition_for_ticker(options['create_ticker_partition'])
        elif options['show_partitions']:
            self.show_partitions()
        elif options['analyze']:
            self.analyze_tables()
        else:
            self.stdout.write(self.style.WARNING('No action specified. Use --help for available options.'))

    def create_partitions_for_all_tickers(self):
        """Create partitions for all tickers in database."""
        self.stdout.write(self.style.SUCCESS('Creating partitions for all tickers...'))
        
        tickers = Ticker.objects.all()
        
        for ticker in tickers:
            try:
                self.create_partition_for_ticker(ticker.symbol)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating partitions for {ticker.symbol}: {e}'))
                continue
        
        self.stdout.write(self.style.SUCCESS('Done creating partitions for all tickers.'))

    def create_partition_for_ticker(self, symbol):
        """Create partitions for a specific ticker."""
        self.stdout.write(self.style.SUCCESS(f'Creating partitions for ticker {symbol}...'))
        
        try:
            ticker = Ticker.objects.get(symbol=symbol)
        except Ticker.DoesNotExist:
            raise CommandError(f'Ticker with symbol "{symbol}" does not exist')
        
        # Create partitions for each candle type
        DailyCandle.create_partition_for_ticker(symbol)
        FiveMinCandle.create_partition_for_ticker(symbol)
        ThirtyMinCandle.create_partition_for_ticker(symbol)
        HourCandle.create_partition_for_ticker(symbol)
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created partitions for ticker {symbol}'))

    def show_partitions(self):
        """Show all existing partitions."""
        self.stdout.write(self.style.SUCCESS('Showing all partitions:'))
        
        with connection.cursor() as cursor:
            cursor.execute("""
            SELECT
                nmsp_parent.nspname AS parent_schema,
                parent.relname AS parent_table,
                nmsp_child.nspname AS child_schema,
                child.relname AS child_table
            FROM pg_inherits
            JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
            JOIN pg_class child ON pg_inherits.inhrelid = child.oid
            JOIN pg_namespace nmsp_parent ON parent.relnamespace = nmsp_parent.oid
            JOIN pg_namespace nmsp_child ON child.relnamespace = nmsp_child.oid
            WHERE parent.relname LIKE 'stocks_%candle'
            ORDER BY parent_schema, parent_table, child_schema, child_table;
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                self.stdout.write(self.style.WARNING('No partitions found.'))
                return
                
            for row in rows:
                self.stdout.write(f"Parent table: {row[0]}.{row[1]}, Child partition: {row[2]}.{row[3]}")

    def analyze_tables(self):
        """Analyze the tables to update statistics for the query planner."""
        self.stdout.write(self.style.SUCCESS('Analyzing tables to update statistics...'))
        
        with connection.cursor() as cursor:
            cursor.execute("ANALYZE stocks_dailycandle;")
            cursor.execute("ANALYZE stocks_fivemincandle;")
            cursor.execute("ANALYZE stocks_thirtymincandle;")
            cursor.execute("ANALYZE stocks_hourcandle;")
            
        self.stdout.write(self.style.SUCCESS('Done analyzing tables.'))