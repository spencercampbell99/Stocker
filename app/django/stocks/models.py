from django.db import models
from django.utils import timezone

class Ticker(models.Model):
    """Model for stock tickers."""
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    sector = models.CharField(max_length=100, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.symbol


class BaseCandle(models.Model):
    """Abstract base model for all candle types."""
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(db_index=True)
    open = models.DecimalField(max_digits=12, decimal_places=4)
    high = models.DecimalField(max_digits=12, decimal_places=4)
    low = models.DecimalField(max_digits=12, decimal_places=4)
    close = models.DecimalField(max_digits=12, decimal_places=4)
    volume = models.BigIntegerField()
    
    class Meta:
        abstract = True
        ordering = ['ticker', 'timestamp']


class DailyCandle(BaseCandle):
    """Daily candle data."""
    class Meta:
        unique_together = ['ticker', 'timestamp']
        indexes = [
            models.Index(fields=['ticker', 'timestamp']),
        ]
        verbose_name = 'Daily Candle'
        verbose_name_plural = 'Daily Candles'

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d')}"


class FiveMinCandle(BaseCandle):
    """5-minute candle data."""
    class Meta:
        unique_together = ['ticker', 'timestamp']
        indexes = [
            models.Index(fields=['ticker', 'timestamp']),
        ]
        verbose_name = '5-Minute Candle'
        verbose_name_plural = '5-Minute Candles'

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class ThirtyMinCandle(BaseCandle):
    """30-minute candle data."""
    class Meta:
        unique_together = ['ticker', 'timestamp']
        indexes = [
            models.Index(fields=['ticker', 'timestamp']),
        ]
        verbose_name = '30-Minute Candle'
        verbose_name_plural = '30-Minute Candles'

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class HourCandle(BaseCandle):
    """Hourly candle data."""
    class Meta:
        unique_together = ['ticker', 'timestamp']
        indexes = [
            models.Index(fields=['ticker', 'timestamp']),
        ]
        verbose_name = 'Hour Candle'
        verbose_name_plural = 'Hour Candles'

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"