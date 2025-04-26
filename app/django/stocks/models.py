from django.db import models
from psqlextra.models import PostgresPartitionedModel
from psqlextra.types import PostgresPartitioningMethod
import uuid

class Ticker(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    sector = models.CharField(max_length=100, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['is_active'], name='ticker_active'),
        ]

    def __str__(self):
        return self.symbol


class BaseCandle(PostgresPartitionedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    ticker = models.CharField(max_length=10, db_index=True)
    timestamp = models.DateTimeField()
    open = models.DecimalField(max_digits=8, decimal_places=2)
    high = models.DecimalField(max_digits=8, decimal_places=2)
    low = models.DecimalField(max_digits=8, decimal_places=2)
    close = models.DecimalField(max_digits=8, decimal_places=2)
    volume = models.BigIntegerField()
    
    class PartitioningMeta:
        method = PostgresPartitioningMethod.LIST
        key = ["ticker"]
    
    class Meta:
        abstract = True
        ordering = ['ticker', 'timestamp']


class CandleModelMixin:
    """Mixin with common index configurations for candle models."""
    
    class Meta:
        abstract = True
        constraints = [
            models.UniqueConstraint(
                fields=['ticker', 'timestamp'],
                name='%(class)s_uq_tt'
            )
        ]
        indexes = [
            models.Index(
                fields=['ticker', 'timestamp'],
                name='%(class)s_t_ts',
            ),
        ]

    def __init_subclass__(cls, **kwargs):
        """Set BRIN pages_per_range for child classes."""
        super().__init_subclass__(**kwargs)


class DailyCandle(BaseCandle, CandleModelMixin):
    """Daily candle data."""
    
    class PartitioningMeta(BaseCandle.PartitioningMeta):
        pass
    
    class Meta(CandleModelMixin.Meta):
        verbose_name = 'Daily Candle'
        verbose_name_plural = 'Daily Candles'
        abstract = False

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d')}"


class FiveMinCandle(BaseCandle, CandleModelMixin):
    """5-minute candle data."""
    BRIN_PAGES_PER_RANGE = 128  # Higher for more frequent data
    
    class PartitioningMeta(BaseCandle.PartitioningMeta):
        pass
    
    class Meta(CandleModelMixin.Meta):
        verbose_name = '5-Minute Candle'
        verbose_name_plural = '5-Minute Candles'
        abstract = False

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class ThirtyMinCandle(BaseCandle, CandleModelMixin):
    """30-minute candle data."""
    
    class PartitioningMeta(BaseCandle.PartitioningMeta):
        pass
    
    class Meta(CandleModelMixin.Meta):
        verbose_name = '30-Minute Candle'
        verbose_name_plural = '30-Minute Candles'
        abstract = False

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class HourCandle(BaseCandle, CandleModelMixin):
    """Hourly candle data."""
    
    class PartitioningMeta(BaseCandle.PartitioningMeta):
        pass
    
    class Meta(CandleModelMixin.Meta):
        verbose_name = 'Hour Candle'
        verbose_name_plural = 'Hour Candles'
        abstract = False

    def __str__(self):
        return f"{self.ticker.symbol} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"