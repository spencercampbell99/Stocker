from django.contrib import admin
from .models import Ticker, DailyCandle, FiveMinCandle, ThirtyMinCandle, HourCandle

@admin.register(Ticker)
class TickerAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'name', 'sector', 'is_active', 'created_at', 'updated_at')
    search_fields = ('symbol', 'name', 'sector')
    list_filter = ('is_active', 'sector', 'created_at')
    readonly_fields = ('created_at', 'updated_at')

class BaseCandleAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('ticker',)
    search_fields = ('ticker__symbol',)
    date_hierarchy = 'timestamp'
    readonly_fields = ('id',)
    ordering = ('ticker', 'timestamp')

@admin.register(DailyCandle)
class DailyCandleAdmin(BaseCandleAdmin):
    pass

@admin.register(FiveMinCandle)
class FiveMinCandleAdmin(BaseCandleAdmin):
    pass

@admin.register(ThirtyMinCandle)
class ThirtyMinCandleAdmin(BaseCandleAdmin):
    pass

@admin.register(HourCandle)
class HourCandleAdmin(BaseCandleAdmin):
    pass