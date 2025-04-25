from django.contrib import admin
from .models import Ticker, DailyCandle, FiveMinCandle, ThirtyMinCandle, HourCandle

@admin.register(Ticker)
class TickerAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'name', 'sector', 'is_active')
    search_fields = ('symbol', 'name')
    list_filter = ('is_active', 'sector')

@admin.register(DailyCandle)
class DailyCandleAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('ticker',)
    date_hierarchy = 'timestamp'

@admin.register(FiveMinCandle)
class FiveMinCandleAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('ticker',)
    date_hierarchy = 'timestamp'

@admin.register(ThirtyMinCandle)
class ThirtyMinCandleAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('ticker',)
    date_hierarchy = 'timestamp'

@admin.register(HourCandle)
class HourCandleAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('ticker',)
    date_hierarchy = 'timestamp'