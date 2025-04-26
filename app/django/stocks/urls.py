from django.urls import path
from . import views

urlpatterns = [
    path('', views.stock_dashboard, name='stock_dashboard'),
    path('api/stock-data/', views.get_candle_data, name='get_candle_data'),
]