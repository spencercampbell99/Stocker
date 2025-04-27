from django.urls import path
from . import views

urlpatterns = [
    path('api/status/', views.check_api_status, name='alpaca_api_status'),
    path('api/docs/', views.api_docs, name='alpaca_api_docs'),
]