from django.urls import path
from . import views

urlpatterns = [
    path('prediction', views.get_v01_model_predictions, name='get_prediction'),
]