from django.urls import path
from . import views
from .views import SentimentAnalysisAPI


urlpatterns = [
    path('', views.home, name='home'),
    path('api/sentiment/', SentimentAnalysisAPI.as_view(), name='sentiment-api'),

] 