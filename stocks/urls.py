from django.contrib import admin
from django.urls import path, include\

from stocks import views

# from django.urls import re_path
# from django.views.generic import TemplateView

urlpatterns = [
    path("", views.home, name="home"),
    path("results/", views.results, name="results"),
    path('stockname/', views.stock_page, name='stockname'),
]
