from django.urls import path
from . import views

app_name = 'fourieroptics'
urlpatterns = [
    path('result/', views.result, name='result'),
    path('', views.home, name='home'),
    
    path('download_animation/', views.download_animation, name='download_animation'),
    path('download_animation_intensity/', views.download_animation_intensity, name='download_animation_intensity'),
    

]