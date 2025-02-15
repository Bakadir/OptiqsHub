from django.urls import path

from . import views

from django.conf import settings
from django.conf.urls.static import static
app_name = 'polarization'
urlpatterns = [
    path('jones', views.jonescalculus, name='jonescalculus'),
    path('mueller', views.muellercalculus, name='muellercalculus'),
    path('', views.pythoncode, name='pythoncode'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)