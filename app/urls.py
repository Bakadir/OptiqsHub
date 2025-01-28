from django.urls import path , include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'app'
urlpatterns = [
    path('', views.home, name='home'),
    path('refractiveindex_data', views.refractiveindex_data, name='refractiveindex_data'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup, name='signup'),
    path('forgot_password/', views.forgot_password, name='forgot_password'),

    path('logout/', views.logout_view, name='logout'),
    path('<slug:slug>/', views.account_view, name='account'),

    path('<str:username>/<slug:title>/', views.view_simulation, name='view_simulation'),
    
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)