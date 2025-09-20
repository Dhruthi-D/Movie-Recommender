from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Authentication
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    
    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    path('preferences/', views.preferences_view, name='preferences'),
    path('recommendations/', views.recommendations_view, name='recommendations'),
    path('preferences-recommendations/', views.preferences_recommendations_view, name='preferences_recommendations'),
    path('browse-movies/', views.browse_movies_view, name='browse_movies'),
    path('rate-movie/<int:movie_id>/', views.rate_movie, name='rate_movie'),
    path('accuracy/', views.accuracy_view, name='accuracy'),
    
    # API
    path('api/recommendations/<int:user_id>/', views.api_recommendations, name='api_recommendations'),
]

