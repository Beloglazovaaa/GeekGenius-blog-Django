from django.urls import path
from .views import MainView, PostDetailView, SignUpView
from django.contrib.auth import views as auth_views
from .views import SignInView
from django.contrib.auth.views import LogoutView
from django.conf import settings


urlpatterns = [
    path('', MainView.as_view(), name='index'),
    path('blog/<slug>/', PostDetailView.as_view(), name='post_detail'),
    path('signup/', SignUpView.as_view(), name='signup'),
    path('signin/', SignInView.as_view(), name='signin'),
    path('signout/', LogoutView.as_view(), {'next_page': settings.LOGOUT_REDIRECT_URL}, name='signout'),
]