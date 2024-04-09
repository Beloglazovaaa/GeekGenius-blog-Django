# myblog/urls.py
from django.urls import path
from .views import MainView, PostDetailView, SignUpView, SignInView, polynomial_regression_page, gradient_boosting_page, SearchResultsView
from django.contrib.auth.views import LogoutView
from django.conf import settings

urlpatterns = [
    path('', MainView.as_view(), name='index'),
    path('blog/<slug>/', PostDetailView.as_view(), name='post_detail'),
    path('signup/', SignUpView.as_view(), name='signup'),
    path('signin/', SignInView.as_view(), name='signin'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('polynomial-regression/', polynomial_regression_page, name='polynomial_regression'),
    path('gradient-boosting/', gradient_boosting_page, name='gradient_boosting'),
    path('search/', SearchResultsView.as_view(), name='search_results'),
]
