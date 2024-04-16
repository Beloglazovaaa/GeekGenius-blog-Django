# myblog/urls.py
from django.urls import path
from .views import MainView, PostDetailView, SignUpView, SignInView, polynomial_regression_page, gradient_boosting_page, \
    SearchResultsView, predict_diabetes_polynomial, predict_diabetes_gradient, predict_diabetes_recurrent
from django.contrib.auth.views import LogoutView
from django.conf import settings
from .views import recurrent_neural_network_page

urlpatterns = [
    path('', MainView.as_view(), name='index'),
    path('blog/<slug>/', PostDetailView.as_view(), name='post_detail'),
    path('signup/', SignUpView.as_view(), name='signup'),
    path('signin/', SignInView.as_view(), name='signin'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('polynomial-regression/', polynomial_regression_page, name='polynomial_regression'),
    path('gradient-boosting/', gradient_boosting_page, name='gradient_boosting'),
    path('recurrent_neural_network/', recurrent_neural_network_page, name='recurrent_neural_network'),
    path('search/', SearchResultsView.as_view(), name='search_results'),
    path('predict_diabetes_polynomial/', predict_diabetes_polynomial, name='predict_diabetes_polynomial'),
    path('predict_diabetes_gradient/', predict_diabetes_gradient, name='predict_diabetes_gradient'),
    path('predict_diabetes_recurrent/', predict_diabetes_recurrent, name='predict_diabetes_recurrent'),
]
