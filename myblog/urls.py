# myblog/urls.py
from django.urls import path
from .views import MainView, PostDetailView, SignUpView, SignInView, polynomial_regression_page, gradient_boosting_page, \
    SearchResultsView, train_model, predict_model, \
    prediction_page
from django.contrib.auth.views import LogoutView
from django.conf import settings
from .views import recurrent_neural_network_page
from django.urls import path
from . import views

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
    path('train-model/', train_model, name='train-model'),
    path('run-prediction/', predict_model, name='run-prediction'),
    path('prediction-page/', prediction_page, name='prediction-page'),
    path('predict_diabetes/', views.predict_diabetes, name='predict_diabetes'),
    path('get_latest_diabetes_prediction/', views.get_latest_diabetes_prediction, name='get_latest_diabetes_prediction'),
]

