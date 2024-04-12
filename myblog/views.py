from django.shortcuts import render, get_object_or_404, redirect
from django.views import View
from django.core.paginator import Paginator
from .models import Post, DataModel
from .forms import SignUpForm, SignInForm
from django.contrib.auth import login, authenticate, logout
from django.http import HttpResponseRedirect
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render
from django.db.models import Q
from django.http import JsonResponse
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import json


def polynomial_regression_page(request):
    return render(request, 'myblog/polynomial_regression.html')

def gradient_boosting_page(request):
    return render(request, 'myblog/gradient_boosting.html')

def recurrent_neural_network_page(request):
    return render(request, 'myblog/recurrent_neural_network.html')


def generate_data():
    for i in range(100):
        feature1 = np.random.rand()
        feature2 = np.random.rand()
        target = feature1 * 0.5 + feature2 * 0.5 + np.random.rand() * 0.1  # Пример простой линейной зависимости
        DataModel.objects.create(feature1=feature1, feature2=feature2, target=target)


def train_model(request):
    if not DataModel.objects.exists():
        generate_data()

    data = DataModel.objects.all().values_list('feature1', 'feature2', 'target')
    X, y = [list(row[:2]) for row in data], [row[2] for row in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Сохранение модели
    joblib.dump(model, 'linear_regression_model.pkl')

    return JsonResponse({"message": "Модель успешно обучена."})

@csrf_exempt
def predict_model(request):
    # Загрузка обученной модели
    model = joblib.load('linear_regression_model.pkl')

    # Извлечение входных данных из POST-запроса
    data = json.loads(request.body)
    feature1 = data.get('feature1')
    feature2 = data.get('feature2')

    if feature1 is None or feature2 is None:
        return JsonResponse({"error": "Не указаны необходимые входные данные."}, status=400)

    try:
        # Предсказание с помощью модели
        prediction = model.predict([[int(feature1), int(feature2)]])
        return JsonResponse({"prediction": prediction[0]})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def prediction_page(request):
    return render(request, 'myblog/prediction_page.html')


class MainView(View):
    def get(self, request, *args, **kwargs):
        posts = Post.objects.all().order_by('-created_at')
        paginator = Paginator(posts, 3)

        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, 'myblog/index.html', context={
            'page_obj': page_obj
        })


class PostDetailView(View):
    def get(self, request, slug, *args, **kwargs):
        post = get_object_or_404(Post, url=slug)
        return render(request, 'myblog/post_detail.html', context={
            'post': post
        })

class SignUpView(View):
    def get(self, request, *args, **kwargs):
        form = SignUpForm()
        return render(request, 'myblog/signup.html', context={
            'form': form,
        })

    def post(self, request, *args, **kwargs):
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            if user is not None:
                login(request, user)
                return HttpResponseRedirect('/')
        return render(request, 'myblog/signup.html', context={
            'form': form,
        })


class SignInView(View):
    def get(self, request, *args, **kwargs):
        form = SignInForm()
        return render(request, 'myblog/signin.html', context={
            'form': form,
        })

    def post(self, request, *args, **kwargs):
        form = SignInForm(request.POST)
        if form.is_valid():
            username = request.POST['username']
            password = request.POST['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return HttpResponseRedirect('/')
            else:
                form.add_error(None, "Неправильный пароль или указанная учётная запись не существует!")
                return render(request, "myblog/signin.html", {"form": form})
        return render(request, 'myblog/signin.html', context={
            'form': form,
        })


class LogoutView(View):
    def get(self, request, *args, **kwargs):
        logout(request)
        return redirect('index')  # Перенаправляем на главную страницу после выхода

class SearchResultsView(View):
    def get(self, request, *args, **kwargs):
        query = request.GET.get('q')
        results = ""
        if query:
            results = Post.objects.filter(
                Q(content__icontains=query)
            )
        paginator = Paginator(results, 3)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        return render(request, 'myblog/search.html', context={
            'query': query,
            'title': 'Поиск',
            'results': page_obj,
            'count': paginator.count
        })



from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


def train_model_function():
    # Генерация симуляционных данных
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

    # Разделение данных на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Сохранение обученной модели для последующего использования
    joblib.dump(model, 'linear_regression_model.pkl')

    return "Модель обучена и сохранена."





from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

import warnings
from .models import DiabetesModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Отключение предупреждений
warnings.filterwarnings("ignore")


def polynomial_regression_page(request):
    return render(request, 'myblog/polynomial_regression.html')


def train_model_function():
    # Генерация симуляционных данных
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

    # Разделение данных на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Сохранение обученной модели для последующего использования
    joblib.dump(model, 'linear_regression_model.pkl')

    return "Модель обучена и сохранена."

@csrf_exempt
def predict_diabetes(request):
    if request.method == 'POST':
        # Получение данных из POST-запроса
        pregnancies = float(request.POST.get('pregnancies'))
        glucose = float(request.POST.get('glucose'))
        blood_pressure = float(request.POST.get('blood_pressure'))
        skin_thickness = float(request.POST.get('skin_thickness'))
        insulin = float(request.POST.get('insulin'))
        bmi = float(request.POST.get('bmi'))
        diabetes_pedigree_function = float(request.POST.get('diabetes_pedigree_function'))
        age = float(request.POST.get('age'))

        # Загрузка данных для модели логистической регрессии
        data = pd.read_csv('myblog/diabetes.csv')

        # Разделение данных на признаки (X) и целевую переменную (y)
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']

        # Масштабирование признаков
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        # Создание и обучение модели логистической регрессии
        model = LogisticRegression()
        model.fit(X_scaled, y)

        # Масштабирование введенных пользователем данных
        user_data = scaler.transform(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

        # Предсказание вероятности возникновения диабета
        probability = model.predict_proba(user_data)[:, 1][0]

        # Сохранение предсказанных данных в базе данных
        DiabetesModel.objects.create(pregnancies=pregnancies, glucose=glucose, blood_pressure=blood_pressure,
                                     skin_thickness=skin_thickness, insulin=insulin, bmi=bmi,
                                     diabetes_pedigree_function=diabetes_pedigree_function, age=age,
                                     probability=probability)

        # Возврат предсказанной вероятности диабета в формате JSON
        return JsonResponse({'probability': probability})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
