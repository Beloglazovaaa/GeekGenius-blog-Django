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
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


from sklearn.ensemble import GradientBoostingClassifier

from django.http import JsonResponse
from .models import DiabetesModel

import warnings
from .models import DiabetesModel

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, SimpleRNN

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import LSTM, Dense

def polynomial_regression_page(request):
    return render(request, 'myblog/polynomial_regression.html')


def gradient_boosting_page(request):
    return render(request, 'myblog/gradient_boosting.html')


def recurrent_neural_network_page(request):
    return render(request, 'myblog/recurrent_neural_network.html')


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


def train_model_polynomial():
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

    # Сохранение обученной модели для последующего использования
    joblib.dump(model, 'polynomial_regression_model.pkl')
    return model, scaler
    return "Модель обучена и сохранена."
@csrf_exempt
def predict_diabetes_polynomial(request):
    # Получение данных из POST-запроса
    pregnancies = float(request.POST.get('pregnancies'))
    glucose = float(request.POST['glucose'])
    blood_pressure = float(request.POST.get('blood-pressure'))
    skin_thickness = float(request.POST.get('skin-thickness'))
    insulin = float(request.POST.get('insulin'))
    bmi = float(request.POST.get('bmi'))
    diabetes_pedigree_function = float(request.POST.get('diabetes-pedigree'))
    age = float(request.POST.get('age'))

    model, scaler = train_model_polynomial()

    model = joblib.load('polynomial_regression_model.pkl')
    # Масштабирование введенных пользователем данных
    user_data = scaler.transform(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Предсказание вероятности возникновения диабета
    probability = model.predict_proba(user_data)[:, 1][0]

    # Сохранение предсказанных данных в базе данных
    DiabetesModel.objects.create(pregnancies=pregnancies, glucose=glucose, bloodpressure=blood_pressure,
                                 skinthickness=skin_thickness, insulin=insulin, bmi=bmi,
                                 diabetespedigreefunction=diabetes_pedigree_function, age=age,
                                 probability=probability)

    # Возврат предсказанной вероятности диабета в формате JSON
    return JsonResponse({'probability': probability})

def train_model_gradient():
    data = pd.read_csv('myblog/diabetes.csv')

    # Разделение данных на признаки (X) и целевую переменную (y)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Масштабирование признаков
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Создание и обучение модели логистической регрессии
    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)

    # Сохранение обученной модели для последующего использования
    joblib.dump(model, 'gradient_boosting_model.pkl')
    return model, scaler
    return "Модель обучена и сохранена."


def predict_diabetes_gradient(request):
    # Получение данных из POST-запроса
    pregnancies = float(request.POST.get('pregnancies'))
    glucose = float(request.POST['glucose'])
    blood_pressure = float(request.POST.get('blood-pressure'))
    skin_thickness = float(request.POST.get('skin-thickness'))
    insulin = float(request.POST.get('insulin'))
    bmi = float(request.POST.get('bmi'))
    diabetes_pedigree_function = float(request.POST.get('diabetes-pedigree'))
    age = float(request.POST.get('age'))

    model, scaler = train_model_gradient()

    model = joblib.load('gradient_boosting_model.pkl')
    # Масштабирование введенных пользователем данных
    user_data = scaler.transform(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Предсказание вероятности возникновения диабета
    probability = model.predict_proba(user_data)[:, 1][0]

    # Сохранение предсказанных данных в базе данных
    DiabetesModel.objects.create(pregnancies=pregnancies, glucose=glucose, bloodpressure=blood_pressure,
                                 skinthickness=skin_thickness, insulin=insulin, bmi=bmi,
                                 diabetespedigreefunction=diabetes_pedigree_function, age=age,
                                 probability=probability)

    # Возврат предсказанной вероятности диабета в формате JSON
    return JsonResponse({'probability': probability})



def train_model_recurrent():
    # Загрузка данных
    data = pd.read_csv('myblog/diabetes.csv')

    # Разделение данных на признаки (X) и целевую переменную (y)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Преобразование данных для использования в RNN
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Создание модели RNN
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))

    # Компиляция модели
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Обучение модели
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Сохранение обученной модели для последующего использования
    model.save('rnn_model.h5')
    return model, scaler
    return "Модель обучена и сохранена."


@csrf_exempt
def predict_diabetes_recurrent(request):
    # Получение данных из POST-запроса
    pregnancies = float(request.POST.get('pregnancies'))
    glucose = float(request.POST['glucose'])
    blood_pressure = float(request.POST.get('blood-pressure'))
    skin_thickness = float(request.POST.get('skin-thickness'))
    insulin = float(request.POST.get('insulin'))
    bmi = float(request.POST.get('bmi'))
    diabetes_pedigree_function = float(request.POST.get('diabetes-pedigree'))
    age = float(request.POST.get('age'))

    # Загрузка и использование обученной модели RNN
    model, scaler = train_model_recurrent()

    # Загрузка сохраненной модели
    model = tf.keras.models.load_model('rnn_model.h5')

    # Масштабирование введенных пользователем данных
    user_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Преобразование данных для использования в RNN
    '''user_data = np.reshape(user_data, (user_data.shape[0], 1, user_data.shape[1]))
'''
    # Предсказание вероятности возникновения диабета
    probability = model.predict(user_data)[0][0]

    # Сохранение предсказанных данных в базе данных
    DiabetesModel.objects.create(pregnancies=pregnancies, glucose=glucose, bloodpressure=blood_pressure,
                                 skinthickness=skin_thickness, insulin=insulin, bmi=bmi,
                                 diabetespedigreefunction=diabetes_pedigree_function, age=age,
                                 probability=probability)

    # Возврат предсказанной вероятности диабета в формате JSON
    return JsonResponse({'probability': probability})



def get_latest_diabetes_prediction(request):
    if request.method == 'GET':
        # Получаем последнюю запись из таблицы DiabetesModel
        latest_prediction = DiabetesModel.objects.latest('id')

        # Формируем JSON-ответ с последним результатом
        response_data = {
            'probability': latest_prediction.probability
        }

        # Возвращаем JSON-ответ
        return JsonResponse(response_data)
