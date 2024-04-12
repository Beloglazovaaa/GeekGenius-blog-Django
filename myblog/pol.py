import warnings
from .models import DiabetesModel

# Отключение предупреждений
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('myblog/diabetes.csv')  # Поменяй 'diabetes.csv' на путь к вашему файлу

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


def predict_diabetes():
    print("Введите значения для всех признаков:")
    pregnancies = float(input("Количество беременностей: "))
    glucose = float(input("Уровень глюкозы: "))
    blood_pressure = float(input("Артериальное давление: "))
    skin_thickness = float(input("Толщина кожной складки: "))
    insulin = float(input("Инсулин: "))
    bmi = float(input("ИМТ: "))
    diabetes_pedigree_function = float(input("Генетический фактор: "))
    age = float(input("Возраст: "))

    # Масштабирование введенных пользователем данных
    user_data = scaler.transform(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Предсказание вероятности возникновения диабета
    probability = model.predict_proba(user_data)[:, 1][0]
    DiabetesModel.objects.create(pregnancies=pregnancies, glucose=glucose, blood_pressure=blood_pressure,
                                 skin_thickness=skin_thickness, insulin=insulin, bmi=bmi,
                                 diabetes_pedigree_function=diabetes_pedigree_function, age=age,
                                 probability=probability)
    return probability


# Пример использования функции для предсказания вероятности диабета у пациента
predicted_probability = predict_diabetes()
print("Predicted Probability of Diabetes:", predicted_probability)
