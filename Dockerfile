# Используйте официальный образ Python как базовый
FROM python:3.8-slim
# Установите рабочую директорию в контейнере
WORKDIR /code
# Скопируйте файлы проекта в контейнер
COPY . /code/
# Установите зависимости
RUN pip install --no-cache-dir -r requirements.txt
