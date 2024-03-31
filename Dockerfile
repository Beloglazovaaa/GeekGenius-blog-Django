# Используйте официальный образ Python как базовый
FROM python:3.8-slim
# Установите рабочую директорию в контейнере
WORKDIR /code
# Скопируйте файлы проекта в контейнер
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r /code/requirements.txt

