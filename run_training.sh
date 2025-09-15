#!/bin/bash

# Скрипт для запуска обучения модели трансформеров

echo "Создание виртуального окружения..."
python3 -m venv .venv

echo "Активация виртуального окружения..."
source .venv/bin/activate

echo "Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Запуск обучения модели..."
python transformer_predictor.py

echo "Обучение завершено!"
