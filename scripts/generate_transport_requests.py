#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для генерации таблицы заявок на перевозку контейнеров
на основе данных о событиях с контейнерами.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Tuple

def load_data():
    """Загружает исходные данные"""
    events_df = pd.read_csv('data/cnt_events.csv')
    empty_count_df = pd.read_csv('data/cnt_empty_count.csv')
    
    # Преобразуем даты
    events_df['dt'] = pd.to_datetime(events_df['dt'])
    empty_count_df['dt'] = pd.to_datetime(empty_count_df['dt'])
    
    return events_df, empty_count_df

def get_available_locations_and_types(events_df):
    """Извлекает доступные локации и типы контейнеров"""
    locations = events_df['location_name'].unique()
    container_types = events_df['cnt_type'].unique()
    
    # Убираем пустые значения
    locations = [loc for loc in locations if pd.notna(loc)]
    container_types = [ct for ct in container_types if pd.notna(ct)]
    
    return locations, container_types

def generate_transport_requests(events_df, empty_count_df, num_requests=1000):
    """
    Генерирует заявки на перевозку контейнеров
    
    Параметры:
    - events_df: DataFrame с событиями контейнеров
    - empty_count_df: DataFrame с количеством пустых контейнеров
    - num_requests: количество заявок для генерации
    """
    
    locations, container_types = get_available_locations_and_types(events_df)
    
    # Диапазон дат из исходных данных
    start_date = events_df['dt'].min()
    end_date = events_df['dt'].max()
    
    requests = []
    
    for i in range(num_requests):
        # Генерируем случайную дату заявки в диапазоне данных
        request_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        # Планируемая дата отправления - через 1-7 дней после заявки
        planned_departure = request_date + timedelta(
            days=random.randint(1, 7)
        )
        
        # Количество контейнеров (1-5)
        container_count = random.randint(1, 5)
        
        # Тип контейнера
        container_type = random.choice(container_types)
        
        # Локации отправления и назначения (разные)
        origin = random.choice(locations)
        destination = random.choice([loc for loc in locations if loc != origin])
        
        request = {
            'Дата_заявки': request_date.strftime('%Y-%m-%d'),
            'Планируемая_дата_отправления': planned_departure.strftime('%Y-%m-%d'),
            'Количество_контейнеров': container_count,
            'Тип_контейнера': container_type,
            'Локация_отправления': origin,
            'Локация_назначения': destination
        }
        
        requests.append(request)
    
    return pd.DataFrame(requests)

def add_realistic_patterns(df, empty_count_df):
    """
    Добавляет реалистичные паттерны на основе данных о пустых контейнерах
    """
    # Сортируем по дате заявки
    df = df.sort_values('Дата_заявки').reset_index(drop=True)
    
    # Добавляем сезонность - больше заявок в определенные месяцы
    df['month'] = pd.to_datetime(df['Дата_заявки']).dt.month
    seasonal_multiplier = {
        1: 0.8, 2: 0.7, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.4,
        7: 1.5, 8: 1.4, 9: 1.2, 10: 1.1, 11: 1.0, 12: 0.9
    }
    
    # Применяем сезонный коэффициент к количеству контейнеров
    for month, multiplier in seasonal_multiplier.items():
        mask = df['month'] == month
        df.loc[mask, 'Количество_контейнеров'] = np.ceil(
            df.loc[mask, 'Количество_контейнеров'] * multiplier
        ).astype(int)
    
    # Убираем временный столбец
    df = df.drop('month', axis=1)
    
    return df

def main():
    """Основная функция"""
    print("Загружаем данные...")
    events_df, empty_count_df = load_data()
    
    print(f"Загружено {len(events_df)} событий с контейнерами")
    print(f"Диапазон дат: {events_df['dt'].min().date()} - {events_df['dt'].max().date()}")
    
    print("Генерируем заявки на перевозку...")
    requests_df = generate_transport_requests(events_df, empty_count_df, num_requests=2000)
    
    print("Добавляем реалистичные паттерны...")
    requests_df = add_realistic_patterns(requests_df, empty_count_df)
    
    # Сохраняем результат
    output_file = 'data/transport_requests.csv'
    requests_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Сгенерировано {len(requests_df)} заявок на перевозку")
    print(f"Результат сохранен в файл: {output_file}")
    
    # Показываем статистику
    print("\nСтатистика по заявкам:")
    print(f"Типы контейнеров: {sorted(requests_df['Тип_контейнера'].unique())}")
    print(f"Локации: {sorted(requests_df['Локация_отправления'].unique())}")
    print(f"Среднее количество контейнеров: {requests_df['Количество_контейнеров'].mean():.2f}")
    print(f"Диапазон дат заявок: {requests_df['Дата_заявки'].min()} - {requests_df['Дата_заявки'].max()}")
    
    # Показываем первые несколько записей
    print("\nПервые 10 заявок:")
    print(requests_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
