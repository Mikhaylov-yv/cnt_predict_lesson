"""
Скрипт преобразования набора данных cnt_events.csv

Создает датасет для ML модели прогнозирования освобождения контейнера на локации.
Каждая последовательность (от "Передан_на_погрузку" до "Прибыл_свободным") обрабатывается отдельно.
Последовательность начинается строго с события "Передан_на_погрузку" и заканчивается "Прибыл_свободным".
"""

import pandas as pd
import numpy as np
from datetime import datetime


def transform_cnt_events(input_file='data/cnt_events.csv', output_file='data/cnt_events_transformed.csv'):
    """
    Преобразует события контейнеров в датасет для ML модели.
    
    Параметры:
    ----------
    input_file : str
        Путь к исходному файлу с событиями
    output_file : str
        Путь к выходному файлу
    """
    print("Загрузка данных...")
    df = pd.read_csv(input_file)
    df['dt'] = pd.to_datetime(df['dt'])
    
    # Сохраняем исходный индекс строки для правильной сортировки при одинаковых датах
    df['original_index'] = df.index
    
    # Сортируем по контейнеру, дате и исходному индексу (для сохранения порядка при одинаковых датах)
    df = df.sort_values(['cnt_id', 'dt', 'original_index']).reset_index(drop=True)
    
    print(f"Загружено {len(df)} событий для {df['cnt_id'].nunique()} контейнеров")
    
    # Результирующий датафрейм
    result_rows = []
    
    # Обрабатываем каждый контейнер отдельно
    for cnt_id in df['cnt_id'].unique():
        container_events = df[df['cnt_id'] == cnt_id].copy()
        # Сортируем по дате и исходному индексу (для сохранения порядка при одинаковых датах)
        container_events = container_events.sort_values(['dt', 'original_index']).reset_index(drop=True)
        
        # Находим все последовательности для этого контейнера
        sequences = find_sequences(container_events)
        
        # Обрабатываем каждую последовательность
        for seq_num, seq in enumerate(sequences, 1):
            cycle_id = f"{cnt_id}_cycle_{seq_num}"
            end_location = seq['end_location']
            start_idx = seq['start_idx']
            end_idx = seq['end_idx']
            cycle_end_date = seq['end_date']
            
            # Получаем все события последовательности от "Передан_на_погрузку" до "Прибыл_свободным" включительно
            cycle_events = container_events.iloc[start_idx:end_idx + 1].copy()
            cycle_events = cycle_events.reset_index(drop=True)
            
            # Добавляем метрики для каждого события
            for i, (_, row) in enumerate(cycle_events.iterrows()):
                # Дни с последнего event (для этого контейнера в этом цикле)
                if i == 0:
                    days_since_last = 0
                else:
                    prev_date = cycle_events.iloc[i - 1]['dt']
                    days_since_last = (row['dt'] - prev_date).days
                
                # Дни до "Прибыл_свободным" для этого цикла
                # Если событие происходит в тот же день, что и "Прибыл_свободным", то days_until_free = 0
                if row['dt'].date() == cycle_end_date.date():
                    days_until_free = 0
                else:
                    days_until_free = (cycle_end_date - row['dt']).days
                
                result_rows.append({
                    'dt': row['dt'],
                    'cnt_id': row['cnt_id'],
                    'cycle_id': cycle_id,
                    'event': row['event'],
                    'location_name': row['location_name'],
                    'end_location_name': end_location,
                    'cnt_type': row['cnt_type'],
                    'days_since_last_event': days_since_last,
                    'days_until_free': days_until_free
                })
    
    # Создаем результирующий датафрейм
    result_df = pd.DataFrame(result_rows)
    
    # Сортируем по дате и контейнеру
    result_df = result_df.sort_values(['dt', 'cnt_id']).reset_index(drop=True)
    
    print(f"\nРезультаты преобразования:")
    print(f"  Всего записей: {len(result_df)}")
    print(f"  Уникальных контейнеров: {result_df['cnt_id'].nunique()}")
    print(f"  Уникальных последовательностей: {result_df['cycle_id'].nunique()}")
    print(f"  Период данных: {result_df['dt'].min()} - {result_df['dt'].max()}")
    
    # Статистика по событиям
    print(f"\nРаспределение событий:")
    print(result_df['event'].value_counts())
    
    # Сохраняем результат
    result_df.to_csv(output_file, index=False)
    print(f"\nРезультат сохранен в {output_file}")
    
    return result_df


def find_sequences(container_events):
    """
    Находит все последовательности перевозки для контейнера.
    Последовательность начинается строго с "Передан_на_погрузку" и заканчивается "Прибыл_свободным".
    
    Параметры:
    ----------
    container_events : pd.DataFrame
        События одного контейнера, отсортированные по дате
    
    Возвращает:
    -----------
    list of dict
        Список словарей с информацией о последовательностях:
        - start_idx: индекс начала последовательности (событие "Передан_на_погрузку")
        - end_idx: индекс конца последовательности (событие "Прибыл_свободным")
        - end_date: дата окончания последовательности
        - end_location: локация, куда контейнер прибыл свободным
    """
    sequences = []
    i = 0
    
    while i < len(container_events):
        # Ищем начало последовательности - "Передан_на_погрузку"
        if container_events.iloc[i]['event'] == 'Передан_на_погрузку':
            start_idx = i
            
            # Ищем конец последовательности - "Прибыл_свободным" после начала
            found_end = False
            for j in range(i + 1, len(container_events)):
                if container_events.iloc[j]['event'] == 'Прибыл_свободным':
                    end_idx = j
                    end_date = container_events.iloc[j]['dt']
                    end_location = container_events.iloc[j]['location_name']
                    
                    sequences.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'end_date': end_date,
                        'end_location': end_location
                    })
                    
                    found_end = True
                    i = j + 1  # Продолжаем поиск следующей последовательности
                    break
            
            # Если последовательность не завершена, пропускаем её
            if not found_end:
                i += 1
        else:
            i += 1
    
    return sequences


if __name__ == '__main__':
    result = transform_cnt_events()
    print("\nПреобразование завершено успешно!")

