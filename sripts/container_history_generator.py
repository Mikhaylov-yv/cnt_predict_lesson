import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import random

# Конфигурация
start_date = datetime(2006, 1, 1)
end_date = datetime(2008, 1, 1)
locations = ["Москва", "Владивосток", "Санкт-Петербург"]
cnt_types = ["20", "40"]
cnt_type_probs = [2/3, 1/3]
num_containers = 100
# Префиксы из South Park
characters = ["KNY", "STN", "CRT", "BUT", "ERK", "TIM", "WED", "TOW"]

# События без маршрутов и их параметры (медиана, std)
event_params = {
    "Прибыл_свободным": (20, 20),
    "Передан_на_погрузку": (10, 6),
    "Прибыл_груженый": (2, 1),
    "Прибыл_по_ЖД": (2, 1),
    "Отправлен_грузополучателю": (14, 10)
}
# Параметры Отправлен_по_ЖД в зависимости от маршрута (медиана)
routes_median = {
    ("Санкт-Петербург", "Владивосток"): 16,
    ("Москва", "Владивосток"): 13,
    ("Санкт-Петербург", "Москва"): 1
}
# в обратном направлении те же значения
for (a, b), m in list(routes_median.items()):
    routes_median[(b, a)] = m
# std для ЖД-события
rail_std = 3

# Функция для выборки длительности (с учётом, что отрицательных нет)
def sample_duration(mean_days, std_days):
    duration = max(0, np.random.normal(mean_days, std_days))
    return timedelta(days=int(duration))

# Генерация уникальных идентификаторов

def generate_container_ids(n):
    ids = set()
    while len(ids) < n:
        prefix = random.choice(characters)
        num = f"{random.randint(1000000, 9999999)}"
        ids.add(f"{prefix}U{num}")
    return list(ids)

# Генерация маршрута одного контейнера

def generate_container_route(cnt_id):
    data = []
    current_date = start_date + timedelta(days=random.randint(0, 30))
    cnt_type = np.random.choice(cnt_types, p=cnt_type_probs)
    location = random.choice(locations)

    # Цикл событий
    while current_date < end_date:
        # Прибыл_свободным
        for event in ["Прибыл_свободным", "Передан_на_погрузку", "Прибыл_груженый"]:
            mean, std = event_params[event]
            duration = sample_duration(mean, std)
            data.append({
                "dt": current_date.strftime("%Y-%m-%d"),
                "event": event,
                "cnt_id": cnt_id,
                "cnt_type": cnt_type,
                "location_name": location,
                "end_location_name": None
            })
            current_date += duration
        # Отправлен_по_ЖД
        end_loc = random.choice([loc for loc in locations if loc != location])
        median = routes_median.get((location, end_loc), np.mean([v for v in routes_median.values()]))
        duration = sample_duration(median, rail_std)
        data.append({
            "dt": current_date.strftime("%Y-%m-%d"),
            "event": "Отправлен_по_ЖД",
            "cnt_id": cnt_id,
            "cnt_type": cnt_type,
            "location_name": location,
            "end_location_name": end_loc
        })
        current_date += duration
        # Прибыл_по_ЖД
        mean, std = event_params["Прибыл_по_ЖД"]
        duration = sample_duration(mean, std)
        data.append({
            "dt": current_date.strftime("%Y-%m-%d"),
            "event": "Прибыл_по_ЖД",
            "cnt_id": cnt_id,
            "cnt_type": cnt_type,
            "location_name": end_loc,
            "end_location_name": None
        })
        current_date += duration
        # Отправлен_грузополучателю
        mean, std = event_params["Отправлен_грузополучателю"]
        duration = sample_duration(mean, std)
        data.append({
            "dt": current_date.strftime("%Y-%m-%d"),
            "event": "Отправлен_грузополучателю",
            "cnt_id": cnt_id,
            "cnt_type": cnt_type,
            "location_name": end_loc,
            "end_location_name": None
        })
        current_date += duration
        # Цикл повторяется, контейнер снова свободен в end_loc
        location = end_loc

    return data

# Основная генерация
t_container_ids = generate_container_ids(num_containers)
all_data = []
for cid in t_container_ids:
    all_data.extend(generate_container_route(cid))

# Сохранение в CSV
df = pd.DataFrame(all_data)
df.to_csv("./../data/container_history.csv", index=False, encoding='utf-8-sig')
print("Файл container_movements.csv сгенерирован.")
