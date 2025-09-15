#!/usr/bin/env python3
"""
Простой предиктор для контейнеров с метриками (без PyTorch)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SimpleContainerPredictor:
    """Простой предиктор для контейнеров"""
    
    def __init__(self):
        self.calendar = None
        self.daily_snapshots = None
        self.global_features = None
        self.merged_data = None
        self.model = None
        self.scaler = None
        self.state_encoder = None
        self.location_encoder = None
        
    def create_calendar(self, start_date='2006-01-01', end_date=None):
        """Создание календаря всех дат в диапазоне"""
        print("Создание календаря...")
        
        if end_date is None:
            # Найдем максимальную дату из всех данных
            events = pd.read_csv('data/cnt_events.csv')
            events['dt'] = pd.to_datetime(events['dt'])
            max_date = events['dt'].max()
            end_date = max_date.strftime('%Y-%m-%d')
        
        self.calendar = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq='D'
        ).to_frame(name='dt')
        
        print(f"Календарь создан: {len(self.calendar)} дней")
        return self.calendar
    
    def process_transactions_to_daily_snapshots(self):
        """Преобразование транзакций в daily-snapshots"""
        print("Обработка транзакций в daily-snapshots...")
        
        # Загружаем транзакции
        events = pd.read_csv('data/cnt_events.csv')
        events['dt'] = pd.to_datetime(events['dt'])
        
        # Сортируем по cnt_id и времени
        events = events.sort_values(['cnt_id', 'dt'])
        
        # Создаем mapping событий в состояния
        event_to_state = {
            'Прибыл_свободным': 'free',
            'Передан_на_погрузку': 'loading',
            'Отправлен_загруженным': 'loaded',
            'Прибыл_загруженным': 'arrived_loaded',
            'Передан_на_выгрузку': 'unloading',
            'Отправлен_свободным': 'free'
        }
        
        events['state'] = events['event'].map(event_to_state)
        events['location'] = events['location_name']
        
        # Для каждого контейнера создаем daily snapshots
        daily_snapshots = []
        
        for cnt_id in events['cnt_id'].unique():
            cnt_events = events[events['cnt_id'] == cnt_id].copy()
            
            # Создаем последовательность дат для контейнера
            start_date = cnt_events['dt'].min()
            end_date = cnt_events['dt'].max()
            cnt_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Инициализируем состояние
            current_state = 'unknown'
            current_location = None
            current_cnt_type = cnt_events['cnt_type'].iloc[0]
            
            for date in cnt_dates:
                # Проверяем, есть ли событие в этот день
                day_events = cnt_events[cnt_events['dt'] == date]
                
                if not day_events.empty:
                    # Берем последнее событие дня
                    last_event = day_events.iloc[-1]
                    current_state = last_event['state']
                    current_location = last_event['location']
                
                daily_snapshots.append({
                    'dt': date,
                    'cnt_id': cnt_id,
                    'cnt_type': current_cnt_type,
                    'state': current_state,
                    'location': current_location
                })
        
        self.daily_snapshots = pd.DataFrame(daily_snapshots)
        print(f"Daily snapshots созданы: {len(self.daily_snapshots)} записей")
        return self.daily_snapshots
    
    def add_global_features(self):
        """Добавление глобальных признаков"""
        print("Добавление глобальных признаков...")
        
        # Загружаем агрегаты
        aggregates = pd.read_csv('data/cnt_empty_count.csv')
        aggregates['dt'] = pd.to_datetime(aggregates['dt'])
        
        # Загружаем заявки
        requests = pd.read_csv('data/transport_requests.csv')
        requests['Дата_заявки'] = pd.to_datetime(requests['Дата_заявки'])
        requests['Планируемая_дата_отправления'] = pd.to_datetime(requests['Планируемая_дата_отправления'])
        
        # Создаем глобальные признаки по датам
        global_features = []
        
        for date in self.calendar['dt']:
            # Агрегаты на эту дату
            agg_row = aggregates[aggregates['dt'] == date]
            agg_features = {}
            
            if not agg_row.empty:
                for col in aggregates.columns[1:]:  # Пропускаем dt
                    agg_features[f'agg_{col}'] = agg_row[col].iloc[0]
            else:
                # Если нет данных, заполняем нулями
                for col in aggregates.columns[1:]:
                    agg_features[f'agg_{col}'] = 0
            
            # Заявки на эту дату (по дате заявки)
            day_requests = requests[requests['Дата_заявки'] == date]
            
            # Количество заявок по типам и направлениям
            request_features = {}
            for cnt_type in [20, 40]:
                for from_loc in ['Москва', 'Владивосток', 'Санкт-Петербург']:
                    for to_loc in ['Москва', 'Владивосток', 'Санкт-Петербург']:
                        if from_loc != to_loc:
                            key = f'req_{from_loc}_{to_loc}_{cnt_type}'
                            count = len(day_requests[
                                (day_requests['Тип_контейнера'] == cnt_type) &
                                (day_requests['Локация_отправления'] == from_loc) &
                                (day_requests['Локация_назначения'] == to_loc)
                            ])
                            request_features[key] = count
            
            # Lagged features (заявки на будущее)
            future_requests = requests[requests['Планируемая_дата_отправления'] == date]
            for cnt_type in [20, 40]:
                for from_loc in ['Москва', 'Владивосток', 'Санкт-Петербург']:
                    for to_loc in ['Москва', 'Владивосток', 'Санкт-Петербург']:
                        if from_loc != to_loc:
                            key = f'future_req_{from_loc}_{to_loc}_{cnt_type}'
                            count = len(future_requests[
                                (future_requests['Тип_контейнера'] == cnt_type) &
                                (future_requests['Локация_отправления'] == from_loc) &
                                (future_requests['Локация_назначения'] == to_loc)
                            ])
                            request_features[key] = count
            
            # Дополнительные временные признаки
            time_features = {
                'day_of_week': date.weekday(),
                'day_of_year': date.timetuple().tm_yday,
                'month': date.month,
                'year': date.year
            }
            
            # Объединяем все признаки
            all_features = {**agg_features, **request_features, **time_features}
            all_features['dt'] = date
            
            global_features.append(all_features)
        
        self.global_features = pd.DataFrame(global_features)
        print(f"Глобальные признаки созданы: {len(self.global_features)} записей")
        return self.global_features
    
    def merge_local_and_global(self):
        """Соединение локальных и глобальных данных"""
        print("Соединение локальных и глобальных данных...")
        
        # Объединяем daily_snapshots с глобальными признаками
        self.merged_data = self.daily_snapshots.merge(
            self.global_features, 
            on='dt', 
            how='left'
        )
        
        # Заполняем пропуски в глобальных признаках
        global_cols = [col for col in self.merged_data.columns if col.startswith(('agg_', 'req_', 'future_req_'))]
        self.merged_data[global_cols] = self.merged_data[global_cols].fillna(0)
        
        print(f"Объединенные данные: {len(self.merged_data)} записей")
        return self.merged_data
    
    def prepare_features(self):
        """Подготовка признаков для обучения"""
        print("Подготовка признаков...")
        
        # Обрабатываем пропуски
        self.merged_data['state'] = self.merged_data['state'].fillna('unknown')
        self.merged_data['location'] = self.merged_data['location'].fillna('unknown')
        
        # Кодирование категориальных признаков
        self.state_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        
        self.merged_data['state_encoded'] = self.state_encoder.fit_transform(self.merged_data['state'])
        self.merged_data['location_encoded'] = self.location_encoder.fit_transform(self.merged_data['location'])
        
        # Временные признаки
        self.merged_data['day_of_week'] = self.merged_data['dt'].dt.dayofweek
        self.merged_data['day_of_year'] = self.merged_data['dt'].dt.dayofyear
        self.merged_data['month'] = self.merged_data['dt'].dt.month
        self.merged_data['year'] = self.merged_data['dt'].dt.year
        
        # Нормализация числовых признаков
        numeric_cols = ['cnt_type', 'state_encoded', 'location_encoded', 'day_of_week', 
                       'day_of_year', 'month', 'year']
        global_cols = [col for col in self.merged_data.columns if col.startswith(('agg_', 'req_', 'future_req_'))]
        
        all_numeric_cols = numeric_cols + global_cols
        
        self.scaler = StandardScaler()
        self.merged_data[all_numeric_cols] = self.scaler.fit_transform(self.merged_data[all_numeric_cols])
        
        return all_numeric_cols
    
    def create_aggregate_targets(self, target_col='Москва_20'):
        """Создание целевых переменных для агрегатов"""
        print(f"Создание целевых переменных для {target_col}...")
        
        # Загружаем истинные агрегаты
        true_aggregates = pd.read_csv('data/cnt_empty_count.csv')
        true_aggregates['dt'] = pd.to_datetime(true_aggregates['dt'])
        
        # Создаем целевые переменные для каждого дня
        targets = []
        
        for date in self.merged_data['dt'].unique():
            # Подсчитываем количество свободных контейнеров в Москве 20 на эту дату
            day_data = self.merged_data[self.merged_data['dt'] == date]
            
            free_moscow_20_count = len(day_data[
                (day_data['state'] == 'free') & 
                (day_data['location'] == 'Москва') & 
                (day_data['cnt_type'] == 20)
            ])
            
            # Берем истинное значение из агрегатов
            true_value = true_aggregates[true_aggregates['dt'] == date][target_col].iloc[0] if len(true_aggregates[true_aggregates['dt'] == date]) > 0 else 0
            
            targets.append({
                'dt': date,
                'predicted_count': free_moscow_20_count,
                'true_count': true_value
            })
        
        return pd.DataFrame(targets)
    
    def train_model(self, feature_cols, target_col='Москва_20'):
        """Обучение модели"""
        print("Обучение модели...")
        
        # Создаем агрегированные данные по дням
        daily_data = self.merged_data.groupby('dt').agg({
            'state_encoded': 'mean',
            'location_encoded': 'mean',
            'cnt_type': 'mean',
            'day_of_week': 'first',
            'day_of_year': 'first',
            'month': 'first',
            'year': 'first',
            **{col: 'mean' for col in self.merged_data.columns if col.startswith(('agg_', 'req_', 'future_req_'))}
        }).reset_index()
        
        # Добавляем лаговые признаки
        for lag in [1, 2, 3, 7, 14]:
            for col in ['agg_Москва_20', 'agg_Владивосток_20', 'agg_Санкт-Петербург_20']:
                daily_data[f'{col}_lag_{lag}'] = daily_data[col].shift(lag)
        
        # Заполняем пропуски
        daily_data = daily_data.fillna(0)
        
        # Создаем целевую переменную
        targets_df = self.create_aggregate_targets(target_col)
        daily_data = daily_data.merge(targets_df[['dt', 'true_count']], on='dt', how='left')
        daily_data['true_count'] = daily_data['true_count'].fillna(0)
        
        # Подготавливаем признаки
        feature_cols = [col for col in daily_data.columns if col not in ['dt', 'true_count']]
        X = daily_data[feature_cols]
        y = daily_data['true_count']
        
        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучаем модель
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        
        # Вычисляем метрики
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"МЕТРИКИ ДЛЯ {target_col}")
        print(f"{'='*50}")
        print(f"MSE (Mean Squared Error): {mse:.4f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"R² (R-squared): {r2:.4f}")
        print(f"{'='*50}")
        
        # Визуализация
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted')
        
        plt.subplot(2, 3, 2)
        residuals = y_pred - y_test
        plt.scatter(y_test, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        plt.subplot(2, 3, 3)
        plt.plot(y_test.values, label='True', marker='o')
        plt.plot(y_pred, label='Predicted', marker='s')
        plt.xlabel('Sample')
        plt.ylabel('Count')
        plt.title('True vs Predicted (Time Series)')
        plt.legend()
        
        plt.subplot(2, 3, 4)
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:]
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.yticks(range(len(top_features)), [feature_cols[i] for i in top_features])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importance')
        
        plt.subplot(2, 3, 5)
        plt.hist(residuals, bins=20, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        plt.subplot(2, 3, 6)
        plt.plot(y_test.values, y_pred, 'o', alpha=0.6)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Scatter Plot')
        
        plt.tight_layout()
        plt.savefig('metrics_analysis_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'mse': mse,
            'mape': mape,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': feature_importance,
            'feature_names': feature_cols
        }

def main():
    """Основная функция"""
    print("Запуск простого контейнерного предиктора с метриками...")
    
    # Создаем предиктор
    predictor = SimpleContainerPredictor()
    
    # Обрабатываем данные
    calendar = predictor.create_calendar()
    daily_snapshots = predictor.process_transactions_to_daily_snapshots()
    global_features = predictor.add_global_features()
    merged_data = predictor.merge_local_and_global()
    
    # Подготавливаем признаки
    feature_cols = predictor.prepare_features()
    
    # Обучаем модель и получаем метрики
    metrics = predictor.train_model(feature_cols, target_col='Москва_20')
    
    print(f"\nОбучение завершено!")
    print(f"Модель сохранена в памяти")
    
    return predictor, metrics

if __name__ == "__main__":
    predictor, metrics = main()

