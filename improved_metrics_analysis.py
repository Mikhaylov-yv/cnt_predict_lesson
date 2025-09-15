#!/usr/bin/env python3
"""
Улучшенный анализ метрик для агрегатов контейнеров
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    print("Загрузка данных...")
    
    # Загружаем агрегаты
    aggregates = pd.read_csv('data/cnt_empty_count.csv')
    aggregates['dt'] = pd.to_datetime(aggregates['dt'])
    
    # Загружаем заявки
    requests = pd.read_csv('data/transport_requests.csv')
    requests['Дата_заявки'] = pd.to_datetime(requests['Дата_заявки'])
    requests['Планируемая_дата_отправления'] = pd.to_datetime(requests['Планируемая_дата_отправления'])
    
    # Загружаем события
    events = pd.read_csv('data/cnt_events.csv')
    events['dt'] = pd.to_datetime(events['dt'])
    
    print(f"Агрегаты: {len(aggregates)} записей")
    print(f"Заявки: {len(requests)} записей")
    print(f"События: {len(events)} записей")
    
    return aggregates, requests, events

def create_features(aggregates, requests, events):
    """Создание признаков для модели"""
    print("Создание признаков...")
    
    # Создаем календарь
    start_date = aggregates['dt'].min()
    end_date = aggregates['dt'].max()
    calendar = pd.date_range(start=start_date, end=end_date, freq='D')
    
    features_data = []
    
    for date in calendar:
        # Базовые признаки из агрегатов
        agg_row = aggregates[aggregates['dt'] == date]
        if agg_row.empty:
            continue
            
        row_data = {'dt': date}
        
        # Агрегаты на эту дату
        for col in aggregates.columns[1:]:
            row_data[f'agg_{col}'] = agg_row[col].iloc[0]
        
        # Заявки на эту дату
        day_requests = requests[requests['Дата_заявки'] == date]
        row_data['total_requests'] = len(day_requests)
        
        # Заявки по типам контейнеров
        for cnt_type in [20, 40]:
            type_requests = day_requests[day_requests['Тип_контейнера'] == cnt_type]
            row_data[f'requests_{cnt_type}'] = len(type_requests)
            
            # Заявки по направлениям
            for from_loc in ['Москва', 'Владивосток', 'Санкт-Петербург']:
                for to_loc in ['Москва', 'Владивосток', 'Санкт-Петербург']:
                    if from_loc != to_loc:
                        route_requests = type_requests[
                            (type_requests['Локация_отправления'] == from_loc) &
                            (type_requests['Локация_назначения'] == to_loc)
                        ]
                        row_data[f'req_{from_loc}_{to_loc}_{cnt_type}'] = len(route_requests)
        
        # Заявки на будущее (планируемые отправления)
        future_requests = requests[requests['Планируемая_дата_отправления'] == date]
        row_data['future_requests'] = len(future_requests)
        
        # События на эту дату
        day_events = events[events['dt'] == date]
        row_data['total_events'] = len(day_events)
        
        # События по типам
        for event_type in ['Прибыл_свободным', 'Передан_на_погрузку', 'Отправлен_загруженным', 
                          'Прибыл_загруженным', 'Передан_на_выгрузку', 'Отправлен_свободным']:
            event_count = len(day_events[day_events['event'] == event_type])
            row_data[f'event_{event_type}'] = event_count
        
        # Временные признаки
        row_data['day_of_week'] = date.weekday()
        row_data['day_of_year'] = date.timetuple().tm_yday
        row_data['month'] = date.month
        row_data['year'] = date.year
        row_data['is_weekend'] = 1 if date.weekday() >= 5 else 0
        
        features_data.append(row_data)
    
    features_df = pd.DataFrame(features_data)
    print(f"Создано {len(features_df)} записей с признаками")
    
    return features_df

def add_lag_features(df, target_col='Москва_20', lags=[1, 2, 3, 7, 14, 30]):
    """Добавление лаговых признаков"""
    print("Добавление лаговых признаков...")
    
    # Используем правильное имя колонки
    target_col_name = f'agg_{target_col}'
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col_name].shift(lag)
    
    # Добавляем скользящие средние
    for window in [3, 7, 14, 30]:
        df[f'{target_col}_ma_{window}'] = df[target_col_name].rolling(window=window).mean()
    
    # Добавляем тренд (разность между текущим и предыдущим значением)
    df[f'{target_col}_trend'] = df[target_col_name].diff()
    
    return df

def calculate_improved_metrics(y_true, y_pred):
    """Вычисление улучшенных метрик"""
    
    # MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # R²
    r2 = r2_score(y_true, y_pred)
    
    # MAPE с защитой от деления на ноль
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    # Mean Absolute Scaled Error (MASE)
    # Используем naive forecast (предыдущее значение) как baseline
    y_true_array = y_true.values if hasattr(y_true, 'values') else y_true
    naive_forecast = np.roll(y_true_array, 1)
    naive_forecast[0] = y_true_array[0]  # Первое значение остается как есть
    mase = np.mean(np.abs(y_true_array - y_pred)) / np.mean(np.abs(y_true_array - naive_forecast))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'mase': mase,
        'r2': r2
    }

def train_and_evaluate_model(df, target_col='Москва_20'):
    """Обучение и оценка модели"""
    print(f"Обучение модели для {target_col}...")
    
    # Используем правильное имя колонки
    target_col_name = f'agg_{target_col}'
    
    # Подготавливаем данные
    feature_cols = [col for col in df.columns if col not in ['dt', target_col_name]]
    
    # Заполняем пропуски
    df = df.fillna(0)
    
    X = df[feature_cols]
    y = df[target_col_name]
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучаем модель
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Вычисляем улучшенные метрики
    metrics = calculate_improved_metrics(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"МЕТРИКИ ДЛЯ {target_col}")
    print(f"{'='*70}")
    print(f"MSE (Mean Squared Error): {metrics['mse']:.4f}")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
    print(f"SMAPE (Symmetric MAPE): {metrics['smape']:.2f}%")
    print(f"MASE (Mean Absolute Scaled Error): {metrics['mase']:.4f}")
    print(f"R² (R-squared): {metrics['r2']:.4f}")
    print(f"{'='*70}")
    
    # Визуализация
    plt.figure(figsize=(20, 15))
    
    # График 1: True vs Predicted
    plt.subplot(4, 4, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted')
    
    # График 2: Остатки
    plt.subplot(4, 4, 2)
    residuals = y_pred - y_test
    plt.scatter(y_test, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    # График 3: Временной ряд
    plt.subplot(4, 4, 3)
    test_indices = y_test.index
    plt.plot(test_indices, y_test.values, label='True', marker='o', markersize=3)
    plt.plot(test_indices, y_pred, label='Predicted', marker='s', markersize=3)
    plt.xlabel('Sample Index')
    plt.ylabel('Count')
    plt.title('True vs Predicted (Time Series)')
    plt.legend()
    
    # График 4: Важность признаков
    plt.subplot(4, 4, 4)
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-15:]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [feature_cols[i] for i in top_features])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance')
    
    # График 5: Распределение остатков
    plt.subplot(4, 4, 5)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    # График 6: Q-Q plot
    plt.subplot(4, 4, 6)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    # График 7: Абсолютные ошибки
    plt.subplot(4, 4, 7)
    abs_errors = np.abs(residuals)
    plt.scatter(y_test, abs_errors, alpha=0.6, s=20)
    plt.xlabel('True Values')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs True Values')
    
    # График 8: Процентные ошибки
    plt.subplot(4, 4, 8)
    pct_errors = np.abs(residuals) / (y_test + 1e-8) * 100
    plt.scatter(y_test, pct_errors, alpha=0.6, s=20)
    plt.xlabel('True Values')
    plt.ylabel('Percentage Error (%)')
    plt.title('Percentage Error vs True Values')
    
    # График 9: Кумулятивная ошибка
    plt.subplot(4, 4, 9)
    cumulative_error = np.cumsum(np.abs(residuals))
    plt.plot(cumulative_error)
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative Absolute Error')
    plt.title('Cumulative Absolute Error')
    
    # График 10: Скользящее среднее ошибки
    plt.subplot(4, 4, 10)
    window = 10
    rolling_error = pd.Series(np.abs(residuals)).rolling(window=window).mean()
    plt.plot(rolling_error)
    plt.xlabel('Sample Index')
    plt.ylabel(f'Rolling Mean Absolute Error (window={window})')
    plt.title('Rolling Mean Absolute Error')
    
    # График 11: Корреляция между признаками и целевой переменной
    plt.subplot(4, 4, 11)
    correlations = []
    feature_names = []
    for i, col in enumerate(feature_cols):
        if i < 10:  # Показываем только первые 10
            corr = np.corrcoef(X_test[col], y_test)[0, 1]
            correlations.append(corr)
            feature_names.append(col)
    
    plt.barh(range(len(correlations)), correlations)
    plt.yticks(range(len(correlations)), feature_names)
    plt.xlabel('Correlation with Target')
    plt.title('Feature Correlations')
    
    # График 12: Сравнение метрик
    plt.subplot(4, 4, 12)
    metrics_names = ['MSE', 'MAE', 'RMSE', 'R²']
    metrics_values = [metrics['mse'], metrics['mae'], metrics['rmse'], metrics['r2']]
    colors = ['red', 'orange', 'blue', 'green']
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.ylabel('Value')
    plt.title('Model Metrics')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # График 13: Распределение целевой переменной
    plt.subplot(4, 4, 13)
    plt.hist(y_test, bins=20, alpha=0.7, edgecolor='black', label='True')
    plt.hist(y_pred, bins=20, alpha=0.7, edgecolor='black', label='Predicted')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Target Variable')
    plt.legend()
    
    # График 14: Ошибка по времени
    plt.subplot(4, 4, 14)
    plt.plot(test_indices, abs_errors, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error Over Time')
    
    # График 15: Сравнение с naive forecast
    plt.subplot(4, 4, 15)
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    naive_forecast = np.roll(y_test_array, 1)
    naive_forecast[0] = y_test_array[0]
    naive_errors = np.abs(y_test_array - naive_forecast)
    model_errors = np.abs(y_test_array - y_pred)
    
    plt.plot(test_indices, naive_errors, label='Naive Forecast', alpha=0.7)
    plt.plot(test_indices, model_errors, label='Our Model', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error')
    plt.title('Model vs Naive Forecast')
    plt.legend()
    
    # График 16: Метрики в виде таблицы
    plt.subplot(4, 4, 16)
    plt.axis('off')
    metrics_text = f"""
    MSE: {metrics['mse']:.4f}
    MAE: {metrics['mae']:.4f}
    RMSE: {metrics['rmse']:.4f}
    MAPE: {metrics['mape']:.2f}%
    SMAPE: {metrics['smape']:.2f}%
    MASE: {metrics['mase']:.4f}
    R²: {metrics['r2']:.4f}
    """
    plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.title('All Metrics Summary')
    
    plt.tight_layout()
    plt.savefig(f'improved_metrics_analysis_{target_col}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model': model,
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': feature_importance,
        'feature_names': feature_cols
    }

def main():
    """Основная функция"""
    print("Запуск улучшенного анализа метрик...")
    
    # Загружаем данные
    aggregates, requests, events = load_and_prepare_data()
    
    # Создаем признаки
    features_df = create_features(aggregates, requests, events)
    
    # Добавляем лаговые признаки
    features_df = add_lag_features(features_df, target_col='Москва_20')
    
    # Обучаем модель и получаем метрики
    results = train_and_evaluate_model(features_df, target_col='Москва_20')
    
    print(f"\nАнализ завершен!")
    print(f"Модель обучена на {len(features_df)} записях")
    
    return results

if __name__ == "__main__":
    results = main()
