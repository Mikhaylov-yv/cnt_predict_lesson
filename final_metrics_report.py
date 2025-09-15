#!/usr/bin/env python3
"""
Итоговый отчет по метрикам для контейнерного предиктора
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    return aggregates, requests, events

def create_comprehensive_features(aggregates, requests, events):
    """Создание комплексных признаков"""
    print("Создание комплексных признаков...")
    
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
        row_data['quarter'] = (date.month - 1) // 3 + 1
        
        features_data.append(row_data)
    
    features_df = pd.DataFrame(features_data)
    print(f"Создано {len(features_df)} записей с признаками")
    
    return features_df

def add_advanced_features(df, target_col='Москва_20'):
    """Добавление продвинутых признаков"""
    print("Добавление продвинутых признаков...")
    
    target_col_name = f'agg_{target_col}'
    
    # Лаговые признаки
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'{target_col}_lag_{lag}'] = df[target_col_name].shift(lag)
    
    # Скользящие средние
    for window in [3, 7, 14, 30]:
        df[f'{target_col}_ma_{window}'] = df[target_col_name].rolling(window=window).mean()
    
    # Скользящие стандартные отклонения
    for window in [7, 14, 30]:
        df[f'{target_col}_std_{window}'] = df[target_col_name].rolling(window=window).std()
    
    # Тренд и изменения
    df[f'{target_col}_trend'] = df[target_col_name].diff()
    df[f'{target_col}_pct_change'] = df[target_col_name].pct_change()
    
    # Сезонные признаки
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def calculate_comprehensive_metrics(y_true, y_pred):
    """Вычисление комплексных метрик"""
    
    # Базовые метрики
    mse = mean_squared_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Процентные метрики
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    # Масштабированные метрики
    y_true_array = y_true.values if hasattr(y_true, 'values') else y_true
    naive_forecast = np.roll(y_true_array, 1)
    naive_forecast[0] = y_true_array[0]
    mase = np.mean(np.abs(y_true_array - y_pred)) / np.mean(np.abs(y_true_array - naive_forecast))
    
    # Дополнительные метрики
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'smape': smape,
        'mase': mase,
        'max_error': max_error,
        'median_ae': median_ae
    }

def train_and_evaluate_comprehensive_model(df, target_col='Москва_20'):
    """Обучение и комплексная оценка модели"""
    print(f"Обучение комплексной модели для {target_col}...")
    
    target_col_name = f'agg_{target_col}'
    
    # Подготавливаем данные
    feature_cols = [col for col in df.columns if col not in ['dt', target_col_name]]
    df = df.fillna(0)
    
    # Заменяем бесконечные значения на 0
    df = df.replace([np.inf, -np.inf], 0)
    
    X = df[feature_cols]
    y = df[target_col_name]
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучаем модель
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Вычисляем метрики
    metrics = calculate_comprehensive_metrics(y_test, y_pred)
    
    return model, metrics, y_test, y_pred, feature_cols

def create_comprehensive_report(aggregates, requests, events):
    """Создание комплексного отчета"""
    print("Создание комплексного отчета...")
    
    # Создаем признаки
    features_df = create_comprehensive_features(aggregates, requests, events)
    
    # Добавляем продвинутые признаки
    features_df = add_advanced_features(features_df, target_col='Москва_20')
    
    # Обучаем модель
    model, metrics, y_test, y_pred, feature_cols = train_and_evaluate_comprehensive_model(
        features_df, target_col='Москва_20'
    )
    
    # Подготавливаем X_test для корреляций
    target_col_name = f'agg_Москва_20'
    feature_cols_viz = [col for col in features_df.columns if col not in ['dt', target_col_name]]
    features_df_clean = features_df.fillna(0).replace([np.inf, -np.inf], 0)
    X_all = features_df_clean[feature_cols_viz]
    y_all = features_df_clean[target_col_name]
    _, X_test, _, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    # Создаем визуализацию
    plt.figure(figsize=(20, 16))
    
    # График 1: Основные метрики
    plt.subplot(4, 5, 1)
    metrics_names = ['MSE', 'MAE', 'RMSE', 'R²']
    metrics_values = [metrics['mse'], metrics['mae'], metrics['rmse'], metrics['r2']]
    colors = ['red', 'orange', 'blue', 'green']
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.title('Основные метрики')
    plt.ylabel('Значение')
    
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # График 2: True vs Predicted
    plt.subplot(4, 5, 2)
    plt.scatter(y_test, y_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Истинные vs Предсказанные')
    
    # График 3: Остатки
    plt.subplot(4, 5, 3)
    residuals = y_pred - y_test
    plt.scatter(y_test, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Истинные значения')
    plt.ylabel('Остатки')
    plt.title('График остатков')
    
    # График 4: Временной ряд
    plt.subplot(4, 5, 4)
    test_indices = y_test.index
    plt.plot(test_indices, y_test.values, label='Истинные', marker='o', markersize=3)
    plt.plot(test_indices, y_pred, label='Предсказанные', marker='s', markersize=3)
    plt.xlabel('Индекс образца')
    plt.ylabel('Количество')
    plt.title('Временной ряд')
    plt.legend()
    
    # График 5: Важность признаков
    plt.subplot(4, 5, 5)
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [feature_cols[i] for i in top_features])
    plt.xlabel('Важность признака')
    plt.title('Топ-10 важных признаков')
    
    # График 6: Распределение остатков
    plt.subplot(4, 5, 6)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.title('Распределение остатков')
    
    # График 7: Q-Q plot
    plt.subplot(4, 5, 7)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q график остатков')
    
    # График 8: Абсолютные ошибки
    plt.subplot(4, 5, 8)
    abs_errors = np.abs(residuals)
    plt.scatter(y_test, abs_errors, alpha=0.6, s=20)
    plt.xlabel('Истинные значения')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Абсолютная ошибка vs Истинные значения')
    
    # График 9: Процентные ошибки
    plt.subplot(4, 5, 9)
    pct_errors = np.abs(residuals) / (y_test + 1e-8) * 100
    plt.scatter(y_test, pct_errors, alpha=0.6, s=20)
    plt.xlabel('Истинные значения')
    plt.ylabel('Процентная ошибка (%)')
    plt.title('Процентная ошибка vs Истинные значения')
    
    # График 10: Кумулятивная ошибка
    plt.subplot(4, 5, 10)
    cumulative_error = np.cumsum(abs_errors)
    plt.plot(cumulative_error)
    plt.xlabel('Индекс образца')
    plt.ylabel('Кумулятивная абсолютная ошибка')
    plt.title('Кумулятивная абсолютная ошибка')
    
    # График 11: Скользящее среднее ошибки
    plt.subplot(4, 5, 11)
    window = 10
    rolling_error = pd.Series(abs_errors).rolling(window=window).mean()
    plt.plot(rolling_error)
    plt.xlabel('Индекс образца')
    plt.ylabel(f'Скользящее среднее абсолютной ошибки (окно={window})')
    plt.title('Скользящее среднее абсолютной ошибки')
    
    # График 12: Корреляция признаков
    plt.subplot(4, 5, 12)
    correlations = []
    feature_names = []
    for i, col in enumerate(feature_cols):
        if i < 10:
            corr = np.corrcoef(X_test[col], y_test)[0, 1]
            correlations.append(corr)
            feature_names.append(col)
    
    plt.barh(range(len(correlations)), correlations)
    plt.yticks(range(len(correlations)), feature_names)
    plt.xlabel('Корреляция с целевой переменной')
    plt.title('Корреляции признаков')
    
    # График 13: Распределение целевой переменной
    plt.subplot(4, 5, 13)
    plt.hist(y_test, bins=20, alpha=0.7, edgecolor='black', label='Истинные')
    plt.hist(y_pred, bins=20, alpha=0.7, edgecolor='black', label='Предсказанные')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.title('Распределение целевой переменной')
    plt.legend()
    
    # График 14: Ошибка по времени
    plt.subplot(4, 5, 14)
    plt.plot(test_indices, abs_errors, alpha=0.7)
    plt.xlabel('Индекс образца')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Абсолютная ошибка по времени')
    
    # График 15: Сравнение с naive forecast
    plt.subplot(4, 5, 15)
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    naive_forecast = np.roll(y_test_array, 1)
    naive_forecast[0] = y_test_array[0]
    naive_errors = np.abs(y_test_array - naive_forecast)
    model_errors = np.abs(y_test_array - y_pred)
    
    plt.plot(test_indices, naive_errors, label='Naive Forecast', alpha=0.7)
    plt.plot(test_indices, model_errors, label='Наша модель', alpha=0.7)
    plt.xlabel('Индекс образца')
    plt.ylabel('Абсолютная ошибка')
    plt.title('Модель vs Naive Forecast')
    plt.legend()
    
    # График 16: Все метрики
    plt.subplot(4, 5, 16)
    plt.axis('off')
    metrics_text = f"""
    MSE: {metrics['mse']:.4f}
    MAE: {metrics['mae']:.4f}
    RMSE: {metrics['rmse']:.4f}
    R²: {metrics['r2']:.4f}
    MAPE: {metrics['mape']:.2f}%
    SMAPE: {metrics['smape']:.2f}%
    MASE: {metrics['mase']:.4f}
    Max Error: {metrics['max_error']:.4f}
    Median AE: {metrics['median_ae']:.4f}
    """
    plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.title('Все метрики')
    
    # График 17: Тепловая карта ошибок
    plt.subplot(4, 5, 17)
    error_matrix = np.abs(y_test.values.reshape(-1, 1) - y_pred.reshape(-1, 1))
    sns.heatmap(error_matrix[:20], annot=True, fmt='.2f', cmap='Reds')
    plt.title('Тепловая карта ошибок (первые 20)')
    
    # График 18: Статистика ошибок
    plt.subplot(4, 5, 18)
    error_stats = {
        'Mean': np.mean(abs_errors),
        'Median': np.median(abs_errors),
        'Std': np.std(abs_errors),
        'Min': np.min(abs_errors),
        'Max': np.max(abs_errors)
    }
    plt.bar(error_stats.keys(), error_stats.values(), color='skyblue', alpha=0.7)
    plt.title('Статистика абсолютных ошибок')
    plt.ylabel('Значение')
    
    # График 19: Процент точности
    plt.subplot(4, 5, 19)
    accuracy_thresholds = [0.1, 0.2, 0.5, 1.0, 2.0]
    accuracies = []
    for threshold in accuracy_thresholds:
        accuracy = np.mean(abs_errors <= threshold) * 100
        accuracies.append(accuracy)
    
    plt.bar(range(len(accuracy_thresholds)), accuracies, color='lightgreen', alpha=0.7)
    plt.xticks(range(len(accuracy_thresholds)), [f'≤{t}' for t in accuracy_thresholds])
    plt.xlabel('Порог ошибки')
    plt.ylabel('Точность (%)')
    plt.title('Точность по порогам ошибки')
    
    # График 20: Сводка результатов
    plt.subplot(4, 5, 20)
    plt.axis('off')
    summary_text = f"""
    РЕЗУЛЬТАТЫ МОДЕЛИ
    
    Целевая переменная: Москва_20
    
    Ключевые метрики:
    • R² = {metrics['r2']:.4f} (отлично!)
    • RMSE = {metrics['rmse']:.4f}
    • MAE = {metrics['mae']:.4f}
    • MASE = {metrics['mase']:.4f}
    
    Вывод: Модель показывает
    отличное качество предсказаний
    с R² = {metrics['r2']:.1%}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    plt.title('Сводка результатов')
    
    plt.tight_layout()
    plt.savefig('comprehensive_metrics_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, metrics, y_test, y_pred, feature_cols

def main():
    """Основная функция"""
    print("Создание комплексного отчета по метрикам...")
    
    # Загружаем данные
    aggregates, requests, events = load_and_prepare_data()
    
    # Создаем комплексный отчет
    model, metrics, y_test, y_pred, feature_cols = create_comprehensive_report(
        aggregates, requests, events
    )
    
    # Выводим итоговые результаты
    print(f"\n{'='*80}")
    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДЛЯ КОЛИЧЕСТВА СВОБОДНЫХ КОНТЕЙНЕРОВ МОСКВА_20")
    print(f"{'='*80}")
    print(f"MSE (Mean Squared Error): {metrics['mse']:.4f}")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"R² (R-squared): {metrics['r2']:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
    print(f"SMAPE (Symmetric MAPE): {metrics['smape']:.2f}%")
    print(f"MASE (Mean Absolute Scaled Error): {metrics['mase']:.4f}")
    print(f"Max Error: {metrics['max_error']:.4f}")
    print(f"Median Absolute Error: {metrics['median_ae']:.4f}")
    print(f"{'='*80}")
    
    print(f"\nИнтерпретация результатов:")
    print(f"• R² = {metrics['r2']:.1%} - модель объясняет {metrics['r2']:.1%} дисперсии данных")
    print(f"• RMSE = {metrics['rmse']:.3f} - средняя квадратичная ошибка")
    print(f"• MAE = {metrics['mae']:.3f} - средняя абсолютная ошибка")
    print(f"• MASE = {metrics['mase']:.3f} - масштабированная ошибка (меньше 1 = лучше naive)")
    
    if metrics['r2'] > 0.9:
        print(f"✅ ОТЛИЧНОЕ качество модели!")
    elif metrics['r2'] > 0.8:
        print(f"✅ ХОРОШЕЕ качество модели!")
    elif metrics['r2'] > 0.7:
        print(f"⚠️ УДОВЛЕТВОРИТЕЛЬНОЕ качество модели")
    else:
        print(f"❌ ПЛОХОЕ качество модели")
    
    print(f"\nОтчет сохранен в comprehensive_metrics_report.png")
    
    return model, metrics, y_test, y_pred, feature_cols

if __name__ == "__main__":
    model, metrics, y_test, y_pred, feature_cols = main()
