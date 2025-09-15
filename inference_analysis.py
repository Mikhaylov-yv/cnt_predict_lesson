#!/usr/bin/env python3
"""
Скрипт для инференса и анализа результатов модели трансформеров
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from transformer_predictor import ContainerDataProcessor, ContainerDataset, ContainerTransformer
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path='container_transformer_model.pth'):
    """Загрузка обученной модели"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Создаем модель
    model = ContainerTransformer(
        local_vocab_sizes=checkpoint['local_vocab_sizes'],
        local_embed_dims=checkpoint['local_embed_dims'],
        global_feature_dim=checkpoint['global_feature_dim'],
        d_model=256,
        nhead=8,
        num_layers=6,
        prediction_horizon=7
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def predict_future_states(model, data, state_encoder, location_encoder, 
                         cnt_id, sequence_length=30, prediction_horizon=7):
    """Предсказание будущих состояний для конкретного контейнера"""
    
    # Фильтруем данные по контейнеру
    cnt_data = data[data['cnt_id'] == cnt_id].copy()
    
    if len(cnt_data) < sequence_length:
        print(f"Недостаточно данных для контейнера {cnt_id}")
        return None
    
    # Берем последнюю последовательность
    last_sequence = cnt_data.tail(sequence_length)
    
    # Подготавливаем данные
    local_features = []
    
    # State encoding
    states = state_encoder.transform(last_sequence['state'].values)
    local_features.append(states)
    
    # Location encoding
    locations = location_encoder.transform(last_sequence['location'].fillna('unknown').values)
    local_features.append(locations)
    
    # Container type encoding
    cnt_types = state_encoder.transform(last_sequence['cnt_type'].values)  # Используем state_encoder для простоты
    local_features.append(cnt_types)
    
    # Time delta
    time_deltas = np.arange(sequence_length)
    local_features.append(time_deltas)
    
    local_features = np.column_stack(local_features)
    
    # Глобальные признаки
    global_cols = [col for col in last_sequence.columns if col.startswith(('agg_', 'req_', 'future_req_', 'day_of_', 'month', 'year'))]
    global_features = last_sequence[global_cols].values
    
    # Преобразуем в тензоры
    local_features = torch.FloatTensor(local_features).unsqueeze(0)
    global_features = torch.FloatTensor(global_features).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(local_features, global_features)
        
        state_probs = torch.softmax(outputs['state_predictions'], dim=-1)
        location_probs = torch.softmax(outputs['location_predictions'], dim=-1)
        
        # Получаем наиболее вероятные предсказания
        predicted_states = torch.argmax(state_probs, dim=-1).squeeze().numpy()
        predicted_locations = torch.argmax(location_probs, dim=-1).squeeze().numpy()
        
        # Декодируем
        predicted_states = state_encoder.inverse_transform(predicted_states)
        predicted_locations = location_encoder.inverse_transform(predicted_locations)
    
    return {
        'predicted_states': predicted_states,
        'predicted_locations': predicted_locations,
        'state_probabilities': state_probs.squeeze().numpy(),
        'location_probabilities': location_probs.squeeze().numpy()
    }

def aggregate_predictions(model, data, state_encoder, location_encoder, 
                         sequence_length=30, prediction_horizon=7):
    """Агрегация предсказаний по всем контейнерам"""
    
    # Получаем уникальные контейнеры
    unique_containers = data['cnt_id'].unique()
    
    # Агрегированные предсказания по дням
    daily_predictions = {}
    
    for day in range(prediction_horizon):
        daily_predictions[day] = {
            'states': {},
            'locations': {}
        }
    
    print(f"Обработка {len(unique_containers)} контейнеров...")
    
    for i, cnt_id in enumerate(unique_containers[:100]):  # Ограничиваем для демонстрации
        if i % 10 == 0:
            print(f"Обработано {i}/{min(100, len(unique_containers))} контейнеров")
        
        predictions = predict_future_states(
            model, data, state_encoder, location_encoder, 
            cnt_id, sequence_length, prediction_horizon
        )
        
        if predictions is None:
            continue
        
        # Агрегируем предсказания
        for day in range(prediction_horizon):
            state = predictions['predicted_states'][day]
            location = predictions['predicted_locations'][day]
            
            # Подсчитываем состояния
            if state not in daily_predictions[day]['states']:
                daily_predictions[day]['states'][state] = 0
            daily_predictions[day]['states'][state] += 1
            
            # Подсчитываем локации
            if location not in daily_predictions[day]['locations']:
                daily_predictions[day]['locations'][location] = 0
            daily_predictions[day]['locations'][location] += 1
    
    return daily_predictions

def plot_predictions(daily_predictions, save_path='predictions_analysis.png'):
    """Визуализация предсказаний"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График состояний по дням
    states_data = []
    for day, data in daily_predictions.items():
        for state, count in data['states'].items():
            states_data.append({'day': day, 'state': state, 'count': count})
    
    states_df = pd.DataFrame(states_data)
    if not states_df.empty:
        states_pivot = states_df.pivot(index='day', columns='state', values='count').fillna(0)
        states_pivot.plot(kind='bar', ax=axes[0, 0], stacked=True)
        axes[0, 0].set_title('Предсказанные состояния по дням')
        axes[0, 0].set_xlabel('День предсказания')
        axes[0, 0].set_ylabel('Количество контейнеров')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # График локаций по дням
    locations_data = []
    for day, data in daily_predictions.items():
        for location, count in data['locations'].items():
            locations_data.append({'day': day, 'location': location, 'count': count})
    
    locations_df = pd.DataFrame(locations_data)
    if not locations_df.empty:
        locations_pivot = locations_df.pivot(index='day', columns='location', values='count').fillna(0)
        locations_pivot.plot(kind='bar', ax=axes[0, 1], stacked=True)
        axes[0, 1].set_title('Предсказанные локации по дням')
        axes[0, 1].set_xlabel('День предсказания')
        axes[0, 1].set_ylabel('Количество контейнеров')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Тепловая карта состояний
    if not states_df.empty:
        states_heatmap = states_df.pivot_table(index='state', columns='day', values='count', aggfunc='sum').fillna(0)
        sns.heatmap(states_heatmap, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Тепловая карта состояний')
    
    # Тепловая карта локаций
    if not locations_df.empty:
        locations_heatmap = locations_df.pivot_table(index='location', columns='day', values='count', aggfunc='sum').fillna(0)
        sns.heatmap(locations_heatmap, annot=True, fmt='d', ax=axes[1, 1], cmap='Greens')
        axes[1, 1].set_title('Тепловая карта локаций')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция для инференса и анализа"""
    print("Загрузка модели...")
    
    try:
        model, checkpoint = load_model()
        state_encoder = checkpoint['state_encoder']
        location_encoder = checkpoint['location_encoder']
        cnt_type_encoder = checkpoint['cnt_type_encoder']
        scaler = checkpoint['scaler']
        
        print("Модель загружена успешно!")
        
    except FileNotFoundError:
        print("Модель не найдена. Сначала запустите обучение.")
        return
    
    # Обрабатываем данные для инференса
    print("Обработка данных...")
    processor = ContainerDataProcessor()
    processor.create_calendar()
    processor.process_transactions_to_daily_snapshots()
    processor.add_global_features()
    merged_data = processor.merge_local_and_global()
    
    # Нормализация глобальных признаков
    global_cols = [col for col in merged_data.columns if col.startswith(('agg_', 'req_', 'future_req_', 'day_of_', 'month', 'year'))]
    merged_data[global_cols] = scaler.transform(merged_data[global_cols])
    
    # Агрегация предсказаний
    print("Генерация предсказаний...")
    daily_predictions = aggregate_predictions(
        model, merged_data, state_encoder, location_encoder
    )
    
    # Визуализация
    print("Создание визуализаций...")
    plot_predictions(daily_predictions)
    
    # Вывод статистики
    print("\nСтатистика предсказаний:")
    for day, data in daily_predictions.items():
        print(f"\nДень {day + 1}:")
        print("Состояния:", data['states'])
        print("Локации:", data['locations'])
    
    print("\nАнализ завершен!")

if __name__ == "__main__":
    main()
