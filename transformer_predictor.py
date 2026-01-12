#!/usr/bin/env python3
"""
Контейнерный предиктор на основе трансформеров
Полный pipeline от данных до предсказаний
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Настройка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

class ContainerDataProcessor:
    """Класс для обработки данных контейнеров"""
    
    def __init__(self):
        self.calendar = None
        self.daily_snapshots = None
        self.global_features = None
        self.merged_data = None
        
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

class ContainerDataset(Dataset):
    """Датасет для обучения трансформера"""
    
    def __init__(self, data, sequence_length=30, prediction_horizon=7, 
                 state_encoder=None, location_encoder=None, cnt_type_encoder=None, scaler=None):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.state_encoder = state_encoder
        self.location_encoder = location_encoder
        self.cnt_type_encoder = cnt_type_encoder
        self.scaler = scaler
        
        # Подготавливаем данные
        self._prepare_data()
    
    def _prepare_data(self):
        """Подготовка данных для обучения"""
        # Сортируем по cnt_id и дате
        self.data = self.data.sort_values(['cnt_id', 'dt'])
        
        # Создаем последовательности
        self.sequences = []
        self.targets = []
        
        for cnt_id in self.data['cnt_id'].unique():
            cnt_data = self.data[self.data['cnt_id'] == cnt_id].copy()
            
            if len(cnt_data) < self.sequence_length + self.prediction_horizon:
                continue
            
            # Создаем скользящие окна
            for i in range(len(cnt_data) - self.sequence_length - self.prediction_horizon + 1):
                # Входная последовательность
                seq_data = cnt_data.iloc[i:i + self.sequence_length]
                
                # Целевая последовательность
                target_data = cnt_data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                
                self.sequences.append(seq_data)
                self.targets.append(target_data)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        target_data = self.targets[idx]
        
        # Локальные признаки
        local_features = []
        
        # State encoding
        states = self.state_encoder.transform(seq_data['state'].values)
        local_features.append(states)
        
        # Location encoding
        locations = self.location_encoder.transform(seq_data['location'].fillna('unknown').values)
        local_features.append(locations)
        
        # Container type encoding
        cnt_types = self.cnt_type_encoder.transform(seq_data['cnt_type'].values)
        local_features.append(cnt_types)
        
        # Time delta (дни с начала последовательности)
        time_deltas = np.arange(self.sequence_length)
        local_features.append(time_deltas)
        
        # Объединяем локальные признаки
        local_features = np.column_stack(local_features)
        
        # Глобальные признаки
        global_cols = [col for col in seq_data.columns if col.startswith(('agg_', 'req_', 'future_req_', 'day_of_', 'month', 'year'))]
        global_features = seq_data[global_cols].values
        
        # Нормализация глобальных признаков
        if self.scaler is not None:
            global_features = self.scaler.transform(global_features)
        
        # Целевые значения
        target_states = self.state_encoder.transform(target_data['state'].values)
        target_locations = self.location_encoder.transform(target_data['location'].fillna('unknown').values)
        
        return {
            'local_features': torch.FloatTensor(local_features),
            'global_features': torch.FloatTensor(global_features),
            'target_states': torch.LongTensor(target_states),
            'target_locations': torch.LongTensor(target_locations)
        }

class ContainerTransformer(nn.Module):
    """Трансформер для предсказания состояний контейнеров"""
    
    def __init__(self, 
                 local_vocab_sizes,  # [state_vocab, location_vocab, cnt_type_vocab]
                 local_embed_dims,   # [state_embed, location_embed, cnt_type_embed]
                 global_feature_dim,
                 d_model=256,
                 nhead=8,
                 num_layers=6,
                 prediction_horizon=7):
        super().__init__()
        
        self.prediction_horizon = prediction_horizon
        
        # Embedding слои для локальных признаков
        self.state_embedding = nn.Embedding(local_vocab_sizes[0], local_embed_dims[0])
        self.location_embedding = nn.Embedding(local_vocab_sizes[1], local_embed_dims[1])
        self.cnt_type_embedding = nn.Embedding(local_vocab_sizes[2], local_embed_dims[2])
        self.time_embedding = nn.Linear(1, local_embed_dims[3])  # для time_delta
        
        # Проекция глобальных признаков
        self.global_projection = nn.Linear(global_feature_dim, d_model)
        
        # Общая проекция локальных признаков
        local_embed_total = sum(local_embed_dims)
        self.local_projection = nn.Linear(local_embed_total, d_model)
        
        # Трансформер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Головы предсказания
        self.state_head = nn.Linear(d_model, local_vocab_sizes[0])
        self.location_head = nn.Linear(d_model, local_vocab_sizes[1])
        
    def forward(self, local_features, global_features):
        batch_size, seq_len = local_features.shape[:2]
        
        # Embedding локальных признаков
        state_emb = self.state_embedding(local_features[:, :, 0].long())
        location_emb = self.location_embedding(local_features[:, :, 1].long())
        cnt_type_emb = self.cnt_type_embedding(local_features[:, :, 2].long())
        time_emb = self.time_embedding(local_features[:, :, 3:4].float())
        
        # Объединяем локальные эмбеддинги
        local_emb = torch.cat([state_emb, location_emb, cnt_type_emb, time_emb], dim=-1)
        local_emb = self.local_projection(local_emb)
        
        # Глобальные признаки
        global_emb = self.global_projection(global_features)
        
        # Объединяем локальные и глобальные признаки
        combined_emb = local_emb + global_emb
        
        # Трансформер
        transformer_out = self.transformer(combined_emb)
        
        # Предсказания для каждого шага горизонта
        state_predictions = []
        location_predictions = []
        
        for i in range(self.prediction_horizon):
            # Используем последний выход трансформера для предсказания
            state_pred = self.state_head(transformer_out[:, -1, :])
            location_pred = self.location_head(transformer_out[:, -1, :])
            
            state_predictions.append(state_pred)
            location_predictions.append(location_pred)
        
        return {
            'state_predictions': torch.stack(state_predictions, dim=1),
            'location_predictions': torch.stack(location_predictions, dim=1)
        }

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Обучение модели"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            local_features = batch['local_features'].to(device)
            global_features = batch['global_features'].to(device)
            target_states = batch['target_states'].to(device)
            target_locations = batch['target_locations'].to(device)
            
            outputs = model(local_features, global_features)
            
            # Вычисляем лосс для всех шагов горизонта
            state_loss = 0
            location_loss = 0
            
            for i in range(target_states.shape[1]):
                state_loss += criterion(outputs['state_predictions'][:, i, :], target_states[:, i])
                location_loss += criterion(outputs['location_predictions'][:, i, :], target_locations[:, i])
            
            total_loss = state_loss + location_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                local_features = batch['local_features'].to(device)
                global_features = batch['global_features'].to(device)
                target_states = batch['target_states'].to(device)
                target_locations = batch['target_locations'].to(device)
                
                outputs = model(local_features, global_features)
                
                state_loss = 0
                location_loss = 0
                
                for i in range(target_states.shape[1]):
                    state_loss += criterion(outputs['state_predictions'][:, i, :], target_states[:, i])
                    location_loss += criterion(outputs['location_predictions'][:, i, :], target_locations[:, i])
                
                total_loss = state_loss + location_loss
                val_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def predict_and_aggregate(model, test_loader, state_encoder, location_encoder):
    """Предсказание и агрегация результатов"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            local_features = batch['local_features'].to(device)
            global_features = batch['global_features'].to(device)
            
            outputs = model(local_features, global_features)
            
            # Получаем вероятности
            state_probs = torch.softmax(outputs['state_predictions'], dim=-1)
            location_probs = torch.softmax(outputs['location_predictions'], dim=-1)
            
            predictions.append({
                'state_probs': state_probs.cpu().numpy(),
                'location_probs': location_probs.cpu().numpy()
            })
    
    return predictions

def main():
    """Основная функция"""
    print("Запуск контейнерного предиктора...")
    
    # 1. Обработка данных
    processor = ContainerDataProcessor()
    
    # Создаем календарь
    calendar = processor.create_calendar()
    
    # Обрабатываем транзакции
    daily_snapshots = processor.process_transactions_to_daily_snapshots()
    
    # Добавляем глобальные признаки
    global_features = processor.add_global_features()
    
    # Объединяем данные
    merged_data = processor.merge_local_and_global()
    
    # 2. Подготовка данных для обучения
    print("Подготовка данных для обучения...")
    
    # Кодирование категориальных признаков
    state_encoder = LabelEncoder()
    location_encoder = LabelEncoder()
    cnt_type_encoder = LabelEncoder()
    
    # Обрабатываем пропуски
    merged_data['state'] = merged_data['state'].fillna('unknown')
    merged_data['location'] = merged_data['location'].fillna('unknown')
    
    # Обучаем энкодеры
    state_encoder.fit(merged_data['state'].unique())
    location_encoder.fit(merged_data['location'].unique())
    cnt_type_encoder.fit(merged_data['cnt_type'].unique())
    
    # Нормализация глобальных признаков
    global_cols = [col for col in merged_data.columns if col.startswith(('agg_', 'req_', 'future_req_', 'day_of_', 'month', 'year'))]
    scaler = StandardScaler()
    merged_data[global_cols] = scaler.fit_transform(merged_data[global_cols])
    
    # 3. Создание датасетов
    print("Создание датасетов...")
    
    # Разделяем на train/val/test
    train_data, temp_data = train_test_split(merged_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Создаем датасеты
    train_dataset = ContainerDataset(
        train_data, 
        sequence_length=30, 
        prediction_horizon=7,
        state_encoder=state_encoder,
        location_encoder=location_encoder,
        cnt_type_encoder=cnt_type_encoder,
        scaler=scaler
    )
    
    val_dataset = ContainerDataset(
        val_data, 
        sequence_length=30, 
        prediction_horizon=7,
        state_encoder=state_encoder,
        location_encoder=location_encoder,
        cnt_type_encoder=cnt_type_encoder,
        scaler=scaler
    )
    
    test_dataset = ContainerDataset(
        test_data, 
        sequence_length=30, 
        prediction_horizon=7,
        state_encoder=state_encoder,
        location_encoder=location_encoder,
        cnt_type_encoder=cnt_type_encoder,
        scaler=scaler
    )
    
    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")
    print(f"Test dataset: {len(test_dataset)} sequences")
    
    # 4. Создание модели
    print("Создание модели...")
    
    local_vocab_sizes = [
        len(state_encoder.classes_),
        len(location_encoder.classes_),
        len(cnt_type_encoder.classes_)
    ]
    
    local_embed_dims = [64, 64, 32, 32]  # state, location, cnt_type, time
    global_feature_dim = len(global_cols)
    
    model = ContainerTransformer(
        local_vocab_sizes=local_vocab_sizes,
        local_embed_dims=local_embed_dims,
        global_feature_dim=global_feature_dim,
        d_model=256,
        nhead=8,
        num_layers=6,
        prediction_horizon=7
    ).to(device)
    
    print(f"Модель создана. Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Обучение
    print("Начало обучения...")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=50, learning_rate=0.001
    )
    
    # 6. Предсказание и агрегация
    print("Предсказание и агрегация...")
    
    predictions = predict_and_aggregate(model, test_loader, state_encoder, location_encoder)
    
    print("Обучение завершено!")
    print(f"Финальная train loss: {train_losses[-1]:.4f}")
    print(f"Финальная val loss: {val_losses[-1]:.4f}")
    
    # Сохраняем модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_encoder': state_encoder,
        'location_encoder': location_encoder,
        'cnt_type_encoder': cnt_type_encoder,
        'scaler': scaler,
        'local_vocab_sizes': local_vocab_sizes,
        'local_embed_dims': local_embed_dims,
        'global_feature_dim': global_feature_dim
    }, 'container_transformer_model.pth')
    
    print("Модель сохранена в container_transformer_model.pth")

if __name__ == "__main__":
    main()
