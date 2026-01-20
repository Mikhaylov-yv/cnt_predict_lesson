"""
Скрипт преобразования набора данных cnt_events.csv

Создает датасет для ML модели прогнозирования дней до прибытия порожнего контейнера.
Базовая логика:
- Для каждого cnt_id выделяются завершённые циклы: от "Передан_на_погрузку" до "Прибыл_свободным".
- На уровне событий формируется последовательность событий внутри цикла.
- Опционально строится daily-датасет (1 строка = 1 календарный день цикла) с корректной логикой:
    * event_age_days (days_in_current_event) обнуляется при смене события и растёт ежедневно.
    * prev_event_duration_days хранит длительность предыдущего события (константа внутри текущего события).
"""

import pandas as pd
import numpy as np


START_EVENT = "Передан_на_погрузку"
END_EVENT = "Прибыл_свободным"
EVENT_MAP = {
    "Передан_на_погрузку": 1,  
    "Прибыл_груженый": 2,
    "Отправлен_по_ЖД": 3, 
    "Прибыл_по_ЖД": 4, 
    "Отправлен_грузополучателю": 5,
    "Прибыл_свободным": 6
}

def find_sequences(container_events: pd.DataFrame):
    """
    Находит все завершённые циклы для одного контейнера.

    Цикл:
      START_EVENT -> ... -> END_EVENT

    Возвращает список dict:
      {start_idx, end_idx, end_date, end_location}
    """
    sequences = []
    start_idx = None

    for i, row in container_events.iterrows():
        event = row["event"]

        if event == START_EVENT:
            start_idx = i

        if start_idx is not None and event == END_EVENT:
            sequences.append(
                {
                    "start_idx": start_idx,
                    "end_idx": i,
                    "end_date": row["dt"],
                    "end_location": row.get("location_name", None),
                }
            )
            start_idx = None

    return sequences


def build_event_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит датасет на уровне событий:
    1 строка = 1 событие внутри завершённого цикла.

    Важно:
    - days_between_events: разница в днях между текущим и предыдущим событием внутри цикла.
      (это НЕ "дни в текущем событии", это разрыв между событиями)
    - days_until_free: дни до END_EVENT в рамках цикла.
    """
    result_rows = []

    for cnt_id, container_events in df.groupby("cnt_id"):
        container_events = container_events.sort_values(["dt", "original_index"])

        sequences = find_sequences(container_events)
        if not sequences:
            continue

        for seq_num, seq in enumerate(sequences, start=1):
            cycle_id = f"{cnt_id}_cycle_{seq_num}"

            cycle_events = container_events.loc[seq["start_idx"] : seq["end_idx"]].copy()
            cycle_events = cycle_events.sort_values(["dt", "original_index"])

            cycle_end_date = seq["end_date"]
            end_location_name = seq["end_location"]

            for i, row in cycle_events.reset_index(drop=True).iterrows():
                if i == 0:
                    days_between = 0
                else:
                    prev_dt = cycle_events.reset_index(drop=True).iloc[i - 1]["dt"]
                    days_between = (row["dt"] - prev_dt).days

                # days_until_free считаем в днях по календарю
                if row["dt"].date() == cycle_end_date.date():
                    days_until_free = 0
                else:
                    days_until_free = (cycle_end_date - row["dt"]).days

                result_rows.append(
                    {
                        "dt": row["dt"],
                        "cnt_id": row["cnt_id"],
                        "cycle_id": cycle_id,
                        "event": row["event"],
                        "location_name": row.get("location_name", None),
                        "end_location_name": end_location_name,
                        "cnt_type": row.get("cnt_type", None),
                        # gap между событиями (event-level)
                        "days_between_events": int(days_between),
                        "days_until_free": int(days_until_free),
                    }
                )

    result_df = pd.DataFrame(result_rows)
    if result_df.empty:
        return result_df

    return result_df.sort_values(["cycle_id", "dt"]).reset_index(drop=True)

def expand_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Превращает event-level датасет в daily-датасет:
    - 1 строка = 1 календарный день внутри цикла (от первого события цикла до END_EVENT включительно)
    - event / location / end_location берутся из "текущего" события и тянутся до следующего события

    Дальше пересчитываются корректные фичи:
    - event_age_days (days_in_current_event): 0 в день смены события, затем 1,2,3...
    - prev_event_duration_days: длительность предыдущего события (константа внутри текущего события)
    - days_until_free: пересчитывается как (end_date - dt).days по календарю (устойчиво к любым сдвигам)
    """

    df = df.copy()

    df["event_code"] = df["event"].map(EVENT_MAP)

    # Важно: в daily-логике работаем по дням
    df["dt"] = pd.to_datetime(df["dt"]).dt.normalize()
    df = df.sort_values(["cycle_id", "dt"])

    expanded_cycles = []

    for cycle_id, g in df.groupby("cycle_id", sort=False):
        g = g.sort_values("dt")

        parts = [g]

        # 1) Вставляем пропущенные дни между событиями, копируя "текущий" статус
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]

            delta_days = (nxt["dt"] - cur["dt"]).days
            if delta_days <= 1:
                continue

            add = pd.DataFrame([cur.to_dict()] * (delta_days - 1))
            add["dt"] = [cur["dt"] + pd.Timedelta(days=d) for d in range(1, delta_days)]
            add["days_until_free"] = cur["days_until_free"] - np.arange(1, delta_days)

            parts.append(add)

        g_exp = pd.concat(parts, ignore_index=True).sort_values("dt")

        # 2) Пересчитываем возраст текущего события (0 в день смены event_code)
        # Новый сегмент начинается если:
        #   - event_code сменился, или
        #   - есть разрыв по дням (на всякий случай)
        day_gap = g_exp["dt"].diff().dt.days.fillna(1)
        new_segment = (g_exp["event_code"] != g_exp["event_code"].shift(1)) | (day_gap != 1)
        seg_id = new_segment.cumsum()

        g_exp["event_age_days"] = g_exp.groupby(seg_id).cumcount()


        # 3) (Опционально, но полезно) длительность предыдущего события
        seg_len = g_exp.groupby(seg_id)["dt"].size()
        prev_len = seg_len.shift(1)
        g_exp["prev_event_duration_days"] = seg_id.map(prev_len).fillna(0).astype(int)

        expanded_cycles.append(g_exp)

    daily = pd.concat(expanded_cycles, ignore_index=True).sort_values(["cycle_id", "dt"]).drop(columns=["days_between_events"])
    return daily




def transform_cnt_events_v2(
    input_file: str = "data/cnt_events.csv",
    output_file: str = "data/cnt_events_transformed_v2.csv",
    make_daily: bool = False,
    daily_output_file: str | None = None,
):
    """
    Основная функция.

    make_daily=False:
      - сохраняет event-level датасет в output_file

    make_daily=True:
      - сохраняет event-level датасет в output_file
      - строит daily-датасет и сохраняет в daily_output_file (или рядом с output_file)
    """
    print("Загрузка данных...")
    df = pd.read_csv(input_file)

    # Базовые проверки
    required_cols = {"cnt_id", "dt", "event"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Входной файл не содержит обязательные колонки: {missing}")

    df["dt"] = pd.to_datetime(df["dt"])
    df["original_index"] = np.arange(len(df))

    df = df.sort_values(["cnt_id", "dt", "original_index"]).reset_index(drop=True)

    print("Формирование event-level датасета по циклам...")
    event_df = build_event_level_dataset(df)

    print(f"Событий в результате: {len(event_df):,}")
    if not event_df.empty:
        print(f"Контейнеров: {event_df['cnt_id'].nunique():,}")
        print(f"Циклов: {event_df['cycle_id'].nunique():,}")
        print(f"Период: {event_df['dt'].min().date()} — {event_df['dt'].max().date()}")
        print("\nРаспределение событий (top-20):")
        print(event_df["event"].value_counts().head(20))

    event_df.to_csv(output_file, index=False)
    print(f"\nEvent-level датасет сохранён в {output_file}")

    if make_daily:
        if daily_output_file is None:
            if output_file.endswith(".csv"):
                daily_output_file = output_file.replace(".csv", "_daily.csv")
            else:
                daily_output_file = output_file + "_daily.csv"

        print("\nРасширение до daily-датасета и пересчёт корректных фич...")
        daily_df = expand_to_daily(event_df)

        print(f"Daily строк: {len(daily_df):,}")
        daily_df.to_csv(daily_output_file, index=False)
        print(f"Daily датасет сохранён в {daily_output_file}")

        return event_df, daily_df

    return event_df


if __name__ == "__main__":
    # Пример:
    transform_cnt_events_v2(make_daily=True)
    # transform_cnt_events_v2()
