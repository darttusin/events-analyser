# Анализ пользовательской активности с применением методов машинного обучения для оценки эффективности рекламных предложений

Данный проект представляет собой систему анализа пользовательского поведения с применением машинного обучения. Решение построено с использованием моделей LSTM / BERT / Word2Vec и развернуто с помощью FastAPI + MLServer в Docker-контейнерах.

---

## 📁 Структура проекта

<pre>
├── 📂 data/
│   ├── 📂 dataset/                 # Исходные (сырые) данные
│   │   ├── 📄 clicks.csv           # Клики пользователей
│   │   ├── 📄 events.csv           # События и действия
│   │   └── 📄 offers.csv           # Предложенные офферы
│   │
│   ├── 📄 train.parquet            # Обучающий датасет
│   ├── 📄 test.parquet             # Тестовый датасет
│   └── 📄 final_data.parquet       # Финальный датасет
│
├── 📂 edu/                         # Код для обучения моделей
│   ├── 📂 bert/                    # Модель BERT + BERT MLM CatBoost
│   ├── 📂 coles/                   # CoLES
│   ├── 📂 rnn/                     # RNN-модели: GRU и LSTM
│   └── 📂 w2v/                     # Word2Vec + CatBoost классификатор
│
├── 📂 inference/                   # FastAPI-приложение (веб API)
│   ├── 🐋 Dockerfile               # Сборка образа с API
│   ├── 📄 main.py                  # Код приложения
│   └── 📄 requirements.txt         # Зависимости для API
│
├── 📂 mlserver/                    # MLServer-контейнер для модели
│   ├── 🐋 Dockerfile               # Docker для MLServer
│   └── 📄 requirements.txt         # Зависимости MLServer
│
├── 📂 model/                       # Финальная модель и связанные файлы
│   ├── 📂 data/                    # Вспомогательные данные для модели
│   ├── 📄 model.pt                 # Сохранённая модель
│   ├── 📄 tokenizer.json           # Токенизатор
│   ├── 📄 model-settings.json      # Конфигурация модели для MLServer
│   └── 📄 model.py                 # Кастомный класс модели
│
├── 📂 notebooks/                   # Jupyter-ноутбуки (EDA)
│   └── 📓 EDA.ipynb                # Анализ данных и визуализация
│
├── 📂 preprocess/                  # Скрипты предобработки данных
│   ├── ⚙️ prepare_events.py        # Обработка событий пользователей
│   ├── ⚙️ prepare_offers.py        # Обработка рекламных предложений
│   └── ⚙️ prepare_seq_parquet.py   # Сериализация последовательностей
│
├── 🐋 docker-compose.yml           # Compose для сервисов (FastAPI + MLServer)
└── 📄 README.md                    # Документация проекта
</pre>

---

## 🚀 Запуск

```bash
docker compose up --build
```

После запуска Swagger приложения будет доступен на: http://localhost:8000/api/docs

## 🧪 Эндпоинты FastAPI
### Запрос

```http
POST /api/v1/score
Content-Type: application/json

[
    {
        "timestamp": "2025-05-25T14:20:00",
        "domain": "example.com"
    }
]
```

### Ответ

```json
{
  "score": 0.87
}
```