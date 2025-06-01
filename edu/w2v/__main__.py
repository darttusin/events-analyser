import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score
import pyarrow.parquet as pq


def read_parquet_skip_broken_rows(parquet_file, columns, batch_size=400_000, rows=2_000_000):
    pf = pq.ParquetFile(parquet_file)
    batches = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        try:
            df = batch.to_pandas()
            batches.append(df)
        except Exception as e:
            print(f"Skipping a batch due to error: {e}")
        if len(batches) * batch_size > rows:
            break
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


# Загрузка данных
df = read_parquet_skip_broken_rows(
    "final_data.parquet",
    columns=["target", "_seq_len", "domain", "user_id"]
)

# Группируем домены по пользователю
user_sequences = df.groupby("user_id")["domain"].apply(list).apply(lambda x: [i for sub in x for i in sub])

# Объединяем с target
targets = df.groupby("user_id")["target"].first()

# Обучение Word2Vec
sentences = user_sequences.tolist()
w2v_model = Word2Vec(
    vector_size=64,
    window=5,
    min_count=2,
    workers=4,
    seed=42,
    sg=1
)

w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=5)
w2v_model = Word2Vec(sentences=sentences, vector_size=64, window=5, min_count=2, workers=30, seed=42, sg=1)

# Функция для получения среднего вектора
def get_user_vector(seq, model):
    vectors = [model.wv[word] for word in seq if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Признаки
X = np.vstack([get_user_vector(seq, w2v_model) for seq in user_sequences])
y = targets.values

# Трен/тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Обучение CatBoost
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    verbose=50,
    random_seed=42
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)

# Метрики
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
