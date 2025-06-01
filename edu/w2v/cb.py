# train_catboost.py
import pandas as pd
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from itertools import product


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
    "final_data.parquet", columns=["domain", "user_id", "target"]
)

# Загрузка обученной Word2Vec модели
w2v_model = Word2Vec.load("models/domain_w2v.model")
embedding_dim = w2v_model.vector_size

# Группируем домены по пользователям
user_sequences = df.groupby("user_id").agg({
    "domain": list,
    "target": "first"  # предполагаем, что у каждого user_id одна цель
}).reset_index()

# Вычисляем эмбеддинги пользователей как среднее эмбеддингов доменов
def get_user_embedding(domains):
    vectors = [w2v_model.wv[str(d)] for d in domains if str(d) in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim)

user_sequences["features"] = user_sequences["domain"].apply(get_user_embedding)

# Формируем матрицу признаков и метки
X = np.stack(user_sequences["features"].values)
y = user_sequences["target"].values

# Разделение на трейн/тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# === Гиперпараметры для перебора ===
param_grid = {
    "depth": [2, 4, 6, 8],
    "learning_rate": [0.001, 0.01, 0.1],
    "l2_leaf_reg": [1, 3, 5],
    "iterations": [500, 1000, 2000]
}

results = []

# === Перебор всех комбинаций параметров ===
for depth, learning_rate, l2_leaf_reg, iterations in product(
    param_grid["depth"],
    param_grid["learning_rate"],
    param_grid["l2_leaf_reg"],
    param_grid["iterations"]
):
    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        iterations=iterations,
        verbose=0,
        thread_count=30,
        random_seed=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='macro')

    results.append({
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "iterations": iterations,
        "roc_auc": roc,
        "f1_micro": f1
    })

    print(depth, learning_rate, l2_leaf_reg, iterations, roc, f1)

# === Печать таблицы результатов ===
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="roc_auc", ascending=False).to_string(index=False))