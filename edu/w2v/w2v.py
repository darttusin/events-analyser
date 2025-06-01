# train_word2vec.py
import pandas as pd
import pyarrow.parquet as pq
from gensim.models import Word2Vec
import os


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
    "final_data.parquet", columns=["domain", "user_id"]
)

# Группировка: пользователь -> список доменов
user_sequences = df.groupby("user_id")["domain"].apply(list)

sentences = [[str(domain) for domain in domains] for domains in user_sequences.tolist()]

# Обучаем Word2Vec
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=64,
    window=5,
    min_count=2,
    workers=4,
    seed=42,
    sg=1  # skip-gram
)

# Сохраняем модель
os.makedirs("models", exist_ok=True)
w2v_model.save("models/domain_w2v.model")
print("Word2Vec model saved to models/domain_w2v.model")
