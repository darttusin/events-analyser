# train_catboost_with_bert.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
from itertools import product

# === Функция для чтения parquet ===
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

# === Загрузка данных ===
df = read_parquet_skip_broken_rows("final_data.parquet", columns=["domain", "user_id", "target"])

# === Группируем домены по пользователям ===
user_sequences = df.groupby("user_id").agg({
    "domain": list,
    "target": "first"
}).reset_index()

# === Загрузка BERT-модели и токенизатора ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()
bert_model = bert_model.to("cuda" if torch.cuda.is_available() else "cpu")

# === Получение CLS-эмбеддинга для пользователя ===
def get_bert_embedding(domains):
    text = " ".join([str(d) for d in domains])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=200)
    inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

def get_bert_embeddings_batch(texts, tokenizer, model, device, batch_size=128, max_length=200):
    all_embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS токен
        all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)


texts = [" ".join([str(d) for d in domains]) for domains in tqdm(user_sequences["domain"])]
device = "cuda:1" if torch.cuda.is_available() else "cpu"
bert_model = bert_model.to(device)

features = get_bert_embeddings_batch(texts, tokenizer, bert_model, device, 256)

X = np.stack(features)
y = user_sequences["target"].values

# === Разделение на трейн/тест ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# === Гиперпараметры ===
param_grid = {
    "depth": [4, 6],
    "learning_rate": [0.01, 0.1],
    "l2_leaf_reg": [1, 3],
    "iterations": [500, 1000, 2000]
}

results = []

# === Обучение CatBoost с перебором параметров ===
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
        thread_count=16,
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
        "f1_macro": f1
    })

    print(depth, learning_rate, l2_leaf_reg, iterations, roc, f1)

# === Вывод результатов ===
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="roc_auc", ascending=False).to_string(index=False))
