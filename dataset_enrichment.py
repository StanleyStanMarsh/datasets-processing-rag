import pandas as pd
import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

first_df = pd.read_csv("attackqa_parquet_human_answered.csv")
second_df = pd.read_csv("cybersecurityQAAdataset_only_reviewed_by_expert.csv")
third_df = pd.read_csv("cybersecurity_qa_updated_column_names.csv")

# 1. Объединяем все датафреймы
df_all = pd.concat([first_df[['question', 'answer']], second_df[['question', 'answer']], third_df[['question', 'answer']]], ignore_index=True)

# 2. Список всех ответов (документов)
all_answers = df_all['answer'].tolist()

# 3. Модель для эмбеддингов (можно заменить на свою)
model = SentenceTransformer('./models/FALCON')

# Получаем векторы всех ответов
answer_embeddings = model.encode(
    all_answers, convert_to_numpy=True, show_progress_bar=True, batch_size=1
).astype('float32')

# 4. Строим FAISS индекс (L2-расстояние)
dim = answer_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(answer_embeddings)

print(f"Индекс построен, количество документов: {index.ntotal}")

# 5. Обрабатываем каждую пару (question, answer)
hard_negatives = []
negatives = []
k = 3  # количество ближайших соседей, чтобы пропустить точное совпадение

for i, row in df_all.iterrows():
    question = row['question']
    true_answer = row['answer']

    # Эмбеддинг вопроса
    q_emb = model.encode([question], convert_to_numpy=True).astype('float32')
    # Поиск ближайших ответов
    distances, indices = index.search(q_emb, k)

    # --- Hard negative ---
    hard_neg = None
    for idx in indices[0]:
        candidate = all_answers[idx]
        if candidate != true_answer:
            hard_neg = candidate
            break

    # Если все top-k совпали с истинным ответом (маловероятно), берём случайный другой ответ
    if hard_neg is None:
        # любой ответ, не равный true_answer
        candidates = [ans for ans in all_answers if ans != true_answer]
        hard_neg = random.choice(candidates) if candidates else true_answer  # fallback

    hard_negatives.append(hard_neg)

    # --- Random negative ---
    # Выбираем случайный ответ, исключая true_answer и hard_neg
    while True:
        rand_neg = random.choice(all_answers)
        if rand_neg != true_answer and rand_neg != hard_neg:
            break
    negatives.append(rand_neg)

# 6. Добавляем результаты в общий датафрейм
df_all['hard_negative'] = hard_negatives
df_all['negative'] = negatives

# # 7. (Опционально) Разделяем обратно на три датафрейма
# # Допустим, у нас сохранены исходные размеры:
# n1 = len(first_df)
# n2 = len(second_df)
# n3 = len(third_df)

# first_aug = df_all.iloc[:n1].reset_index(drop=True)
# second_aug = df_all.iloc[n1:n1+n2].reset_index(drop=True)
# third_aug = df_all.iloc[n1+n2:].reset_index(drop=True)

# # Теперь first_aug, second_aug, third_aug содержат дополнительные столбцы
# print(first_aug[['question', 'answer', 'hard_negative', 'negative']].head())

df_all[['question', 'answer', 'hard_negative', 'negative']].to_csv("enriched_dataset.csv")

print(df_all.head())