import os
import random
import numpy as np
import pandas as pd
import faiss
import torch

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

TOP_K = 20       # Количество кандидатов для поиска похожих вопросов
HARD_NEG_TOP_N = 5  # Из скольких лучших кандидатов выбирать случайный hard negative
SEED = 42

# Модель с Hugging Face. Можно заменить на любую другую, поддерживаемую sentence-transformers
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "./hf_cache"

random.seed(SEED)
np.random.seed(SEED)

OUTPUT_FILE = "qa_cross_mined_dataset.csv"

# =========================================================
# LOAD DATA
# =========================================================

first_df = pd.read_csv("attackqa_parquet_processed_all.csv")
second_df = pd.read_csv("cybersecurityQAAdataset_only_reviewed_by_expert.csv")
third_df = pd.read_csv("cybersecurity_qa_updated_column_names.csv")

df_all = pd.concat(
    [
        first_df[["question", "answer"]],
        second_df[["question", "answer"]],
        third_df[["question", "answer"]],
    ],
    ignore_index=True,
).drop_duplicates().reset_index(drop=True)

questions = df_all["question"].tolist()
answers = df_all["answer"].tolist()

print(f"Total QA pairs: {len(df_all)}")

# =========================================================
# 1. LOAD MODEL
# =========================================================

os.makedirs(CACHE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device, cache_folder=CACHE_DIR)

def encode_texts(texts):
    """
    Кодирует тексты, возвращает numpy array float32.
    """
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True
    ).astype("float32")
    return embs

# =========================================================
# 2. ENCODE QUESTIONS & BUILD INDEX
# =========================================================

print("Encoding questions...")
q_embeddings = encode_texts(questions)

# Normalize for cosine similarity (IndexFlatIP with normalized vectors)
faiss.normalize_L2(q_embeddings)

dim = q_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(q_embeddings)

print("FAISS index built:", index.ntotal)

# =========================================================
# 3. HARD + RANDOM NEGATIVE MINING
# =========================================================

hard_negatives = []
random_negatives = []

accepted = 0
rejected = 0

# Предварительное вычисление списка всех ответов для быстрого доступа
all_answers_set = set(answers)

for i in range(len(df_all)):
    
    q_text = questions[i]
    pos_answer = answers[i]
    
    # Получаем эмбеддинг текущего вопроса из предварительно рассчитанного массива
    q_vec = q_embeddings[i].reshape(1, -1)
    
    # Search similar QUESTIONS
    sims, idxs = index.search(q_vec, TOP_K)
    
    candidate_pool = []
    
    for sim_score, idx in zip(sims[0], idxs[0]):
        
        # Skip itself
        if idx == i:
            continue
            
        cand_question = questions[idx]
        cand_answer = answers[idx]
        
        # Skip identical answers (positive case)
        if cand_answer == pos_answer:
            continue
            
        # Skip exact same question text (redundant check if IDs are unique, but safe)
        if cand_question == q_text:
            continue
            
        candidate_pool.append(cand_answer)
        
    # =====================================================
    # Select Hard Negative
    # =====================================================
    
    if candidate_pool:
        # Берем топ-N кандидатов и выбираем случайно среди них
        # Если кандидатов меньше чем HARD_NEG_TOP_N, берем всех доступных
        top_n_candidates = candidate_pool[:HARD_NEG_TOP_N]
        hard_neg = random.choice(top_n_candidates)
    else:
        # Fallback: если похожих вопросов с другими ответами не нашлось
        # Выбираем случайный ответ, отличный от правильного
        other_answers = [ans for ans in all_answers_set if ans != pos_answer]
        if other_answers:
            hard_neg = random.choice(other_answers)
        else:
            # Крайний случай (весь датасет из одинаковых ответов?)
            hard_neg = pos_answer 
            rejected += 1
            
    hard_negatives.append(hard_neg)
    
    # =====================================================
    # Select Random (Easy) Negative
    # =====================================================
    
    while True:
        rand_neg = random.choice(answers)
        if rand_neg != pos_answer and rand_neg != hard_neg:
            break
            
    random_negatives.append(rand_neg)
    accepted += 1

# =========================================================
# 4. SAVE DATASET
# =========================================================

df_all["hard_negative"] = hard_negatives
df_all["negative"] = random_negatives

result_df = df_all[["question", "answer", "hard_negative", "negative"]]
result_df.rename(columns={"answer": "positive", "negative": "soft_negative"}).to_csv(OUTPUT_FILE, index=False)

# =========================================================
# STATS
# =========================================================

print("\n==============================")
print("FINISHED")
print("==============================")
print("Accepted:", accepted)
print("Rejected (fallback used):", rejected)
print("Saved:", OUTPUT_FILE)
print(result_df.head())