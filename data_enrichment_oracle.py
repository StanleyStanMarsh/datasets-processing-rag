import os
import random
import numpy as np
import pandas as pd
import faiss
import torch

from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================

TOP_K = 200  # больше, чтобы не потерять кандидатов
PCA_VARIANCE = 0.95
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

OUTPUT_FILE = "hn_pca_ensemble_dataset.csv"

# =========================================================
# LOAD DATA
# =========================================================

first_df = pd.read_csv("attackqa_parquet_human_answered.csv")
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
# 1. ENSEMBLE MODELS (E1..E6)
# =========================================================

model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "BAAI/bge-base-en-v1.5",
]

CACHE_DIR = "./hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
models = [SentenceTransformer(m, device=device, cache_folder=CACHE_DIR) for m in model_names]

def encode_ensemble(texts):
    embs = []
    for model in models:
        e = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=256,
            show_progress_bar=False
        ).astype("float32")
        embs.append(e)
    return np.concatenate(embs, axis=1)  # X_concat

print("Encoding questions...")
q_concat = encode_ensemble(questions)

print("Encoding answers...")
a_concat = encode_ensemble(answers)

# =========================================================
# 2. PCA REDUCTION (95% variance)
# =========================================================

print("Fitting PCA...")
pca = PCA(n_components=PCA_VARIANCE, random_state=SEED)

q_pca = np.ascontiguousarray(pca.fit_transform(q_concat).astype('float32'))
a_pca = np.ascontiguousarray(pca.transform(a_concat).astype('float32'))

# normalize for cosine similarity
faiss.normalize_L2(q_pca)
faiss.normalize_L2(a_pca)

print("PCA dim:", q_pca.shape[1])

# =========================================================
# 3. FAISS INDEX (documents)
# =========================================================

dim = a_pca.shape[1]
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(res, dim)
index.add(a_pca)

print("FAISS index built:", index.ntotal)

# =========================================================
# 4. HARD + SOFT NEGATIVE MINING
# =========================================================

rows = []
accepted = 0
rejected = 0

for i in range(len(df_all)):

    q = questions[i]
    pd_pos = answers[i]

    q_vec = q_pca[i].reshape(1, -1)
    pd_vec = a_pca[i].reshape(1, -1)

    # sim(Q, PD)
    sim_q_pd = np.dot(q_vec, pd_vec.T).item()

    # retrieve candidates
    sims, idxs = index.search(q_vec, TOP_K)

    hard_candidates = []

    best_hard = None
    best_hard_score = -1e9

    farthest_doc = None
    farthest_score = 1e9

    for sim_q_d, idx in zip(sims[0], idxs[0]):

        if idx == i:
            continue

        d_vec = a_pca[idx].reshape(1, -1)

        sim_pd_d = sim_pd_d = np.dot(pd_vec, d_vec.T).item()

        # =====================================================
        # HARD NEGATIVE CONDITIONS (STRICT Eq.5 + Eq.6)
        # =====================================================

        cond1 = sim_q_d > sim_q_pd      # distance form -> higher similarity is closer
        cond2 = sim_q_d > sim_pd_d

        if cond1 and cond2:

            # pick BEST hard negative (closest to Q)
            if sim_q_d > best_hard_score:
                best_hard_score = sim_q_d
                best_hard = idx

        # =====================================================
        # SOFT NEGATIVE (farthest from query)
        # =====================================================

        if sim_q_d < farthest_score:
            farthest_score = sim_q_d
            farthest_doc = idx

    if best_hard is None:
        rejected += 1
        continue

    hard_negative = answers[best_hard]
    soft_negative = answers[farthest_doc]

    rows.append({
        "question": q,
        "positive": pd_pos,
        "hard_negative": hard_negative,
        "soft_negative": soft_negative,
        "sim_q_pd": sim_q_pd,
        "hard_neg_sim": best_hard_score,
        "soft_neg_sim": farthest_score,
    })

    accepted += 1

# =========================================================
# SAVE DATASET
# =========================================================

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_FILE, index=False)

# =========================================================
# STATS
# =========================================================

print("\n==============================")
print("FINISHED")
print("==============================")

print("Accepted:", accepted)
print("Rejected:", rejected)
print("Coverage:", accepted / len(df_all) * 100)

print("Saved:", OUTPUT_FILE)
print(result_df.head())