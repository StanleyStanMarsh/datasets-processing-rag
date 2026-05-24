import random

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

TOP_K = 50

# how close candidate should be to positive-query similarity
EPSILON = 0.05

# how much closer candidate should be to query than to positive
DELTA = 0.05

OUTPUT_FILE = "strict_geometry_aware_dataset.csv"

# =========================================================
# LOAD DATASETS
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
)

df_all = df_all.drop_duplicates().reset_index(drop=True)

print(f"Total QA pairs: {len(df_all)}")

# =========================================================
# MODEL
# =========================================================

model = SentenceTransformer("./models/all-MiniLM-L6-v2")

questions = df_all["question"].tolist()
answers = df_all["answer"].tolist()

# =========================================================
# ENCODE QUESTIONS + ANSWERS
# =========================================================

question_embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=32,
).astype("float32")

answer_embeddings = model.encode(
    answers,
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=32,
).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(question_embeddings)
faiss.normalize_L2(answer_embeddings)

# =========================================================
# BUILD ANSWER INDEX
# =========================================================

dim = answer_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(answer_embeddings)

print(f"FAISS index size: {index.ntotal}")

# =========================================================
# STRICT HARD NEGATIVE MINING
# =========================================================

rows = []

accepted = 0
rejected = 0

for i in range(len(df_all)):

    query = questions[i]
    positive_answer = answers[i]

    q_emb = question_embeddings[i].reshape(1, -1)
    p_emb = answer_embeddings[i].reshape(1, -1)

    # similarity(query, positive)
    sim_q_p = float(np.dot(q_emb, p_emb.T))

    # retrieve nearest answers
    sims, indices = index.search(q_emb, TOP_K)

    candidate_pool = []

    for sim_q_d, idx in zip(sims[0], indices[0]):

        # skip itself
        if idx == i:
            continue

        candidate_answer = answers[idx]

        # skip identical answer
        if candidate_answer == positive_answer:
            continue

        d_emb = answer_embeddings[idx].reshape(1, -1)

        # similarity(positive, candidate)
        sim_p_d = float(np.dot(p_emb, d_emb.T))

        # =================================================
        # PAPER CONDITIONS
        # =================================================

        cond1 = sim_q_d > (sim_q_p - EPSILON)

        cond2 = sim_q_d > (sim_p_d + DELTA)

        if cond1 and cond2:

            candidate_pool.append(
                {
                    "hard_negative": candidate_answer,
                    "sim_q_p": float(sim_q_p),
                    "sim_q_d": float(sim_q_d),
                    "sim_p_d": float(sim_p_d),
                }
            )

    # =====================================================
    # STRICT FILTER:
    # ONLY KEEP VALID HARD NEGATIVES
    # =====================================================

    if not candidate_pool:
        rejected += 1
        continue

    selected = random.choice(candidate_pool)

    hard_negative = selected["hard_negative"]

    # =====================================================
    # EASY NEGATIVE
    # =====================================================

    while True:

        random_negative = random.choice(answers)

        if (
            random_negative != positive_answer
            and random_negative != hard_negative
        ):
            break

    rows.append(
        {
            "question": query,
            "answer": positive_answer,
            "hard_negative": hard_negative,
            "negative": random_negative,
            "sim_q_p": selected["sim_q_p"],
            "sim_q_d": selected["sim_q_d"],
            "sim_p_d": selected["sim_p_d"],
        }
    )

    accepted += 1

# =========================================================
# SAVE DATASET
# =========================================================

result_df = pd.DataFrame(rows)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================================================
# STATS
# =========================================================

print("\n=====================================================")
print("MINING FINISHED")
print("=====================================================")

print(f"Accepted samples: {accepted}")
print(f"Rejected samples: {rejected}")

coverage = accepted / len(df_all) * 100

print(f"Coverage: {coverage:.2f}%")

print(f"\nSaved dataset: {OUTPUT_FILE}")

print("\nExample rows:\n")
print(result_df.head())