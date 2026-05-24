import random

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# =========================================================
# 1. LOAD DATASETS
# =========================================================

first_df = pd.read_csv("attackqa_parquet_human_answered.csv")
second_df = pd.read_csv("cybersecurityQAAdataset_only_reviewed_by_expert.csv")
third_df = pd.read_csv("cybersecurity_qa_updated_column_names.csv")

# =========================================================
# 2. MERGE DATASETS
# =========================================================

df_all = pd.concat(
    [
        first_df[["question", "answer"]],
        second_df[["question", "answer"]],
        third_df[["question", "answer"]],
    ],
    ignore_index=True,
)

# Remove duplicates
df_all = df_all.drop_duplicates().reset_index(drop=True)

print(f"Total QA pairs: {len(df_all)}")

# =========================================================
# 3. LOAD EMBEDDING MODEL
# =========================================================

model = SentenceTransformer("./models/FALCON")

# =========================================================
# 4. ENCODE QUESTIONS (NOT ANSWERS!)
# =========================================================

questions = df_all["question"].tolist()
answers = df_all["answer"].tolist()

question_embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=32,
).astype("float32")

# =========================================================
# 5. NORMALIZE EMBEDDINGS
# =========================================================
# Needed for cosine similarity with IndexFlatIP

faiss.normalize_L2(question_embeddings)

# =========================================================
# 6. BUILD FAISS INDEX
# =========================================================

dim = question_embeddings.shape[1]

# Inner Product + normalized vectors = cosine similarity
index = faiss.IndexFlatIP(dim)

index.add(question_embeddings)

print(f"FAISS index size: {index.ntotal}")

# =========================================================
# 7. HARD NEGATIVE MINING
# =========================================================

hard_negatives = []
random_negatives = []

# Larger k = better hard negatives
TOP_K = 20

for i, row in df_all.iterrows():

    query_question = row["question"]
    true_answer = row["answer"]

    # Encode current question
    q_emb = model.encode(
        [query_question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(q_emb)

    # Search similar QUESTIONS
    scores, indices = index.search(q_emb, TOP_K)

    candidate_pool = []

    # =====================================================
    # Collect hard negative candidates
    # =====================================================

    for idx in indices[0]:

        # Skip itself
        if idx == i:
            continue

        candidate_question = questions[idx]
        candidate_answer = answers[idx]

        # Skip identical answers
        if candidate_answer == true_answer:
            continue

        # Skip exact same question
        if candidate_question == query_question:
            continue

        candidate_pool.append(candidate_answer)

    # =====================================================
    # Select hard negative
    # =====================================================

    if candidate_pool:

        # Random from top similar candidates
        # prevents deterministic easy negatives
        hard_negative = random.choice(candidate_pool[:5])

    else:
        # Fallback
        all_other_answers = [
            ans for ans in answers
            if ans != true_answer
        ]

        hard_negative = random.choice(all_other_answers)

    hard_negatives.append(hard_negative)

    # =====================================================
    # Random negative (easy negative)
    # =====================================================

    while True:

        random_negative = random.choice(answers)

        if (
            random_negative != true_answer
            and random_negative != hard_negative
        ):
            break

    random_negatives.append(random_negative)

# =========================================================
# 8. SAVE ENRICHED DATASET
# =========================================================

df_all["hard_negative"] = hard_negatives
df_all["negative"] = random_negatives

OUTPUT_FILE = "qa_cross_mined_dataset.csv"

df_all[
    ["question", "answer", "hard_negative", "negative"]
].to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved dataset: {OUTPUT_FILE}")

print("\nExample rows:\n")
print(df_all.head())