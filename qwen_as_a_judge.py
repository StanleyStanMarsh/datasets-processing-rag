import os
import pandas as pd
import requests
import json
from tqdm import tqdm

VLLM_URL = "http://localhost:8000/v1/chat/completions"
SMALL_MODEL_NAME = "Qwen2.5-3B-Instruct"
MODEL = "Qwen/" + SMALL_MODEL_NAME


output_lines = []

def log(line=""):
    print(line)
    output_lines.append(line)

# ----------------------------
# 1. QUERY FUNCTION
# ----------------------------
def query_qwen(question, answer):
    prompt = f"""
You are a strict evaluator.

Task:
Determine whether the ANSWER correctly and fully answers the QUESTION.

You MUST choose one option:
- "yes" → fully correct and supported by knowledge
- "no" → clearly incorrect or contradictory
- "uncertain" → not enough information or ambiguous

Rules:
- Be strict and consistent.
- Do NOT assume missing information.
- If unsure, use "uncertain".

Return ONLY valid JSON:
{{"verdict": "yes" or "no" or "uncertain"}}

QUESTION:
{question}

ANSWER:
{answer}
""".strip()

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        r = requests.post(VLLM_URL, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return json.dumps({"verdict": "uncertain", "error": str(e)})


# ----------------------------
# 2. PARSE OUTPUT
# ----------------------------
def parse_verdict(text):
    try:
        data = json.loads(text)
        v = data.get("verdict", "uncertain").lower()

        if v == "yes":
            return 2
        elif v == "no":
            return 0
        else:
            return 1  # uncertain
    except:
        t = text.lower()
        if "yes" in t:
            return 2
        elif "no" in t:
            return 0
        else:
            return 1


# ----------------------------
# 3. LOAD DATASETS
# ----------------------------
print(f"\n===== BEFORE FILTERING =====")
attackqa = pd.read_parquet(
    "https://huggingface.co/datasets/sambanovasystems/attackqa/resolve/main/attackqa.parquet",
    engine="fastparquet"
)

print(f"\n===== AttackQA =====")
print(attackqa.head())
print(attackqa.shape)

cyberqa_json = pd.read_json(
    "https://huggingface.co/datasets/Rowden/CybersecurityQAA/resolve/main/cybersecurityQAAdataset.json"
)
cyberqa = pd.json_normalize(cyberqa_json["vars"])

print(f"\n===== CybersecurityQAA =====")
print(cyberqa.head())
print(cyberqa.shape)


# ----------------------------
# 4. FILTER HUMAN-GENERATED ONLY
# ----------------------------

attackqa_human = attackqa[attackqa["human_answer"] == True].copy()

cyberqa_human = cyberqa[
    cyberqa['reviewed_by_expert'].astype(str).map({'TRUE': True, 'FALSE': False, '': False}).fillna(False) == True
].copy()

print("Проверка на NaN в attackqa_human:")
print(f"Всего строк: {len(attackqa_human)}")
print(f"Строк с NaN в question: {attackqa_human['question'].isna().sum()}")
print(f"Строк с NaN в answer: {attackqa_human['answer'].isna().sum()}")
print(f"Строк с NaN в question ИЛИ answer: {attackqa_human[['question', 'answer']].isna().any(axis=1).sum()}")

# Проверим, сколько останется после dropna
clean_df = attackqa_human.dropna(subset=["question", "answer"])
print(f"Строк после dropna: {len(clean_df)}")

print(f"\n===== AFTER FILTERING =====")

print(f"\n===== AttackQA =====")
print(attackqa_human.head())
print(attackqa_human.shape)

print(f"\n===== CybersecurityQAA =====")
print(cyberqa_human.head())
print(cyberqa_human.shape)

# ----------------------------
# 5. RUN EVALUATION
# ----------------------------
def evaluate(df, name):
    df = df.reset_index(drop=True)
    df = df.dropna(subset=["question", "answer"])
    print(f"\n===== {name} PROCESSING =====")
    print(df.head())
    print(df.shape)
    y_pred = []
    raw_outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row["question"])
        a = str(row["answer"])

        raw = query_qwen(q, a)
        pred = parse_verdict(raw)

        y_pred.append(pred)
        raw_outputs.append(raw)

    results_df = pd.DataFrame({
        "question": df["question"].values,
        "answer": df["answer"].values,
        "y_pred": y_pred,
        "raw_output": raw_outputs
    })

    results_df.to_csv(f"{SMALL_MODEL_NAME}_results/{name}_evaluation.csv", index=False)

    # --------
    # METRICS
    # --------

    total = len(y_pred)

    yes_count = sum(p == 2 for p in y_pred)
    no_count = sum(p == 0 for p in y_pred)
    uncertain_count = sum(p == 1 for p in y_pred)

    agreement_rate = yes_count / total
    uncertainty_rate = uncertain_count / total

    selectivity = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0.0

    print(f"\n===== {name} RESULTS =====")

    print("\n--- Distribution ---")
    print(f"YES        : {yes_count} ({yes_count/total:.3f})")
    print(f"NO         : {no_count} ({no_count/total:.3f})")
    print(f"UNCERTAIN  : {uncertain_count} ({uncertain_count/total:.3f})")

    print("\n--- Key Metrics ---")
    print(f"Agreement rate (YES): {agreement_rate:.3f}")
    print(f"Uncertainty rate    : {uncertainty_rate:.3f}")
    print(f"Selectivity (YES/(YES+NO)): {selectivity:.3f}")

    metrics = {
        "model": MODEL,
        "name": name,
        "total": total,
        "yes": yes_count,
        "no": no_count,
        "uncertain": uncertain_count,
        "agreement_rate": agreement_rate,
        "uncertainty_rate": uncertainty_rate,
        "selectivity": selectivity
    }

    with open(f"{SMALL_MODEL_NAME}_results/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ----------------------------
# 6. RUN BOTH
# ----------------------------
evaluate(attackqa_human.sample(min(400, len(attackqa_human)), random_state=42), "AttackQA_Human")
evaluate(cyberqa_human.sample(min(400, len(cyberqa_human)), random_state=42), "CybersecurityQAA_Human")