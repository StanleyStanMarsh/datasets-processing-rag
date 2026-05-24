import pandas as pd
import requests
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-3B-Instruct"


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
attackqa = pd.read_parquet(
    "https://huggingface.co/datasets/sambanovasystems/attackqa/resolve/main/attackqa.parquet",
    engine="fastparquet"
)

cyberqa_json = pd.read_json(
    "https://huggingface.co/datasets/Rowden/CybersecurityQAA/resolve/main/cybersecurityQAAdataset.json"
)
cyberqa = pd.json_normalize(cyberqa_json["vars"])


# ----------------------------
# 4. FILTER HUMAN-GENERATED ONLY
# ----------------------------

attackqa_human = attackqa[attackqa["human_answer"] == True].copy()

cyberqa_human = cyberqa[
    (cyberqa["reviewed_by_human"] == True) |
    (cyberqa["reviewed_by_expert"] == True)
].copy()


# ----------------------------
# 5. RUN EVALUATION
# ----------------------------
def evaluate(df, name):
    y_true = []
    y_pred = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row["question"])
        a = str(row["answer"])

        # ground truth assumption: all human/expert reviewed are correct
        true_label = 2  # "yes"

        raw = query_qwen(q, a)
        pred = parse_verdict(raw)

        y_true.append(true_label)
        y_pred.append(pred)

    print(f"\n===== {name} RESULTS =====")

    labels = [0, 1, 2]
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=labels))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=["no", "uncertain", "yes"],
        digits=4
    ))

    # additional useful stats
    total = len(y_pred)
    print("\nDistribution:")
    print("yes:", y_pred.count(2) / total)
    print("uncertain:", y_pred.count(1) / total)
    print("no:", y_pred.count(0) / total)


# ----------------------------
# 6. RUN BOTH
# ----------------------------
evaluate(attackqa_human.sample(min(200, len(attackqa_human))), "AttackQA Human")
evaluate(cyberqa_human.sample(min(200, len(cyberqa_human))), "CybersecurityQAA Human")