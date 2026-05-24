import os
import pandas as pd
import requests
import json
import re
from tqdm import tqdm
import time

# Конфигурация
VLLM_URL = "http://localhost:8000/v1/chat/completions"
SMALL_MODEL_NAME = "Qwen2.5-7B-Instruct"
MODEL = "Qwen/" + SMALL_MODEL_NAME
OUTPUT_DIR = f"{SMALL_MODEL_NAME}_results"

# Создаем директорию для результатов, если нет
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_lines = []

def log(line=""):
    print(line)
    output_lines.append(line)

# ----------------------------
# 1. QUERY FUNCTION
# ----------------------------
def query_qwen(question, answer):
    # Обновленный промпт с требованием reasoning
    prompt = f"""You are an expert evaluator in cybersecurity and general knowledge.

Task:
Evaluate whether the ANSWER correctly and fully addresses the QUESTION.

Criteria:
1. Factual Accuracy: The answer must not contain hallucinations or contradictions.
2. Completeness: The answer must address all parts of the question.
3. Relevance: The answer must be direct and on-topic.

Instructions:
- First, provide a brief "reasoning" explaining why the answer is correct, incorrect, or ambiguous.
- Then, choose one verdict:
  - "yes": Fully correct, accurate, and complete.
  - "no": Contains errors, is incomplete, or is irrelevant.
  - "uncertain": Ambiguous, lacks context, or you cannot verify the facts.

Return ONLY valid JSON format with keys "reasoning" and "verdict".

Example Output:
{{
  "reasoning": "The answer correctly identifies the port number but fails to explain the protocol.",
  "verdict": "no"
}}

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
        "temperature": 0,
        "max_tokens": 512  # Увеличиваем лимит токенов для reasoning
    }

    try:
        r = requests.post(VLLM_URL, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        # Возвращаем структуру ошибки в формате JSON для единообразия парсинга
        return json.dumps({"reasoning": f"API Error: {str(e)}", "verdict": "uncertain"})


# ----------------------------
# 2. PARSE OUTPUT
# ----------------------------
def parse_llm_response(text):
    """
    Пытается извлечь JSON из ответа LLM.
    Возвращает dict с ключами 'reasoning' и 'verdict_code'.
    verdict_code: 2 (yes), 0 (no), 1 (uncertain)
    """
    default_result = {"reasoning": "Parse Failed", "verdict_code": 1, "raw_verdict": "uncertain"}
    
    if not text:
        return default_result

    cleaned_text = text.strip()
    
    # Шаг 1: Попытка найти JSON блок внутри текста (часто модели оборачивают в ```json ... ```)
    json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
    
    parsed_data = None
    if json_match:
        try:
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Шаг 2: Если не нашли JSON regex-ом, пробуем распарсить весь текст как JSON
    if parsed_data is None:
        try:
            parsed_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

    # Шаг 3: Если JSON получен, извлекаем данные
    if parsed_data and isinstance(parsed_data, dict):
        reasoning = parsed_data.get("reasoning", "No reasoning provided")
        verdict_raw = str(parsed_data.get("verdict", "uncertain")).lower().strip()
        
        # Нормализация вердикта
        if "yes" in verdict_raw:
            verdict_code = 2
        elif "no" in verdict_raw:
            verdict_code = 0
        else:
            verdict_code = 1 # uncertain или любое другое значение
        
        return {
            "reasoning": reasoning,
            "verdict_code": verdict_code,
            "raw_verdict": verdict_raw
        }

    # Шаг 4: Fallback - если JSON не удалось извлечь, ищем ключевые слова в тексте
    # Это спасает, если модель сломала формат JSON, но написала текст
    lower_text = cleaned_text.lower()
    
    # Ищем паттерны вида "verdict: yes" или просто наличие слов в конце
    # Приоритет: no > yes > uncertain (чтобы быть строгими)
    
    reasoning_extracted = cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
    
    if "verdict" in lower_text:
        # Пытаемся найти значение после слова verdict
        match = re.search(r'verdict["\s:]*([a-z]+)', lower_text)
        if match:
            val = match.group(1)
            if "no" in val:
                return {"reasoning": reasoning_extracted, "verdict_code": 0, "raw_verdict": "no"}
            elif "yes" in val:
                return {"reasoning": reasoning_extracted, "verdict_code": 2, "raw_verdict": "yes"}
            else:
                return {"reasoning": reasoning_extracted, "verdict_code": 1, "raw_verdict": "uncertain"}

    # Полный fallback по содержанию
    if "incorrect" in lower_text or "wrong" in lower_text or "contradict" in lower_text:
        return {"reasoning": reasoning_extracted, "verdict_code": 0, "raw_verdict": "no"}
    
    if "correct" in lower_text and ("fully" in lower_text or "accurate" in lower_text):
        return {"reasoning": reasoning_extracted, "verdict_code": 2, "raw_verdict": "yes"}
        
    # По умолчанию uncertain
    return default_result


# ----------------------------
# 3. LOAD DATASETS
# ----------------------------
print(f"\n===== LOADING DATA =====")

# AttackQA
try:
    attackqa = pd.read_parquet(
        "https://huggingface.co/datasets/sambanovasystems/attackqa/resolve/main/attackqa.parquet",
        engine="pyarrow"
    )
    print(f"AttackQA loaded: {attackqa.shape}")
except Exception as e:
    print(f"Error loading AttackQA: {e}")
    attackqa = pd.DataFrame()

# CybersecurityQAA
try:
    cyberqa_json = pd.read_json(
        "https://huggingface.co/datasets/Rowden/CybersecurityQAA/resolve/main/cybersecurityQAAdataset.json"
    )
    cyberqa = pd.json_normalize(cyberqa_json["vars"])
    print(f"CybersecurityQAA loaded: {cyberqa.shape}")
except Exception as e:
    print(f"Error loading CybersecurityQAA: {e}")
    cyberqa = pd.DataFrame()


# ----------------------------
# 4. FILTER HUMAN-GENERATED ONLY
# ----------------------------

def filter_human_data(df, source_name):
    if df.empty:
        return pd.DataFrame()
        
    if source_name == "AttackQA":
        # Фильтр по human_answer == True
        if "human_answer" in df.columns:
            df_filtered = df[df["human_answer"] == True].copy()
        else:
            df_filtered = df.copy()
            
    elif source_name == "CybersecurityQAA":
        # Фильтр по reviewed_by_expert
        if "reviewed_by_expert" in df.columns:
            # Преобразуем возможные строковые значения 'TRUE'/'FALSE' или булевы
            def is_expert_reviewed(val):
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.upper() == 'TRUE'
                return False
            
            mask = df['reviewed_by_expert'].apply(is_expert_reviewed)
            df_filtered = df[mask].copy()
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()

    # Очистка NaN
    df_filtered = df_filtered.dropna(subset=["question", "answer"])
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Приведение к строке для безопасности
    df_filtered["question"] = df_filtered["question"].astype(str)
    df_filtered["answer"] = df_filtered["answer"].astype(str)
    
    return df_filtered

attackqa_human = filter_human_data(attackqa, "AttackQA")
cyberqa_human = filter_human_data(cyberqa, "CybersecurityQAA")

print(f"\n===== AFTER FILTERING =====")
print(f"AttackQA Human: {attackqa_human.shape}")
print(f"CybersecurityQAA Human: {cyberqa_human.shape}")


# ----------------------------
# 5. RUN EVALUATION
# ----------------------------
def evaluate(df, name):
    if df.empty:
        print(f"Skipping {name} due to empty dataframe.")
        return

    print(f"\n===== {name} PROCESSING =====")
    
    y_pred_codes = []
    reasons = []
    raw_outputs = []
    verdicts_raw = []

    # Ограничиваем выборку для теста, если нужно (уберите .sample для полного прогона)
    # df_eval = df.sample(min(400, len(df)), random_state=42) 
    df_eval = df # Используем все данные или сделайте сэмпл здесь
    
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        q = row["question"]
        a = row["answer"]

        raw_response = query_qwen(q, a)
        parsed = parse_llm_response(raw_response)

        y_pred_codes.append(parsed["verdict_code"])
        reasons.append(parsed["reasoning"])
        raw_outputs.append(raw_response)
        verdicts_raw.append(parsed["raw_verdict"])
        
        # Небольшая задержка, чтобы не перегружать сервер, если нужно
        # time.sleep(0.1) 

    results_df = pd.DataFrame({
        "question": df_eval["question"].values,
        "answer": df_eval["answer"].values,
        "verdict_code": y_pred_codes,
        "verdict_raw": verdicts_raw,
        "reasoning": reasons,
        "raw_output": raw_outputs
    })

    # Сохранение результатов
    output_path = os.path.join(OUTPUT_DIR, f"{name}_evaluation.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # --------
    # METRICS
    # --------
    total = len(y_pred_codes)
    if total == 0:
        return

    yes_count = sum(p == 2 for p in y_pred_codes)
    no_count = sum(p == 0 for p in y_pred_codes)
    uncertain_count = sum(p == 1 for p in y_pred_codes)

    agreement_rate = yes_count / total
    uncertainty_rate = uncertain_count / total
    selectivity = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0.0

    print(f"\n===== {name} RESULTS =====")
    print(f"Total samples: {total}")
    print(f"YES (2)      : {yes_count} ({yes_count/total:.3f})")
    print(f"NO (0)       : {no_count} ({no_count/total:.3f})")
    print(f"UNCERTAIN (1): {uncertain_count} ({uncertain_count/total:.3f})")
    print(f"Selectivity (YES/(YES+NO)): {selectivity:.3f}")

    metrics = {
        "model": MODEL,
        "dataset": name,
        "total": total,
        "yes": yes_count,
        "no": no_count,
        "uncertain": uncertain_count,
        "agreement_rate": round(agreement_rate, 4),
        "uncertainty_rate": round(uncertainty_rate, 4),
        "selectivity": round(selectivity, 4)
    }

    metrics_path = os.path.join(OUTPUT_DIR, f"{name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


# ----------------------------
# 6. RUN BOTH
# ----------------------------
if __name__ == "__main__":
    evaluate(attackqa_human.sample(min(400, len(attackqa_human)), random_state=42), "AttackQA_Human")
    evaluate(cyberqa_human.sample(min(400, len(cyberqa_human)), random_state=42), "CybersecurityQAA_Human")