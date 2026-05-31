import os
os.environ['HF_HUB_ENABLE_XET'] = '0'
os.environ['HF_HOME'] = '/raid/iastafyev/hf_cache'

import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# --- КОНФИГУРАЦИЯ ---

# Путь к датасету (CSV с колонками 'question' и 'positive')
DATASET_PATH = "final_full_test_df.csv"

# Список bi-encoder моделей
BI_ENCODER_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "BAAI/bge-base-en-v1.5"
]

# Список reranker моделей (пути к локальным или название из HF)
RERANKER_MODELS = [
    "./msmarco-minilm-finetuned-ranknet-passed_cross_mining_method_validated_df",
    "./msmarco-minilm-finetuned-ranknet-passed_oracle_method_vlidated_df",
    "./msmarco-minilm-finetuned-triplet-passed_cross_mining_method_validated_df",
    "./msmarco-minilm-finetuned-triplet-passed_oracle_method_vlidated_df",
    "cross-encoder/ms-marco-MiniLM-L6-v2"
]

# Параметры поиска
TOP_K_RETRIEVAL = 15
HIT_RATE_KS = [3, 5, 10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_prefixes(model_name):
    """
    Возвращает префиксы для query и passage в зависимости от модели.
    """
    if "e5" in model_name.lower():
        return "query: ", "passage: "
    elif "bge" in model_name.lower():
        # Для BGE часто рекомендуется префикс только для query.
        return "Represent this sentence for searching relevant passages: ", ""
    else:
        return "", ""

def calculate_metrics_by_indices(retrieved_indices_list, ground_truth_indices_list, top_k_retrieval, hit_rate_ks):
    """
    retrieved_indices_list: список списков индексов документов (np.array или list)
    ground_truth_indices_list: список индексов правильных ответов (int)
    """
    mrr_sum = 0.0
    hit_rates = {k: 0 for k in hit_rate_ks}
    n_questions = len(ground_truth_indices_list)

    for i, gt_idx in enumerate(ground_truth_indices_list):
        docs_indices = retrieved_indices_list[i]

        # Ищем позицию правильного индекса в списке возвращенных
        try:
            if isinstance(docs_indices, np.ndarray):
                rank_arr = np.where(docs_indices == gt_idx)[0]
                if len(rank_arr) > 0:
                    rank = rank_arr[0] + 1 # Rank начинается с 1
                else:
                    rank = None
            else:
                rank = docs_indices.index(gt_idx) + 1
        except ValueError:
            rank = None

        # MRR calculation
        if rank is not None and rank <= top_k_retrieval:
            mrr_sum += 1.0 / rank

        # HitRate calculation
        for k in hit_rate_ks:
            if rank is not None and rank <= k:
                hit_rates[k] += 1

    mrr = mrr_sum / n_questions if n_questions > 0 else 0.0
    hit_rates_normalized = {k: v / n_questions if n_questions > 0 else 0.0 for k, v in hit_rates.items()}

    return mrr, hit_rates_normalized

# --- ЗАГРУЗКА МОДЕЛЕЙ ---

def load_bi_encoder(model_name):
    print(f"Loading bi-encoder: {model_name}")
    model = SentenceTransformer(model_name, device=DEVICE)
    return model

def load_reranker(model_path):
    print(f"Loading reranker: {model_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path, device=DEVICE)
    return model

# --- ОСНОВНОЙ ЦИКЛ ---

def run_experiment():
    # Загрузка датасета
    print(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Проверка на дубликаты или пустые значения
    df.dropna(subset=['question', 'positive'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    questions = df['question'].tolist()
    positives = df['positive'].tolist()
    
    num_questions = len(questions)
    if num_questions == 0:
        print("Dataset is empty after cleaning.")
        return

    # Ground truth indices: индекс строки в df совпадает с индексом в FAISS, 
    # так как мы добавляем документы в том же порядке, в котором они в df.
    ground_truth_indices = list(range(num_questions))

    results = []

    # Перебор bi-encoder моделей
    for be_model_name in BI_ENCODER_MODELS:
        be_model = load_bi_encoder(be_model_name)
        query_prefix, passage_prefix = get_prefixes(be_model_name)

        # 1. Построение индекса FAISS для текущей bi-encoder модели
        print(f"Encoding documents for {be_model_name}...")

        docs_to_encode = [f"{passage_prefix}{doc}" for doc in positives]

        doc_embeddings = be_model.encode(docs_to_encode, convert_to_numpy=True, show_progress_bar=True)

        # Нормализация эмбеддингов для использования косинусного сходства через IP (Inner Product)
        faiss.normalize_L2(doc_embeddings)

        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings)

        print(f"Index built for {be_model_name}. Size: {index.ntotal}")

        # Перебор reranker моделей (включая вариант "None" для чистого bi-encoder)
        reranker_list = [None] + RERANKER_MODELS

        for reranker_path in reranker_list:
            reranker_model = None
            if reranker_path is not None:
                reranker_model = load_reranker(reranker_path)

            reranker_name = "No_Reranker" if reranker_path is None else os.path.basename(reranker_path)
            print(f"\nTesting combination: BE={be_model_name} | RR={reranker_name}")

            # 2. Поиск и Реранкинг с замером времени

            # --- Замер времени Retrieval (Encode Query + FAISS Search) ---
            start_time_retrieval = time.time()

            queries_to_encode = [f"{query_prefix}{q}" for q in questions]
            # Отключаем прогресс-бар внутри замера времени, чтобы не замедлять и не засорять вывод
            query_embeddings = be_model.encode(queries_to_encode, convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(query_embeddings)

            # Поиск топ-15 в FAISS
            D, I = index.search(query_embeddings, TOP_K_RETRIEVAL)

            end_time_retrieval = time.time()
            total_time_retrieval_sec = end_time_retrieval - start_time_retrieval
            
            # Расчет среднего времени на один запрос
            avg_time_retrieval_per_query = total_time_retrieval_sec / num_questions

            # I имеет размерность (num_questions, 15). Содержит индексы документов.
            retrieved_indices_list = [I[i] for i in range(num_questions)]

            # --- Замер времени Reranking ---
            total_time_rerank_sec = 0.0
            avg_time_rerank_per_query = 0.0
            final_retrieved_indices_list = retrieved_indices_list # По умолчанию без реранка

            if reranker_model is not None:
                start_time_rerank = time.time()

                final_retrieved_indices_list = []

                # Для реранкинга оставляем tqdm, чтобы видеть прогресс, так как это долго
                print("Processing results and reranking...")
                for i in tqdm(range(num_questions), desc="Reranking"):
                    doc_indices = I[i]
                    candidate_docs = [positives[idx] for idx in doc_indices]

                    # Подготовка пар для cross-encoder
                    pairs = [[questions[i], doc] for doc in candidate_docs]

                    # Получение скоров
                    scores = reranker_model.predict(pairs, show_progress_bar=False)

                    # Сортировка по убыванию скора
                    scored_indices = list(zip(scores, doc_indices))
                    scored_indices.sort(key=lambda x: x[0], reverse=True)

                    # Извлекаем индексы в новом порядке
                    sorted_indices = np.array([idx for _, idx in scored_indices])
                    final_retrieved_indices_list.append(sorted_indices)

                end_time_rerank = time.time()
                total_time_rerank_sec = end_time_rerank - start_time_rerank
                
                # Расчет среднего времени на один запрос
                avg_time_rerank_per_query = total_time_rerank_sec / num_questions

            # 3. Расчет метрик по индексам
            mrr, hit_rates = calculate_metrics_by_indices(
                final_retrieved_indices_list,
                ground_truth_indices,
                TOP_K_RETRIEVAL,
                HIT_RATE_KS
            )

            print(f"MRR@{TOP_K_RETRIEVAL}: {mrr:.4f}")
            for k, hr in hit_rates.items():
                print(f"HitRate@{k}: {hr:.4f}")
            
            print(f"Avg Time Retrieval per query: {avg_time_retrieval_per_query:.4f}s (Total: {total_time_retrieval_sec:.2f}s)")
            print(f"Avg Time Rerank per query:    {avg_time_rerank_per_query:.4f}s (Total: {total_time_rerank_sec:.2f}s)")

            # Сохранение результата
            results.append({
                "bi_encoder": be_model_name,
                "reranker": reranker_name,
                "MRR@15": mrr,
                **{f"HitRate@{k}": v for k, v in hit_rates.items()},
                "avg_time_retrieval_sec": avg_time_retrieval_per_query,
                "avg_time_rerank_sec": avg_time_rerank_per_query,
                "total_time_retrieval_sec": total_time_retrieval_sec,
                "total_time_rerank_sec": total_time_rerank_sec
            })

            # Освобождение памяти от reranker если он был загружен
            if reranker_model is not None:
                del reranker_model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        # Освобождение памяти от bi-encoder перед следующей моделью
        del be_model
        del index
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Сохранение итоговой таблицы результатов
    results_df = pd.DataFrame(results)
    results_df.to_csv("retrieval_experiment_results.csv", index=False)
    print("\nExperiment finished. Results saved to retrieval_experiment_results.csv")
    print(results_df)

if __name__ == "__main__":
    run_experiment()
