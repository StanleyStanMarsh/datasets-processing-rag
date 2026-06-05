import os
os.environ["HF_HUB_ENABLE_XET"] = "0"
os.environ["HF_HOME"] = "/raid/iastafyev/hf_cache"

import argparse
import json
import time
import gc
import random
import numpy as np
import pandas as pd
import torch
import faiss

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_prefixes(model_name):
    name = model_name.lower()

    if "e5" in name:
        return "query: ", "passage: "

    if "bge" in name:
        return "Represent this sentence for searching relevant passages: ", ""

    return "", ""


def get_short_name(path_or_name):
    path_or_name = path_or_name.rstrip("/")

    if os.path.isdir(path_or_name):
        parts = path_or_name.split(os.sep)

        if len(parts) >= 3 and parts[-1] == "best_model":
            return f"{parts[-3]}__{parts[-2]}"

        return os.path.basename(path_or_name)

    return path_or_name


def resolve_triplet_best_model(experiment_dir):
    """
    Возвращает путь к best_model для Triplet-эксперимента.
    Ожидает структуру:
    experiment_dir/
      best_config.json
      grid_search_results.csv
      run_xxx/
        best_model/
    """
    best_config_path = os.path.join(experiment_dir, "best_config.json")

    if os.path.exists(best_config_path):
        with open(best_config_path, "r", encoding="utf-8") as f:
            best_config = json.load(f)

        if "best_model_dir" in best_config:
            return best_config["best_model_dir"]

        if "run_dir" in best_config:
            return os.path.join(best_config["run_dir"], "best_model")

    grid_path = os.path.join(experiment_dir, "grid_search_results.csv")

    if os.path.exists(grid_path):
        df = pd.read_csv(grid_path)
        best_row = df.sort_values("best_val_loss").iloc[0]
        return best_row["best_model_dir"]

    raise FileNotFoundError(
        f"Не удалось найти best_config.json или grid_search_results.csv в {experiment_dir}"
    )


def build_reranker_list(ranknet_root, triplet_root):
    rerankers = []

    # baseline reranker
    rerankers.append({
        "reranker_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "reranker_path": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "loss_type": "baseline",
        "train_dataset": "ms-marco"
    })

    # RankNet models
    for dataset_dir in sorted(os.listdir(ranknet_root)):
        full_dir = os.path.join(ranknet_root, dataset_dir)
        best_model_dir = os.path.join(full_dir, "best_model")

        if os.path.isdir(best_model_dir):
            rerankers.append({
                "reranker_name": f"ranknet__{dataset_dir}",
                "reranker_path": best_model_dir,
                "loss_type": "ranknet",
                "train_dataset": dataset_dir
            })

    # Triplet models
    for dataset_dir in sorted(os.listdir(triplet_root)):
        full_dir = os.path.join(triplet_root, dataset_dir)

        if os.path.isdir(full_dir):
            best_model_dir = resolve_triplet_best_model(full_dir)

            rerankers.append({
                "reranker_name": f"triplet__{dataset_dir}",
                "reranker_path": best_model_dir,
                "loss_type": "triplet",
                "train_dataset": dataset_dir
            })

    return rerankers


def calculate_metrics_by_indices(retrieved_indices_list, ground_truth_indices, hit_rate_ks):
    mrr_sum = 0.0
    hit_counts = {k: 0 for k in hit_rate_ks}

    n = len(ground_truth_indices)

    for retrieved_indices, gt_idx in zip(retrieved_indices_list, ground_truth_indices):
        retrieved_indices = list(retrieved_indices)

        if gt_idx in retrieved_indices:
            rank = retrieved_indices.index(gt_idx) + 1
            mrr_sum += 1.0 / rank

            for k in hit_rate_ks:
                if rank <= k:
                    hit_counts[k] += 1

    mrr = mrr_sum / n if n > 0 else 0.0
    hit_rates = {k: hit_counts[k] / n if n > 0 else 0.0 for k in hit_rate_ks}

    return mrr, hit_rates


def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["question", "positive"]).reset_index(drop=True)

    duplicated_positive_count = df["positive"].duplicated().sum()

    if duplicated_positive_count > 0:
        print(
            f"⚠️ Найдено duplicate positive: {duplicated_positive_count}. "
            f"При single-ground-truth оценке это может занижать MRR/HitRate."
        )

    return df


def encode_corpus(be_model, positives, passage_prefix, batch_size):
    docs_to_encode = [f"{passage_prefix}{doc}" for doc in positives]

    embeddings = be_model.encode(
        docs_to_encode,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False
    )

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    return embeddings


def encode_queries(be_model, questions, query_prefix, batch_size):
    queries_to_encode = [f"{query_prefix}{q}" for q in questions]

    embeddings = be_model.encode(
        queries_to_encode,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False
    )

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    return embeddings


def rerank_batched(
    reranker_model,
    questions,
    positives,
    retrieved_indices_matrix,
    rerank_batch_size
):
    num_questions = len(questions)
    top_k = retrieved_indices_matrix.shape[1]

    all_pairs = []

    for i in range(num_questions):
        for doc_idx in retrieved_indices_matrix[i]:
            all_pairs.append([questions[i], positives[int(doc_idx)]])

    scores = reranker_model.predict(
        all_pairs,
        batch_size=rerank_batch_size,
        show_progress_bar=True
    )

    scores = np.asarray(scores).reshape(num_questions, top_k)

    final_indices = []

    for i in range(num_questions):
        order = np.argsort(-scores[i])
        sorted_indices = retrieved_indices_matrix[i][order]
        final_indices.append(sorted_indices)

    return final_indices


def run_experiment(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")
    print(f"📌 Seed: {args.seed}")

    df = load_dataset(args.dataset_path)

    questions = df["question"].astype(str).tolist()
    positives = df["positive"].astype(str).tolist()
    ground_truth_indices = list(range(len(df)))

    print(f"📊 Test examples: {len(df)}")

    bi_encoder_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/e5-base-v2",
        "intfloat/e5-large-v2",
        "BAAI/bge-base-en-v1.5"
    ]

    reranker_configs = build_reranker_list(
        ranknet_root=args.ranknet_root,
        triplet_root=args.triplet_root
    )

    print("\n📌 Rerankers:")
    print("   No_Reranker")
    for r in reranker_configs:
        print(f"   {r['reranker_name']} -> {r['reranker_path']}")

    results = []

    for be_name in bi_encoder_models:
        print("\n" + "=" * 100)
        print(f"🔎 Bi-encoder: {be_name}")
        print("=" * 100)

        query_prefix, passage_prefix = get_prefixes(be_name)

        be_model = SentenceTransformer(be_name, device=device)

        print("Encoding corpus...")
        corpus_embeddings = encode_corpus(
            be_model,
            positives,
            passage_prefix,
            args.bi_encoder_batch_size
        )

        dim = corpus_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_embeddings)

        print(f"FAISS index size: {index.ntotal}")

        print("Encoding queries and searching...")
        start_retrieval = time.perf_counter()

        query_embeddings = encode_queries(
            be_model,
            questions,
            query_prefix,
            args.bi_encoder_batch_size
        )

        _, retrieved_indices = index.search(
            query_embeddings,
            args.top_k_retrieval
        )

        end_retrieval = time.perf_counter()

        total_time_retrieval = end_retrieval - start_retrieval
        avg_time_retrieval = total_time_retrieval / len(questions)

        # ---------- No reranker ----------
        mrr, hit_rates = calculate_metrics_by_indices(
            retrieved_indices_list=[retrieved_indices[i] for i in range(len(questions))],
            ground_truth_indices=ground_truth_indices,
            hit_rate_ks=args.hit_rate_ks
        )

        results.append({
            "bi_encoder": be_name,
            "reranker": "No_Reranker",
            "reranker_path": None,
            "loss_type": "none",
            "train_dataset": "none",
            f"MRR@{args.top_k_retrieval}": mrr,
            **{f"HitRate@{k}": v for k, v in hit_rates.items()},
            "avg_time_retrieval_sec": avg_time_retrieval,
            "avg_time_rerank_sec": 0.0,
            "total_time_retrieval_sec": total_time_retrieval,
            "total_time_rerank_sec": 0.0,
            "top_k_retrieval": args.top_k_retrieval
        })

        print(f"\nBE only | MRR@{args.top_k_retrieval}: {mrr:.4f}")

        for k, v in hit_rates.items():
            print(f"BE only | HitRate@{k}: {v:.4f}")

        # ---------- Rerankers ----------
        for reranker_cfg in reranker_configs:
            reranker_path = reranker_cfg["reranker_path"]
            reranker_name = reranker_cfg["reranker_name"]

            print("\n" + "-" * 100)
            print(f"Testing reranker: {reranker_name}")
            print(f"Path: {reranker_path}")
            print("-" * 100)

            reranker_model = CrossEncoder(
                reranker_path,
                device=device,
                max_length=args.cross_encoder_max_length
            )

            start_rerank = time.perf_counter()

            final_indices = rerank_batched(
                reranker_model=reranker_model,
                questions=questions,
                positives=positives,
                retrieved_indices_matrix=retrieved_indices,
                rerank_batch_size=args.rerank_batch_size
            )

            end_rerank = time.perf_counter()

            total_time_rerank = end_rerank - start_rerank
            avg_time_rerank = total_time_rerank / len(questions)

            mrr, hit_rates = calculate_metrics_by_indices(
                retrieved_indices_list=final_indices,
                ground_truth_indices=ground_truth_indices,
                hit_rate_ks=args.hit_rate_ks
            )

            print(f"MRR@{args.top_k_retrieval}: {mrr:.4f}")

            for k, v in hit_rates.items():
                print(f"HitRate@{k}: {v:.4f}")

            print(f"Avg retrieval time/query: {avg_time_retrieval:.6f}s")
            print(f"Avg rerank time/query:    {avg_time_rerank:.6f}s")

            results.append({
                "bi_encoder": be_name,
                "reranker": reranker_name,
                "reranker_path": reranker_path,
                "loss_type": reranker_cfg["loss_type"],
                "train_dataset": reranker_cfg["train_dataset"],
                f"MRR@{args.top_k_retrieval}": mrr,
                **{f"HitRate@{k}": v for k, v in hit_rates.items()},
                "avg_time_retrieval_sec": avg_time_retrieval,
                "avg_time_rerank_sec": avg_time_rerank,
                "total_time_retrieval_sec": total_time_retrieval,
                "total_time_rerank_sec": total_time_rerank,
                "top_k_retrieval": args.top_k_retrieval
            })

            del reranker_model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del be_model
        del index
        del corpus_embeddings
        del query_embeddings
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=[f"MRR@{args.top_k_retrieval}", "HitRate@10"],
        ascending=False
    )

    results_df.to_csv(args.output_csv, index=False)

    print("\n" + "=" * 100)
    print("✅ EXPERIMENT FINISHED")
    print(f"📄 Results saved to: {args.output_csv}")
    print("=" * 100)
    print(results_df.head(20))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="final_full_test_df.csv"
    )

    parser.add_argument(
        "--ranknet_root",
        type=str,
        default="minilm_rerank_ranknet"
    )

    parser.add_argument(
        "--triplet_root",
        type=str,
        default="minilm_rerank_triplet"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="retrieval_experiment_results.csv"
    )

    parser.add_argument(
        "--top_k_retrieval",
        type=int,
        default=15
    )

    parser.add_argument(
        "--hit_rate_ks",
        type=int,
        nargs="+",
        default=[3, 5, 10]
    )

    parser.add_argument(
        "--bi_encoder_batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--rerank_batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--cross_encoder_max_length",
        type=int,
        default=256
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
