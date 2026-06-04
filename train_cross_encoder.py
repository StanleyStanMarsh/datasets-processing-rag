import os
os.environ["HF_HUB_ENABLE_XET"] = "0"
os.environ["HF_HOME"] = "/raid/iastafyev/hf_cache"

import argparse
import random
import json
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from dotenv import load_dotenv


# ==========================================
# 1. Reproducibility
# ==========================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 2. Loss
# ==========================================
class OrderedTripletLoss(nn.Module):
    def __init__(self, w_pos_neg=4.0, w_pos_hard=2.0, w_hard_mid=1.0, margin=0.1):
        super().__init__()
        self.w_pos_neg = w_pos_neg
        self.w_pos_hard = w_pos_hard
        self.w_hard_mid = w_hard_mid
        self.margin = margin

    def forward(self, s_w, s_v, s_u):
        # w = positive
        # v = hard_negative
        # u = soft_negative

        # positive should be higher than soft negative
        loss_pos_neg = F.softplus((s_u - s_w) - self.margin)

        # positive should be higher than hard negative
        loss_pos_hard = F.softplus((s_v - s_w) - self.margin)

        # hard negative should be between positive and soft negative
        midpoint = (s_w + s_u) / 2
        loss_hard_mid = F.softplus((midpoint - s_v) - self.margin)

        total_weight = self.w_pos_neg + self.w_pos_hard + self.w_hard_mid

        return (
            self.w_pos_neg * loss_pos_neg +
            self.w_pos_hard * loss_pos_hard +
            self.w_hard_mid * loss_hard_mid
        ).mean() / total_weight


# ==========================================
# 3. Dataset
# ==========================================
class TripletDataset(Dataset):
    def __init__(self, df):
        required_columns = {"question", "positive", "hard_negative", "soft_negative"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"В датасете отсутствуют обязательные столбцы: {missing}")

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "question": str(row["question"]),
            "positive": str(row["positive"]),
            "hard_negative": str(row["hard_negative"]),
            "soft_negative": str(row["soft_negative"])
        }


# ==========================================
# 4. Scoring
# ==========================================
def compute_scores(tokenizer, model, questions, docs, device, max_length):
    questions = list(questions)
    docs = list(docs)

    inputs = tokenizer(
        questions,
        docs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    return outputs.logits.squeeze(-1)


# ==========================================
# 5. Evaluation
# ==========================================
def evaluate_epoch(tokenizer, model, dataloader, loss_fn, device, max_length):
    model.eval()

    total_loss = 0.0
    correct_pos_neg = 0
    correct_pos_hard = 0
    correct_hard_mid = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            s_w = compute_scores(tokenizer, model, batch["question"], batch["positive"], device, max_length)
            s_v = compute_scores(tokenizer, model, batch["question"], batch["hard_negative"], device, max_length)
            s_u = compute_scores(tokenizer, model, batch["question"], batch["soft_negative"], device, max_length)

            loss = loss_fn(s_w, s_v, s_u)
            total_loss += loss.item()

            correct_pos_neg += (s_w > s_u).sum().item()
            correct_pos_hard += (s_w > s_v).sum().item()
            correct_hard_mid += (s_v > (s_w + s_u) / 2).sum().item()

            total += s_w.numel()

    avg_loss = total_loss / len(dataloader)

    metrics = {
        "pos_gt_soft_neg": correct_pos_neg / total if total > 0 else 0.0,
        "pos_gt_hard_neg": correct_pos_hard / total if total > 0 else 0.0,
        "hard_gt_mid": correct_hard_mid / total if total > 0 else 0.0,
    }

    model.train()
    return avg_loss, metrics


# ==========================================
# 6. Plot
# ==========================================
def plot_and_save_loss(train_losses, val_losses, output_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"📈 График сохранён: {output_path}")


# ==========================================
# 7. One training run
# ==========================================
def train_one_config(
    args,
    train_df,
    val_df,
    loss_params,
    run_dir,
    device
):
    set_seed(args.seed)

    train_dataset = TripletDataset(train_df)
    val_dataset = TripletDataset(val_df)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    loss_fn = OrderedTripletLoss(**loss_params)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []
    val_metrics_history = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🚀 Запуск конфигурации loss: {loss_params}")
    print(f"📁 Run dir: {run_dir}")
    print("=" * 80)

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [TRAIN]"
        )

        for batch in pbar:
            optimizer.zero_grad()

            s_w = compute_scores(tokenizer, model, batch["question"], batch["positive"], device, args.max_length)
            s_v = compute_scores(tokenizer, model, batch["question"], batch["hard_negative"], device, args.max_length)
            s_u = compute_scores(tokenizer, model, batch["question"], batch["soft_negative"], device, args.max_length)

            loss = loss_fn(s_w, s_v, s_u)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss, val_metrics = evaluate_epoch(
            tokenizer,
            model,
            val_loader,
            loss_fn,
            device,
            args.max_length
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_metrics_history.append(val_metrics)

        print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val   Loss: {avg_val_loss:.4f}")
        print(f"   Val Acc [pos>soft_neg]: {val_metrics['pos_gt_soft_neg']:.1%}")
        print(f"   Val Acc [pos>hard_neg]: {val_metrics['pos_gt_hard_neg']:.1%}")
        print(f"   Val Acc [hard>mid]:     {val_metrics['hard_gt_mid']:.1%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            best_model_dir = os.path.join(run_dir, "best_model")
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            print(f"   ✅ Новая лучшая модель сохранена: {best_model_dir}")
        else:
            patience_counter += 1
            print(f"   ⚠️ Validation loss не улучшился. Patience: {patience_counter}/{args.patience}")

        print("-" * 80)

        if patience_counter >= args.patience:
            print(f"🛑 Early stopping на эпохе {epoch + 1}")
            break

    history_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_pos_gt_soft_neg": [m["pos_gt_soft_neg"] for m in val_metrics_history],
        "val_pos_gt_hard_neg": [m["pos_gt_hard_neg"] for m in val_metrics_history],
        "val_hard_gt_mid": [m["hard_gt_mid"] for m in val_metrics_history],
    })

    history_path = os.path.join(run_dir, "loss_history.csv")
    history_df.to_csv(history_path, index=False)

    plot_path = os.path.join(run_dir, "loss_curve.png")
    plot_and_save_loss(train_losses, val_losses, plot_path)

    config_path = os.path.join(run_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "train_csv": args.train_csv,
                "val_csv": args.val_csv,
                "loss_params": loss_params,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "max_length": args.max_length,
                "seed": args.seed,
                "warmup_ratio": args.warmup_ratio,
                "patience": args.patience,
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    return {
        "run_dir": run_dir,
        "best_model_dir": os.path.join(run_dir, "best_model"),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        **loss_params
    }


# ==========================================
# 8. Main
# ==========================================
def main():
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="./finetuned_reranker_grid_search")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"🚀 Device: {device}")
    print(f"📌 Seed: {args.seed}")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    print(f"📊 Train examples: {len(train_df)}")
    print(f"📊 Validation examples: {len(val_df)}")

    train_name = os.path.splitext(os.path.basename(args.train_csv))[0]
    val_name = os.path.splitext(os.path.basename(args.val_csv))[0]

    experiment_dir = os.path.join(
        args.output_dir,
        f"{train_name}__{val_name}"
    )

    os.makedirs(experiment_dir, exist_ok=True)

    # ==========================================
    # Grid search по параметрам функции потерь
    # ==========================================
    loss_grid = {
        "w_pos_neg": [2.5, 4.0, 6.0],
        "w_pos_hard": [1.5, 2.0, 2.5],
        "w_hard_mid": [0.5, 1.0, 1.5],
        "margin": [0.05, 0.1, 0.15]
    }

    keys = list(loss_grid.keys())
    combinations = list(itertools.product(*[loss_grid[k] for k in keys]))

    print(f"🔎 Всего конфигураций grid search: {len(combinations)}")

    all_results = []

    for i, values in enumerate(combinations, start=1):
        loss_params = dict(zip(keys, values))

        run_name = (
            f"run_{i:02d}"
            f"_wpn{loss_params['w_pos_neg']}"
            f"_wph{loss_params['w_pos_hard']}"
            f"_whm{loss_params['w_hard_mid']}"
            f"_m{loss_params['margin']}"
        )

        run_dir = os.path.join(experiment_dir, run_name)

        result = train_one_config(
            args=args,
            train_df=train_df,
            val_df=val_df,
            loss_params=loss_params,
            run_dir=run_dir,
            device=device
        )

        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(experiment_dir, "grid_search_results.csv")
    results_df.to_csv(results_path, index=False)

    best_result = results_df.sort_values("best_val_loss").iloc[0]

    summary_path = os.path.join(experiment_dir, "best_config.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            best_result.to_dict(),
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 80)
    print("✅ GRID SEARCH ЗАВЕРШЁН")
    print(f"📄 Все результаты: {results_path}")
    print(f"🏆 Лучшая конфигурация:")
    print(f"   best_val_loss: {best_result['best_val_loss']:.4f}")
    print(f"   best_epoch:    {int(best_result['best_epoch'])}")
    print(f"   w_pos_neg:     {best_result['w_pos_neg']}")
    print(f"   w_pos_hard:    {best_result['w_pos_hard']}")
    print(f"   w_hard_mid:    {best_result['w_hard_mid']}")
    print(f"   margin:        {best_result['margin']}")
    print(f"💾 Лучшая модель: {best_result['best_model_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()