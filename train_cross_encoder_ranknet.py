import os
os.environ["HF_HUB_ENABLE_XET"] = "0"
os.environ["HF_HOME"] = "/raid/iastafyev/hf_cache"

import argparse
import random
import json
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_pos, s_neg):
        return F.softplus(s_neg - s_pos).mean()


class PairDataset(Dataset):
    def __init__(self, df):
        required_columns = {"question", "positive", "hard_negative"}
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
            "hard_negative": str(row["hard_negative"])
        }


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


def evaluate_epoch(tokenizer, model, dataloader, loss_fn, device, max_length):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            s_pos = compute_scores(
                tokenizer,
                model,
                batch["question"],
                batch["positive"],
                device,
                max_length
            )

            s_neg = compute_scores(
                tokenizer,
                model,
                batch["question"],
                batch["hard_negative"],
                device,
                max_length
            )

            loss = loss_fn(s_pos, s_neg)
            total_loss += loss.item()

            correct += (s_pos > s_neg).sum().item()
            total += s_pos.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0

    model.train()
    return avg_loss, accuracy


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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./msmarco-minilm-finetuned-ranknet"
    )

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

    output_dir = os.path.join(
        args.output_dir,
        f"{train_name}__{val_name}"
    )

    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = PairDataset(train_df)
    val_dataset = PairDataset(val_df)

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

    loss_fn = RankNetLoss()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"📚 Batches: train={len(train_loader)}, val={len(val_loader)}")
    print(f"📚 Total steps: {total_steps} | Warmup steps: {warmup_steps}")
    print("-" * 80)

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [TRAIN]"
        )

        for batch in pbar:
            optimizer.zero_grad()

            s_pos = compute_scores(
                tokenizer,
                model,
                batch["question"],
                batch["positive"],
                device,
                args.max_length
            )

            s_neg = compute_scores(
                tokenizer,
                model,
                batch["question"],
                batch["hard_negative"],
                device,
                args.max_length
            )

            loss = loss_fn(s_pos, s_neg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip
            )

            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)

        avg_val_loss, val_accuracy = evaluate_epoch(
            tokenizer,
            model,
            val_loader,
            loss_fn,
            device,
            args.max_length
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val   Loss: {avg_val_loss:.4f}")
        print(f"   Val Accuracy [pos>hard]: {val_accuracy:.1%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            print(f"   ✅ Новая лучшая модель сохранена: {best_model_dir}")
        else:
            patience_counter += 1
            print(
                f"   ⚠️ Validation loss не улучшился. "
                f"Patience: {patience_counter}/{args.patience}"
            )

        print("-" * 80)

        if patience_counter >= args.patience:
            print(f"🛑 Early stopping на эпохе {epoch + 1}")
            break

    loss_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy_pos_gt_hard": val_accuracies
    })

    loss_history_path = os.path.join(output_dir, "loss_history.csv")
    loss_df.to_csv(loss_history_path, index=False)

    plot_path = os.path.join(output_dir, "loss_curve.png")
    plot_and_save_loss(train_losses, val_losses, plot_path)

    config = {
        "model_name": args.model_name,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "output_dir": output_dir,
        "best_model_dir": best_model_dir,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "seed": args.seed,
        "loss": "RankNetLoss = mean(softplus(s_neg - s_pos))"
    }

    config_path = os.path.join(output_dir, "run_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"📁 Эксперимент: {output_dir}")
    print(f"💾 Лучшая модель: {best_model_dir}")
    print(f"📄 История обучения: {loss_history_path}")
    print(f"📄 Конфигурация запуска: {config_path}")
    print(f"🏆 Лучший validation loss: {best_val_loss:.4f}")
    print(f"🏆 Лучшая эпоха: {best_epoch}")
    print("=" * 80)


if __name__ == "__main__":
    main()