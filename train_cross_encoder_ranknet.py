import os
os.environ['HF_HUB_ENABLE_XET'] = '0'
os.environ['HF_HOME'] = '/raid/iastafyev/hf_cache'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dotenv import load_dotenv

class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, s_pos, s_neg):
        return F.softplus(s_neg - s_pos).mean()

class PairDataset(Dataset):
    def __init__(self, df):
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

def compute_scores(tokenizer, model, questions, docs, device):
    questions, docs = list(questions), list(docs)
    inputs = tokenizer(questions, docs, padding=True, truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.logits.squeeze(-1)

def evaluate_epoch(tokenizer, model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            s_pos = compute_scores(tokenizer, model, batch["question"], batch["positive"], device)
            s_neg = compute_scores(tokenizer, model, batch["question"], batch["hard_negative"], device)
            loss = loss_fn(s_pos, s_neg)
            total_loss += loss.item()
            correct += (s_pos > s_neg).sum().item()
            total += s_pos.numel()
    model.train()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def plot_and_save_loss(train_losses, test_losses, output_path="loss_curve.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if test_losses:
        plt.plot(epochs, test_losses, 'r--', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📈 График сохранён в: {output_path}")

def main():
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    CSV_PATH = "passed_oracle_method_vlidated_df.csv"
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
    OUTPUT_DIR = f"./msmarco-minilm-finetuned-ranknet-{os.path.splitext(CSV_PATH)[0]}"
    PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")
    BATCH_SIZE = 8
    EPOCHS = 3
    LR = 2e-5
    GRAD_CLIP = 1.0
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"📊 Всего примеров: {len(df)}")
    
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")
    
    test_df.to_csv(f"test_dataset_{os.path.splitext(CSV_PATH)[0]}.csv", index=False)
    train_df.to_csv(f"train_dataset_{os.path.splitext(CSV_PATH)[0]}.csv", index=False)
    print(f"✅ Тестовый и тренировочный датасеты сохранены")
    
    train_dataset = PairDataset(train_df)
    test_dataset = PairDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=0.01)
    loss_fn = RankNetLoss()
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    
    train_losses = []
    test_losses = []
    
    print(f"📚 Batches: train={len(train_loader)}, test={len(test_loader)} | Steps: {total_steps}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        
        for batch in pbar:
            optimizer.zero_grad()
            s_pos = compute_scores(tokenizer, model, batch["question"], batch["positive"], device)
            s_neg = compute_scores(tokenizer, model, batch["question"], batch["hard_negative"], device)
            loss = loss_fn(s_pos, s_neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        avg_test_loss, test_acc = evaluate_epoch(tokenizer, model, test_loader, loss_fn, device)
        test_losses.append(avg_test_loss)
        
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Test  Loss: {avg_test_loss:.4f}")
        print(f"   Test Accuracy [pos>hard]: {test_acc:.1%}")
        
        if epoch > 0 and avg_test_loss > train_losses[-2] * 1.2:
            print("   ⚠️  Warning: Test loss rising — possible overfitting!")
        
        print("-" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Модель сохранена в: {OUTPUT_DIR}")
    
    loss_df = pd.DataFrame({
        "epoch": range(1, EPOCHS + 1),
        "train_loss": train_losses,
        "test_loss": test_losses
    })
    loss_df.to_csv(os.path.join(OUTPUT_DIR, "loss_history.csv"), index=False)
    print(f"📄 История лоссов сохранена в: {OUTPUT_DIR}/loss_history.csv")
    
    plot_and_save_loss(train_losses, test_losses, PLOT_PATH)
    
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"   Лучший test loss: {min(test_losses):.4f} (эпоха {test_losses.index(min(test_losses)) + 1})")
    if len(test_losses) > 1:
        delta = test_losses[-1] - test_losses[0]
        print(f"   Изменение test loss: {delta:+.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
