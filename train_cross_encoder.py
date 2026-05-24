import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # ✅ Для графиков

# ==========================================
# 1. Кастомная функция потерь
# ==========================================
class OrderedTripletLoss(nn.Module):
    def __init__(self, w_pos_neg=4.0, w_pos_hard=2.0, w_hard_mid=1.0, margin=0.1):
        super().__init__()
        self.w_pos_neg = w_pos_neg
        self.w_pos_hard = w_pos_hard
        self.w_hard_mid = w_hard_mid
        self.margin = margin

    def forward(self, s_w, s_v, s_u):
        v1 = s_u - s_w
        v3 = s_v - s_w
        v2 = (s_w + s_u) / 2 - s_v
        total_weight = self.w_pos_neg + self.w_pos_hard + self.w_hard_mid
        return (self.w_pos_neg * F.softplus(v1 - self.margin) +
                self.w_pos_hard * F.softplus(v3 - self.margin) +
                self.w_hard_mid * F.softplus(v2 - self.margin)).mean() / total_weight

# ==========================================
# 2. Dataset
# ==========================================
class TripletDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)  # Сбрасываем индексы после split
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "question": str(row["question"]),
            "answer": str(row["answer"]),           # positive (w)
            "hard_negative": str(row["hard_negative"]),  # hard negative (v)
            "negative": str(row["negative"])         # negative (u)
        }

# ==========================================
# 3. Функция скориинга батча
# ==========================================
def compute_scores(tokenizer, model, questions, docs, device):
    questions, docs = list(questions), list(docs)
    inputs = tokenizer(questions, docs, padding=True, truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.logits.squeeze(-1)

# ==========================================
# 4. Оценка на валидации/тесте
# ==========================================
def evaluate_epoch(tokenizer, model, dataloader, loss_fn, device):
    """Оценивает средний лосс и метрики на датасете"""
    model.eval()
    total_loss = 0.0
    metrics = {"c1": 0, "c3": 0, "c2": 0, "n": 0}  # c1=f(w)>f(u), c3=f(w)>f(v), c2=f(v)>mid
    
    with torch.no_grad():
        for batch in dataloader:
            s_w = compute_scores(tokenizer, model, batch["question"], batch["answer"], device)
            s_v = compute_scores(tokenizer, model, batch["question"], batch["hard_negative"], device)
            s_u = compute_scores(tokenizer, model, batch["question"], batch["negative"], device)
            
            loss = loss_fn(s_w, s_v, s_u)
            total_loss += loss.item()
            
            metrics["c1"] += (s_w > s_u).float().sum().item()
            metrics["c3"] += (s_w > s_v).float().sum().item()
            metrics["c2"] += (s_v > (s_w + s_u)/2).float().sum().item()
            metrics["n"] += s_w.numel()
    
    model.train()  # Возвращаем модель в режим обучения
    avg_loss = total_loss / len(dataloader)
    return avg_loss, {k: (v / metrics["n"] if k != "n" else v) for k, v in metrics.items()}

# ==========================================
# 5. Построение и сохранение графика
# ==========================================
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

# ==========================================
# 6. Основной цикл обучения
# ==========================================
def main():
    # === НАСТРОЙКИ ===
    CSV_PATH = "qa_cross_mined_dataset.csv"
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
    OUTPUT_DIR = "./msmarco-minilm-finetuned-qa-cross-mined"
    PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")  # Куда сохранять график
    
    BATCH_SIZE = 8
    EPOCHS = 3
    LR = 1e-5
    GRAD_CLIP = 1.0
    TEST_SIZE = 0.2  # 20% данных в тест
    RANDOM_STATE = 42
    WEIGHT_DECAY = 0.02
    
    # === УСТРОЙСТВО ===
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # === ЗАГРУЗКА И РАЗДЕЛЕНИЕ ДАННЫХ ===
    df = pd.read_csv(CSV_PATH)
    print(f"📊 Всего примеров: {len(df)}")
    
    # Разделяем на train/test с сохранением распределения
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")

    test_df.to_csv("test_dataset.csv", index=False)
    train_df.to_csv("train_dataset.csv", index=False)

    print(f"✅ Тестовый и тренировочный датасеты сохранены в test_dataset.csv и train_dataset.csv")
    
    # === DATASETS И DATALOADERS ===
    train_dataset = TripletDataset(train_df)
    test_dataset = TripletDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # === МОДЕЛЬ И ОПТИМИЗАТОР ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=WEIGHT_DECAY)  # weight_decay для регуляризации
    loss_fn = OrderedTripletLoss(w_pos_neg=4.0, w_pos_hard=2.0, w_hard_mid=1.0, margin=0.1)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10, 
        num_training_steps=total_steps
    )
    
    # === КОНТЕЙНЕРЫ ДЛЯ ЛОГИРОВАНИЯ ===
    train_losses = []
    test_losses = []
    
    print(f"📚 Batches: train={len(train_loader)}, test={len(test_loader)} | Steps: {total_steps}")
    print("-" * 60)
    
    # === ЦИКЛ ОБУЧЕНИЯ ===
    for epoch in range(EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        epoch_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            s_w = compute_scores(tokenizer, model, batch["question"], batch["answer"], device)
            s_v = compute_scores(tokenizer, model, batch["question"], batch["hard_negative"], device)
            s_u = compute_scores(tokenizer, model, batch["question"], batch["negative"], device)
            
            loss = loss_fn(s_w, s_v, s_u)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ---------- TEST ----------
        avg_test_loss, test_metrics = evaluate_epoch(tokenizer, model, test_loader, loss_fn, device)
        test_losses.append(avg_test_loss)
        
        # ---------- ЛОГИРОВАНИЕ ----------
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Test  Loss: {avg_test_loss:.4f}")
        print(f"   Test Metrics:")
        print(f"      • f(w)>f(u) [pos>neg]:     {test_metrics['c1']:.1%}")
        print(f"      • f(w)>f(v) [pos>hard]:    {test_metrics['c3']:.1%}")
        print(f"      • f(v)>(w+u)/2 [hard>mid]: {test_metrics['c2']:.1%}")
        
        # Проверка на переобучение
        if epoch > 0 and avg_test_loss > train_losses[-2] * 1.2:
            print("   ⚠️  Warning: Test loss rising — possible overfitting!")
        
        print("-" * 60)
    
    # === СОХРАНЕНИЕ МОДЕЛИ ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Модель сохранена в: {OUTPUT_DIR}")
    
    # === СОХРАНЕНИЕ ЛОССОВ В CSV ===
    loss_df = pd.DataFrame({
        "epoch": range(1, EPOCHS + 1),
        "train_loss": train_losses,
        "test_loss": test_losses
    })
    loss_df.to_csv(os.path.join(OUTPUT_DIR, "loss_history.csv"), index=False)
    print(f"📄 История лоссов сохранена в: {OUTPUT_DIR}/loss_history.csv")
    
    # === ПОСТРОЕНИЕ ГРАФИКА ===
    plot_and_save_loss(train_losses, test_losses, PLOT_PATH)
    
    # === ФИНАЛЬНЫЙ ОТЧЁТ ===
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"   Лучший test loss: {min(test_losses):.4f} (эпоха {test_losses.index(min(test_losses)) + 1})")
    if len(test_losses) > 1:
        delta = test_losses[-1] - test_losses[0]
        print(f"   Изменение test loss: {delta:+.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()