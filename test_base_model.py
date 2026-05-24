import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class TripletDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {"q": str(row["question"]), "w": str(row["answer"]), "v": str(row["hard_negative"]), "u": str(row["negative"])}

def score_batch(tokenizer, model, queries, docs, device):
    inp = tokenizer(list(queries), list(docs), padding=True, truncation=True, max_length=256, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        return model(**inp).logits.squeeze(-1).cpu().numpy()

def compute_ndcg_mrr(scores, labels, k=3):
    sorted_idx = np.argsort(-scores)[:k]
    dcg = sum((2**labels[i] - 1) / np.log2(rank + 2) for rank, i in enumerate(sorted_idx))
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = sum((2**lab - 1) / np.log2(rank + 2) for rank, lab in enumerate(ideal_labels))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    pos_rank = np.where(sorted_idx == np.argmax(labels))[0][0] + 1
    mrr = 1.0 / pos_rank
    return ndcg, mrr

def main():
    csv_path = "enriched_dataset.csv"
    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    loader = DataLoader(TripletDataset(csv_path), batch_size=16, shuffle=False, num_workers=0)

    ndcg_list, mrr_list = [], []

    for batch in loader:
        s_w = score_batch(tokenizer, model, batch["q"], batch["w"], device)
        s_v = score_batch(tokenizer, model, batch["q"], batch["v"], device)
        s_u = score_batch(tokenizer, model, batch["q"], batch["u"], device)

        for sw, sv, su in zip(s_w, s_v, s_u):
            scores = np.array([sw, sv, su])
            labels = np.array([1, 0, 0])
            ndcg, mrr = compute_ndcg_mrr(scores, labels, k=3)
            ndcg_list.append(ndcg)
            mrr_list.append(mrr)

    print(f"Queries evaluated: {len(ndcg_list)}")
    print(f"MRR@3:   {np.mean(mrr_list):.4f}")
    print(f"NDCG@3:  {np.mean(ndcg_list):.4f}")

if __name__ == "__main__":
    main()