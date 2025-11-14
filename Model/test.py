import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import time

class TestDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        df = pd.read_csv(csv_file)
        self.questions = df["prompt"].astype(str).tolist()
        self.responses = df["response"].astype(str).tolist()
        self.scores = torch.tensor(df["score"].values, dtype=torch.float)

        q_enc = self.tokenizer(
                self.questions,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
        )

        r_enc = self.tokenizer(
                self.responses,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
        )

        self.encodings = {
                "question_input": q_enc,
                "response_input": r_enc,
                "score": self.scores,
                "question_text": self.questions,
            }

    def __len__(self):
        return len(self.encodings["score"])

    def __getitem__(self, idx):
        return {
            "question_input": {k: v[idx] for k, v in self.encodings["question_input"].items()},
            "response_input": {k: v[idx] for k, v in self.encodings["response_input"].items()},
            "score": self.encodings["score"][idx],
            "question_text": self.encodings["question_text"][idx],
        }

def computeCorrelation(model, csv_path, batch_size, tokenizer_name, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    dataset = TestDataset(csv_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, targets, questions = [], [], []
    total_examples = len(dataset)
    total_inference_time = 0.0

    with torch.no_grad():
        for batch in dataloader:
            q_input = {k: v.to(device, non_blocking=True) for k, v in batch["question_input"].items()}
            r_input = {k: v.to(device, non_blocking=True) for k, v in batch["response_input"].items()}
            score_true = batch["score"].to(device, non_blocking=True)

            start_time = time.perf_counter()
            score_pred = model.model(q_input, r_input)
            end_time = time.perf_counter()
            total_inference_time += (end_time - start_time)


            preds.append(score_pred.cpu())
            targets.append(score_true.cpu())
            questions.extend(batch["question_text"])

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    df = pd.DataFrame({
        "question": questions,
        "pred": preds,
        "target": targets,
    })

    pearson_corr = pearsonr(df["pred"], df["target"])[0]

    df.to_csv("pred_vs_actual.csv", index=False)

    print(f"Pearson correlation: {pearson_corr:.4f}")
    avg_time_per_example = total_inference_time / total_examples
    print(f"Avg time for each example: {avg_time_per_example:.4f}")

    plt.figure(figsize=(10, 18))
    plt.xlabel("Actual scores")
    plt.ylabel("Predicted scores (per-prompt normalized)")
    plt.title(f"Predicted vs Actual (r={pearson_corr:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pred_vs_actual_perprompt.png", dpi=150, bbox_inches="tight")
    plt.close()

    df.to_csv("pred_vs_actual_perprompt.csv", index=False)

    return pearson_corr