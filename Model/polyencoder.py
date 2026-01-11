import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader



class PolyEncoder(nn.Module):
    def __init__(self, model_name, poly_m):
        super().__init__()
        self.model_name = model_name
        # BERT model variant
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        # Poly code count
        self.poly_m = poly_m
        self.dropout = nn.Dropout(0.2)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)

        # Poly-code initialization
        self.poly_codes = nn.Embedding(poly_m, self.hidden_size)
        self.register_buffer("poly_code_ids", torch.arange(poly_m))

        # Regression head for final scoring
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 1028),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1028, 1)
        )

    # CLS encoding using BERT
    def encode_candidate(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]
        return cls_vec

    # Poly-context computation
    def encode_context(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        token_embeds = outputs.last_hidden_state

        poly_codes = self.poly_codes(self.poly_code_ids)
        poly_codes = poly_codes.unsqueeze(0).expand(token_embeds.size(0), -1, -1)

        # Attend codes to context, attended = poly-context
        attn_scores = torch.matmul(poly_codes, token_embeds.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, token_embeds)
        attended = self.dropout(attended)
        return attended

    def forward(self, context_inputs, candidate_inputs):
        context_vecs = self.encode_context(**context_inputs)
        candidate_vec = self.encode_candidate(**candidate_inputs)
        context_vecs = F.normalize(context_vecs, dim=-1)
        candidate_vec = F.normalize(candidate_vec, dim=-1)

        # QKV for attention
        query = candidate_vec.unsqueeze(1)
        key = value = context_vecs

        attended, _ = self.cross_attn(query, key, value)
        context_pooled = self.norm(attended.squeeze(1))

        # Poly-context + candidate vec * 2
        combined = torch.cat((
            context_pooled,
            candidate_vec,
            candidate_vec,
        ), dim=1)

        # Single scalar score
        score = self.reg_head(combined)

        return score.squeeze(-1)

    def getName(self):
        return self.model_name

# Lightning wrapper for training
class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model_name, logger, poly_m, lr=1e-5):
        super().__init__()
        self.model_name = model_name
        self.model = PolyEncoder(model_name, poly_m)
        self.lr = lr
        self.val_pearson_ema = None
        self.ema_alpha = 0.5
        self.wandb_logger = logger

        self.val_preds = []
        self.val_labels = []

    def forward(self, batch):
        pred = self.model(batch['question_input'], batch['response_input'])
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        label = batch['label'].float()
        loss = F.smooth_l1_loss(pred, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        label = batch['label'].float()
        loss = F.smooth_l1_loss(pred, label)
        self.log("val_loss", loss, prog_bar=True)

        self.val_preds.append(pred.detach().cpu())
        self.val_labels.append(label.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()

        pearson_corr, _ = pearsonr(preds, labels)

        # Keep track of both correlation curve and correlation EMA curve
        if self.val_pearson_ema is None:
            self.val_pearson_ema = pearson_corr
        else:
            self.val_pearson_ema = (
                    self.ema_alpha * pearson_corr +
                    (1 - self.ema_alpha) * self.val_pearson_ema
            )
        self.log("val_pearson", pearson_corr, prog_bar=True)
        self.log("val_pearson_ema", self.val_pearson_ema, prog_bar=True)
        correlation = computeCorrelation(self, "/home/sam/datasets/test.csv", 16, "roberta-base", 128)

        self.wandb_logger.log_metrics({"correlation": correlation})

        self.val_preds = []
        self.val_labels = []

    # Optimizer LR, smaller lr for the head parameters
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.model.encoder.parameters(), "lr": self.lr},
            {"params": [
                *self.model.poly_codes.parameters(),
                *self.model.reg_head.parameters(),
                *self.model.cross_attn.parameters(),
                *self.model.norm.parameters()
            ], "lr": self.lr / 5},
        ])
        return optimizer

def computeCorrelation(model, csv_path, batch_size, tokenizer_name, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model.model_name, use_fast=True)
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