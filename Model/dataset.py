from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer
import pandas as pd
import torch

class CreativityScoringDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name, max_length=128):
        self.data = pd.read_csv(csv_file, header=0).fillna("")
        self.data["prompt"] = self.data["prompt"].astype(str)
        self.data["response"] = self.data["response"].astype(str)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        question = row["prompt"]
        response = row["response"]

        q_inputs = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        r_inputs = self.tokenizer(
            response,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        row_label = float(row["score"])

        # Return question + response + score

        return {
            "question_input": {k: v.squeeze(0) for k, v in q_inputs.items()},
            "response_input": {k: v.squeeze(0) for k, v in r_inputs.items()},
            "label": torch.tensor(row_label, dtype=torch.float)
        }
