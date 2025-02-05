from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
import pandas as pd
import json
from sklearn.model_selection import train_test_split


base_model = SentenceTransformer("BAAI/bge-multilingual-gemma2")


with open("results.jsonl", "r") as f:
    pairs = [json.loads(pair) for pair in f.readlines()]
df = pd.DataFrame(pairs)
df["label"] = 1
train, test = train_test_split(df, test_size=0.1, random_state=31)

train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)

loss = MultipleNegativesRankingLoss(base_model)
