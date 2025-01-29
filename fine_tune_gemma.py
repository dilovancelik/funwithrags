from sentence_transformers import SentenceTransformer, InputExample, losses
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import json

with open("results.csv", "r") as f:
  pairs = [json.loads(pair) for pair in f.readlines()]
df = pd.DataFrame(pairs)
train, test = train_test_split(df, test_size=0.1, random_state=31)

# Prepare train examples
train_examples = []
for _, row in train.iterrows():
    question = row["question"]
    answer = row["context"]
    train_examples.append(InputExample(
        texts=[question, answer],
        label=1.0
    ))

# (Optional) prepare test examples for evaluation later
test_examples = []
for _, row in test.iterrows():
    question = row["question"]
    answer = row["context"]
    test_examples.append(InputExample(
        texts=[question, answer],
        label=1.0
    ))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=128)

model_id = "BAAI/bge-multilingual-gemma2"
model = SentenceTransformer(model_id, model_kwargs={"torch_dtype": torch.float16})
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=int(len(train_dataloader)*0.1),
    show_progress_bar=True
)

model.save("my_finetuned_gemma2")
