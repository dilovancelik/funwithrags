from sentence_transformers import SentenceTransformer, InputExample, losses, models
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig


model_name = "BAAI/bge-multilingual-gemma2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load in 8-bit precision
base_model = AutoModel.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",  # let HF automatically place weights on GPU(s)
)

with open("results.csv", "r") as f:
    pairs = [json.loads(pair) for pair in f.readlines()]
df = pd.DataFrame(pairs)
train, test = train_test_split(df, test_size=0.1, random_state=31)

train_examples = []
for _, row in train.iterrows():
    question = row["question"]
    answer = row["context"]
    train_examples.append(InputExample(texts=[question, answer], label=1.0))

test_examples = []
for _, row in test.iterrows():
    question = row["question"]
    answer = row["context"]
    test_examples.append(InputExample(texts=[question, answer], label=1.0))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=128)

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,  # LoRA rank
    lora_alpha=32,  # scaling factor
    lora_dropout=0.1,  # dropout on LoRA layers
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.cuda()


base_model._first_module().auto_model = peft_model
train_loss = losses.CosineSimilarityLoss(model=base_model)


base_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    output_path="./lora-bge-qa-checkpoint",
    show_progress_bar=True,
)
