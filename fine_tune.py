from peft import LoraConfig, TaskType, get_peft_model
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
import json
from datasets import Dataset
import pandas as pd

# 1. Define model and load tokenizer
model_name = "BAAI/bge-multilingual-gemma2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit mode
    bnb_4bit_quant_type="nf4",  # Choose a quantization type (e.g., "nf4" or "fp4")
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the base model in 8-bit mode (if supported by your hardware) to reduce memory usage.
base_model = AutoModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Note: requires bitsandbytes package and a supported GPU
    device_map="auto",
)

# 2. Set up the LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # we're fine-tuning the model for feature extraction (embeddings)
    r=8,  # LoRA rank
    lora_alpha=32,  # scaling factor
    lora_dropout=0.1,  # dropout probability for LoRA layers
    bias="none",  # no bias adaptation
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # adjust as necessary for your model architecture
)

# Wrap the base model with LoRA
peft_model = get_peft_model(base_model, lora_config)
peft_model.cuda()  # Move the model to GPU


input_examples = []
with open("results.jsonl", "r") as f:
    pairs = [json.loads(pair) for pair in f.readlines()]
df = pd.DataFrame(pairs)
df["label"] = 1
train, test = train_test_split(df, test_size=0.1, random_state=31)

train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)

tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer)


def tokenize_function(examples):
    return tokenizer(examples["question"], examples["context"], truncation=True)


tokenized_dataset = train.map(tokenize_function, batched=True)


def compute_loss(model, inputs, return_outputs=False):
    outputs = model(**inputs)
    loss_fct = nn.CrossEntropyLoss()
    logits = outputs.logits
    labels = inputs["label"]
    loss = loss_fct(logits, labels)
    return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./peft_finetuned_baai_multilingual_gemma2_danish_law",
    per_device_train_batch_size=32,  # Adjust based on your GPU memory
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    fp16=True,  # Enable mixed precision training
    save_total_limit=2,
    evaluation_strategy="no",
)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_loss_func=compute_loss,
)
trainer.train()
