from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
import torch.nn as nn

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
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit mode
    bnb_4bit_quant_type="nf4",  # Choose a quantization type (e.g., "nf4" or "fp4")
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8,  # LoRA rank
    lora_alpha=32,  # scaling factor
    lora_dropout=0.05,  # dropout probability for LoRA layers
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
peft_model.print_trainable_parameters()
peft_model.cuda()  # Move the model to GPU


input_examples = []
with open("results.jsonl", "r") as f:
    pairs = [json.loads(pair) for pair in f.readlines()]
df = pd.DataFrame(pairs)
df["label"] = 1
train, test = train_test_split(df, test_size=0.1, random_state=31)

train = Dataset.from_pandas(train, preserve_index=False)
test = Dataset.from_pandas(test, preserve_index=False)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer)


def tokenize_function(examples):
    return tokenizer(examples["question"], examples["context"], truncation=True)


tokenized_dataset = train.map(tokenize_function, batched=True)

test_dataset = test.map(tokenize_function, batched=True)


def compute_loss(model, inputs, return_outputs=False, **kwargs):
    outputs = model(**inputs)
    loss_fct = nn.CrossEntropyLoss()
    logits = outputs.logits
    labels = inputs["label"]
    loss = loss_fct(logits, labels)
    return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./peft_finetuned_baai_multilingual_gemma2_danish_law",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
)
trainer = Trainer(
    model=peft_model,
    train_dataset=tokenized_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    data_collator=data_collator,
    compute_loss_func=compute_loss,
)

peft_model.config.use_cache = False
trainer.train()
peft_model.config.use_cache = True
