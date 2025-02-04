from sentence_transformers import SentenceTransformer, InputExample, losses, models
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import json

# 1. Define model and load tokenizer
model_name = "BAAI/bge-multilingual-gemma2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model in 8-bit mode (if supported by your hardware) to reduce memory usage.
base_model = AutoModel.from_pretrained(
    model_name,
    load_in_8bit=True,  # Note: requires bitsandbytes package and a supported GPU
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

# 3. Build the SentenceTransformer model
# Create a transformer module (which loads the model architecture and weights)
word_embedding_model = models.Transformer(model_name)
# Create a pooling module to compute sentence embeddings from token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# Combine both into a SentenceTransformer model
st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Replace the transformer’s underlying model with our LoRA-adapted version.
st_model._first_module().auto_model = peft_model

input_examples = []
with open("results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        question = item["question"]
        context = item["context"]
        # Create an InputExample with the pair [question, context] and a label.
        # The label here is set to 1.0 (indicating similarity); adjust as needed for your training objective.
        input_examples.append(InputExample(texts=[question, context], label=1.0))

# Optionally, split the data into training and test sets (here using 90% for training)
train_examples, test_examples = train_test_split(
    input_examples, test_size=0.1, random_state=42
)

# Create a DataLoader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# 5. Set up the loss function and fine-tune the model
# We use the CosineSimilarityLoss to encourage similar embeddings for the question and context pairs.
train_loss = losses.CosineSimilarityLoss(model=st_model)


st_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,  # adjust number of epochs as needed
    warmup_steps=10,  # adjust warmup steps based on your dataset size
    output_path="./lora-finetuned-bge-danish-law",  # directory to save the finetuned model
    show_progress_bar=True,
    use_amp=True,  # enable automatic mixed precision if your hardware supports it
)
