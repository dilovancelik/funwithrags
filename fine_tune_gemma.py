from typing import List
from pathlib import Path
import random
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, TaskType
from llama_index.llms.ollama import Ollama
from llama_index.finetuning import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset,
    SentenceTransformersFinetuneEngine,
)
from llama_index.core.schema import TextNode


PROMPT_TEMPLATE = """\
Kontekst er nedenfor.

---------------------
{context_str}
---------------------

Givet den givne kontekst og ingen anden viden.
Generer kun spørgsmål baseret på nedenstående forespørgsel.

Du er en advokat / jurist. Din opgave er at opstille \
{num_questions_per_chunk} spørgsmål til en eksamen. \
Spørgsmålene skal være forskellige og dække hele konteksten. \
Begræns spørgsmålene til den givne kontekst. \
Det er vigtigt at du kun svarere med spørgsmålene og intet andet. \
F.eks må du IKKE skrive her er spørgsmålene \
"""

llm = Ollama(model="llama3.3")

base_model = SentenceTransformer("BAAI/bge-multilingual-gemma2")
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,  # LoRA rank
    lora_alpha=32,  # scaling factor
    lora_dropout=0.1,  # dropout on LoRA layers
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

peft_model = get_peft_model(base_model, lora_config)


def load_corpus(file_path: str, val_percentage: float):
    with open(file_path, "r") as f:
        docs = f.readlines()

    random.shuffle(docs)
    train_corpus = docs[: int(len(docs) * (1 - val_percentage))]
    validation_corpus = docs[int(len(docs) * (1 - val_percentage)) :]

    train_nodes: List[TextNode] = []
    for doc in train_corpus:
        node = TextNode()
        node.set_content(doc)
        train_nodes.append(node)

    val_nodes: List[TextNode] = []
    for doc in validation_corpus:
        node = TextNode()
        node.set_content(doc)
        val_nodes.append(node)

    print(f"Parsed {len(train_nodes) + len(val_nodes)} nodes")

    return train_nodes, val_nodes


train_nodes, val_nodes = load_corpus("documents.csv", 0.1)


if Path("train_dataset.json").exists() and Path("val_dataset.json").exists():
    print("loading existing embedding qa dataset")
    train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
    val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")
else:
    print("creating embedding qa dataset")
    train_dataset = generate_qa_embedding_pairs(
        llm=llm,
        nodes=train_nodes,
        num_questions_per_chunk=4,
        qa_generate_prompt_tmpl=PROMPT_TEMPLATE,
        output_path="train_dataset.json",
        verbose=False,
    )

    val_dataset = generate_qa_embedding_pairs(
        llm=llm,
        nodes=val_nodes,
        num_questions_per_chunk=4,
        qa_generate_prompt_tmpl=PROMPT_TEMPLATE,
        output_path="val_dataset.json",
        verbose=False,
    )

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model=peft_model,
    model_output_path="fine_tune_bge_multilinugal-gemma2-danish_law",
    val_dataset=val_dataset,
)

finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
embed_model
