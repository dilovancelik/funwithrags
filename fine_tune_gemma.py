from typing import List
import random
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
"""

llm = Ollama(model="llama3")


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

train_dataset = generate_qa_embedding_pairs(
    llm=llm,
    nodes=train_nodes,
    qa_generate_prompt_tmpl=PROMPT_TEMPLATE,
    output_path="train_dataset.json",
)
val_dataset = generate_qa_embedding_pairs(
    llm=llm,
    nodes=val_nodes,
    qa_generate_prompt_tmpl=PROMPT_TEMPLATE,
    output_path="val_dataset.json",
)
print(f"Train: {len(train_nodes)}, Validation: {len(val_nodes)}")

# [Optional] Load
train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-multilingual-gemma2",
    model_output_path="fine_tune_bge_multilinugal-gemma2-danish_law",
    val_dataset=val_dataset,
)

finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
embed_model
