import ollama
import json
import random
import uuid
from tqdm import tqdm

PROMPT_TEMPLATE = """\
Kontekst er nedenfor.

---------------------
{context_str}
---------------------

Givet den givne kontekst og ingen anden viden.
Genere op til 5 emner som kan beskrive konteksten,. 
Hvis der ikke er emner som let kan beskrive konteksten, besvar med <|NAN|>

Du må kun svarer med emnerne formattet skal være: Emne 1|Emne 2|...|Emne n|
"""

"""
base_model = SentenceTransformer("BAAI/bge-multilingual-gemma2")
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,  # LoRA rank
    lora_alpha=32,  # scaling factor
    lora_dropout=0.1,  # dropout on LoRA layers
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
base_model.add_adapter(lora_config)
"""


def load_corpus(file_path: str, val_percentage: float):
    with open(file_path, "r") as f:
        docs = f.readlines()

    random.shuffle(docs)
    train_corpus = docs[: int(len(docs) * (1 - val_percentage))]
    validation_corpus = docs[int(len(docs) * (1 - val_percentage)) :]

    return train_corpus, validation_corpus


train_nodes, val_nodes = load_corpus("taler.txt", 0.2)

dataset = {"queries": {}, "corpus": {}, "relevant_docs": {}, "mode": "text"}

with tqdm(total=len(train_nodes), desc="Generating Queries") as fpbar:
    i = 0
    for doc in train_nodes:
        prompt = PROMPT_TEMPLATE.format(context_str=doc)
        res = ollama.chat("llama3.3", messages=[{"role": "user", "content": prompt}])
        content_id = uuid.uuid4()
        dataset["corpus"][content_id] = doc
        for query in res.message.content.split("|"):
            query_id = uuid.uuid4()
            dataset["queries"][query_id] = query
            dataset["relevant_docs"][query_id] = [content_id]
        fpbar.update(1)
        i += 1
        if i % 100 == 0:
            with open("train_data.json", "w") as f:
                json.dump(dataset, f)

with open("train_data.json", "w") as f:
    json.dump(dataset, f)

"""
train_dataset = generate_qa_embedding_pairs(
    llm=llm,
    nodes=train_nodes[:10],
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
"""
