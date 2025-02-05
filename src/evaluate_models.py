from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from datasets import load_dataset
import torch
import gc

dataset = load_dataset("dilovancelik/danish_law_qa")
dataset = dataset["train"].train_test_split(test_size=0.05)

models_to_evaluate = [
    "sentence-transformers/all-distilroberta-v1",
    "Salesforce/SFR-Embedding-2_R",
    "Salesforce/SFR-Embedding-Mistral",
    "GritLM/GritLM-7B",
    "GritLM/GritLM-7B",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "nvidia/NV-Embed-v1",
    "intfloat/multilingual-e5-large",
]

with open("eval_results.jsonl", "r") as f:
    saved_results = f.readlines()

for model_name in models_to_evaluate:
    for result in saved_results:
        if model_name in result:
            models_to_evaluate.remove(model_name)

for model_name in models_to_evaluate:
    model = SentenceTransformer(model_name)
    evaluator = TripletEvaluator(
        anchors=dataset["test"]["question"],
        positives=dataset["test"]["context"],
        negatives=dataset["test"]["neg_context"],
        name=f"{model_name}_eval",
        show_progress_bar=True,
    )
    result = evaluator(model)
    print(result)

    with open("eval_results.jsonl", "a") as f:
        f.write(f"{result}\n")

    torch.cuda.empty_cache()
    gc.collect()
