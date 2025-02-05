import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from datasets import load_dataset

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

results = []
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
        for result in results:
            f.write(f"{json.dumps(result)}\n")
