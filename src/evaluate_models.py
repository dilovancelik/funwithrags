import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset
from sklearn.model_selection import train_test_split


with open("results_with_negative.jsonl", "r") as f:
    pairs = [json.loads(pair) for pair in f.readlines()]
df = pd.DataFrame(pairs)[["question", "context", "neg_context"]]

train, val = train_test_split(df, test_size=0.1, random_state=31)
dataset = Dataset.from_pandas(val, preserve_index=False)

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
        anchors=dataset["question"],
        positives=dataset["context"],
        negatives=dataset["neg_context"],
        name=f"{model_name}_eval",
    )
    result = evaluator(model)
    print(result)
    results.append(result)

with open("eval_results.jsonl", "a") as f:
    for result in results:
        f.write(f"{json.dumps(result)}\n")
