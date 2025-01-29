import json

def clean_questions(question: str) -> (str, int):
    questions, processed = (question[question.find(":"):].replace("\n", ""), 1) if question.find(":") != -1 else (question.replace("\n", ""), 0)
    return questions, processed

with open("results.csv", "r") as f:
    pairs = f.readlines()

total_questions = len(pairs)
processed_questions = 0
results2 = []
with open("results2.csv", "w") as f:
    for pair in pairs:
        obj = json.loads(pair)
        obj["question"], processed = clean_questions(obj["question"])
        processed_questions += processed_questions
        f.write(f"{json.dumps(obj)}\n")

print(f"Processed {processed_questions} out of {total_questions}")
    
