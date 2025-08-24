import json
import pandas as pd

with open("./datasets/validation_data/gemini_generated_data.json", "r", encoding="utf-8") as f:
    text = f.read()

data = json.loads(text)

print(f"Loaded {len(data)} entries from gemini_generated_data.json")

id = 1
qid = 1

documents = []
scores = []
val_data = []

for entry in data:
    query = entry['query']
    scores.append({
        'id': qid,
        'query': query,
        'relevant_docs': []
    })
    val_data.append({
        'query_id': qid,
        'query': query
    })
    for doc in entry['pairs']:
        documents.append({
            'Id': id,
            'Title': doc['question'],
            'Body': doc['answer'],
            'Tags': ', '.join(doc['tags'])
        })
        scores[-1]['relevant_docs'].append((id, doc['relevance_score']))
        id += 1
    qid += 1


df = pd.DataFrame(documents)

df.to_csv("./datasets/validation_data/documents.csv", index=False)
print("Saved documents as csv file")

val_data = pd.DataFrame(val_data)

val_data.to_csv("./datasets/validation_data/val_data.csv", index=False)
print("Saved validation data as csv file")

with open("./datasets/validation_data/ground_truth_val.json", "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=4)

print("Saved scores as json file")


