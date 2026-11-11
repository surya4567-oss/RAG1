import json
import sys

with open("vanillarag_evaluation_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else len(data)

for item in data[start:end]:
    print(f"{item['id']} [{item['category']}]: {item['query']}")
    print(f"   Expected: {item['expected_answer']}\n")

