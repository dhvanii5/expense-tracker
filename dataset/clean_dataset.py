import json

INPUT_FILE = "dataset_final_v3.jsonl"
OUTPUT_FILE = "cleaned_dataset.jsonl"

def split_json_objects(text):
    objs = []
    bracket_count = 0
    start = 0

    for i, char in enumerate(text):
        if char == "{":
            if bracket_count == 0:
                start = i
            bracket_count += 1
        elif char == "}":
            bracket_count -= 1
            if bracket_count == 0:
                objs.append(text[start:i+1])
    return objs


cleaned_data = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = split_json_objects(line)

        for part in parts:
            try:
                sample = json.loads(part)
                cleaned_data.append(sample)
            except:
                continue

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in cleaned_data:
        f.write(json.dumps(sample) + "\n")

print(f"✅ Fixed and split dataset: {len(cleaned_data)} samples")