"""
Run this script to upgrade your existing dataset to the new schema.
Place this in D:\SLM\ and run: python upgrade_dataset.py
"""
import json
import sys
from pathlib import Path
 
 
def upgrade_row(row: dict) -> dict:
    msg = row["messages"]
    try:
        data = json.loads(msg[1]["content"])
    except Exception:
        return row
 
    if data.get("intent") == "expense":
        for item in data.get("items", []):
            if not item.get("payment_method"):
                item["payment_method"] = "cash"
            if "remarks" not in item:
                item["remarks"] = None
            if "datetime" not in item:
                item["datetime"] = None
            if "bill_no" not in item:
                item["bill_no"] = None
 
    row["messages"][1]["content"] = json.dumps(data, ensure_ascii=False)
    return row
 
 
def upgrade_file(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
 
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)
 
    rows = []
    errors = 0
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows.append(upgrade_row(row))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i}")
                errors += 1
 
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
 
    print(f"✅ Done! {len(rows)} rows upgraded, {errors} errors")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
 
 
if __name__ == "__main__":
    upgrade_file(
        "dataset/dataset_final.jsonl",
        "dataset/dataset_final_v2.jsonl"
    )
 
    # Verify
    with open("dataset/dataset_final_v2.jsonl", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    print(f"   Total rows: {len(lines)}")
 
    # Show sample
    sample = json.loads(lines[0])
    if sample["messages"][1]:
        content = json.loads(sample["messages"][1]["content"])
        if content.get("intent") == "expense":
            print(f"\nSample item keys: {list(content['items'][0].keys())}")