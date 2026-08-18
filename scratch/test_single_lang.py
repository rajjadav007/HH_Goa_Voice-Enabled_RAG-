from datasets import load_dataset

print("--- TESTING SINGLE LANGUAGE STREAMING (hin = Hindi) ---")
ds = load_dataset("ai4bharat/MSMARCO-XI", data_files="validation/hinval.parquet", split="train", streaming=True)

count = 0
for rec in ds:
    count += 1
    if count == 1:
        print("Sample Hindi Record:")
        print("  Query ID:", rec.get("query_id"))
        print("  Query:", rec.get("query"))
        print("  Eng_Query:", rec.get("Eng_Query"))
        print("  Answer:", rec.get("Answer"))
        print("  Passages count:", len(rec.get("passages", {}).get("Translated_passages", [])))
    if count >= 10:
        break

print(f"Successfully streamed {count} records without memory errors!")
