from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

print("Downloading single validation parquet file (hinval.parquet)...")
local_file = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="validation/hinval.parquet", repo_type="dataset")
print(f"Downloaded to local path: {local_file}")

pf = pq.ParquetFile(local_file)
print(f"Num row groups: {pf.num_row_groups}, Num rows: {pf.metadata.num_rows}")

# Stream record batches in tiny batch_size=50
count = 0
for batch in pf.iter_batches(batch_size=50):
    pydict = batch.to_pydict()
    for i in range(len(pydict["query_id"])):
        count += 1
        if count == 1:
            print("\nSample Record 1:")
            print("  Query ID:", pydict["query_id"][i])
            print("  Query:", pydict["query"][i])
            print("  Eng_Query:", pydict["Eng_Query"][i])
            print("  Passages count:", len(pydict["passages"][i]["Translated_passages"]))
        if count >= 10:
            break
    if count >= 10:
        break

print(f"\nSuccessfully read {count} records using small PyArrow batch_size without memory issues!")
