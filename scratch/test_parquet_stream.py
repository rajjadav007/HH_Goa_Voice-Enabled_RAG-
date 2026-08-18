import pyarrow.parquet as pq
from huggingface_hub import hf_hub_url
import urllib.request

url = hf_hub_url("ai4bharat/MSMARCO-XI", filename="default/validation/0000.parquet", repo_type="dataset")
print("Downloading / streaming parquet file URL:", url)

# Inspect parquet schema and batches
req = urllib.request.urlopen(url)
pf = pq.ParquetFile(req)
print("Parquet file num row groups:", pf.num_row_groups)
print("Parquet schema:", pf.schema)

# Read 1 row group
rg = pf.read_row_group(0)
print("Row group 0 num rows:", rg.num_rows)
p_dict = rg.to_pydict()
print("Keys in row group:", list(p_dict.keys()))
print("First query:", p_dict.get("query", [""])[0])
