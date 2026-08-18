from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

fs = HfFileSystem()
files = fs.ls("datasets/ai4bharat/MSMARCO-XI", detail=False)
print("Files in repo root:", files)

# Find validation parquet files
val_files = [f for f in files if "validation" in f or "data" in f]
print("Validation files:", val_files)

# List all parquet files recursively
all_files = fs.glob("datasets/ai4bharat/MSMARCO-XI/**/*.parquet")
print(f"Total parquet files: {len(all_files)}")
for f in all_files[:5]:
    print("  Parquet file:", f)

# Read 1 parquet file using PyArrow HfFileSystem!
if all_files:
    pf = pq.ParquetFile(all_files[0], filesystem=fs)
    print("Row groups in first file:", pf.num_row_groups)
    rg0 = pf.read_row_group(0)
    print("Rows in row group 0:", rg0.num_rows)
    rec = rg0.to_pydict()
    print("Record keys:", list(rec.keys()))
    print("Query sample:", rec['query'][0])
