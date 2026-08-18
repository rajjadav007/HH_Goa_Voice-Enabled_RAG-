import logging
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder

logging.basicConfig(level=logging.INFO)

print("--- INSPECTING ai4bharat/MSMARCO-XI ON HUGGING FACE ---")
configs = get_dataset_config_names("ai4bharat/MSMARCO-XI")
print("Configs available:", configs)

for cfg in configs[:3]:
    print(f"\n--- Config: '{cfg}' ---")
    splits = get_dataset_split_names("ai4bharat/MSMARCO-XI", config_name=cfg)
    print("Splits:", splits)
    builder = load_dataset_builder("ai4bharat/MSMARCO-XI", name=cfg)
    if builder.info.splits:
        for sname, sinfo in builder.info.splits.items():
            print(f"  Split '{sname}': {sinfo.num_examples} rows")
    print("Features:", builder.info.features)
