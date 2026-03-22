import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from src.utils.config import load_config, get_project_root
from dotenv import load_dotenv
import os

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

config     = load_config()
output_dir = Path(get_project_root()) / config["paths"]["outputs"]

train = load_dataset("json", data_files=str(output_dir / "train.jsonl"))["train"]
val   = load_dataset("json", data_files=str(output_dir / "val.jsonl"))["train"]
test  = load_dataset("json", data_files=str(output_dir / "test.jsonl"))["train"]

ds = DatasetDict({"train": train, "validation": val, "test": test})
ds.push_to_hub("Divb30/loan-risk-explanations", private=True)

print(f"Pushed: train={len(train)} | val={len(val)} | test={len(test)}")
