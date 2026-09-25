from pathlib import Path

ROOT = Path("/kaggle/working/business_entity_resolution")
STUDENT_RESOURCE = Path("/kaggle/input/datasets/sudarshan24/amazon-ml-2026-data/student_resource")   # <-- fix this line based on Step 1's output
DATA_TRAIN = STUDENT_RESOURCE / "dataset" / "train"
DATA_TEST = STUDENT_RESOURCE / "dataset" / "test"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VAL_FRACTION = 0.18
