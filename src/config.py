from pathlib import Path

ROOT = Path("/kaggle/working/business_entity_resolution")
STUDENT_RESOURCE = Path("/kaggle/input/amazon-ml-2026-data/student_resource")  # match your actual dataset folder name — check via !ls /kaggle/input/amazon-ml-2026-data/ first
DATA_TRAIN = STUDENT_RESOURCE / "dataset" / "train"
DATA_TEST = STUDENT_RESOURCE / "dataset" / "test"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VAL_FRACTION = 0.18