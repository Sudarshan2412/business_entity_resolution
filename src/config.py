from pathlib import Path

ROOT = Path("/content/business_entity_resolution")
STUDENT_RESOURCE = ROOT / "student_resource"
DATA_TRAIN = STUDENT_RESOURCE / "dataset" / "train"
DATA_TEST = STUDENT_RESOURCE / "dataset" / "test"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VAL_FRACTION = 0.18
