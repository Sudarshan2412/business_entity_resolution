import pandas as pd
from src.config import DATA_TRAIN, RANDOM_SEED, VAL_FRACTION
from src.scoring import parse_id_list

def load_train():
    s1 = pd.read_csv(DATA_TRAIN / "train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(DATA_TRAIN / "train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(DATA_TRAIN / "train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(DATA_TRAIN / "train_ground_truth.tsv", sep="\t", dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    return s1, s2, s3, gt

def make_split(s1, gt, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    gt = gt.merge(s1[["entity_id", "country"]], left_on="source1_entity_id",
                  right_on="entity_id", how="left")
    gt["is_singleton"] = gt["matched_entity_ids"].apply(lambda s: len(parse_id_list(s)) == 0)
    gt["strata"] = gt["country"] + "_" + gt["is_singleton"].astype(str)

    val_ids = (
        gt.groupby("strata", group_keys=False)
        .apply(lambda g: g.sample(frac=val_fraction, random_state=seed))
        ["source1_entity_id"]
    )
    val_mask = gt["source1_entity_id"].isin(val_ids)
    return gt.loc[~val_mask].drop(columns=["entity_id", "is_singleton", "strata"]), \
           gt.loc[val_mask].drop(columns=["entity_id", "is_singleton", "strata"])
