"""Local macro F0.5 scorer matching the competition's exact definition."""
import pandas as pd

def parse_id_list(s):
    if pd.isna(s) or s == "":
        return set()
    return set(x.strip() for x in str(s).split(",") if x.strip())

def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom

def score_entity(pred_ids: set, true_ids: set) -> float:
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    if len(pred_ids) == 0:
        return 0.0
    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return f_beta(precision, recall, beta=0.5)

def macro_f05(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> float:
    pred_map = dict(zip(pred_df["source1_entity_id"], pred_df["matched_entity_ids"]))
    scores = []
    for _, row in gold_df.iterrows():
        eid = row["source1_entity_id"]
        true_ids = parse_id_list(row["matched_entity_ids"])
        pred_ids = parse_id_list(pred_map.get(eid, ""))
        scores.append(score_entity(pred_ids, true_ids))
    return sum(scores) / len(scores)
