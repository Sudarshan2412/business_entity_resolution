import numpy as np
import pandas as pd
import lightgbm as lgb
from src.features import FEATURE_COLUMNS
from src.scoring import macro_f05

def train_model(train_path, val_path):
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["label"]

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    return model, val_df

def tune_threshold(model, val_df, gt_df, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    val_df = val_df.copy()
    val_df["prob"] = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    val_gt = gt_df[gt_df["source1_entity_id"].isin(val_df["source1_entity_id"].unique())]

    best_t, best_score = None, -1
    for t in thresholds:
        keep = val_df[val_df["prob"] >= t]
        grouped = keep.groupby("source1_entity_id")["candidate_entity_id"].apply(
            lambda s: ",".join(sorted(set(s)))
        )
        pred_df = grouped.reset_index()
        pred_df.columns = ["source1_entity_id", "matched_entity_ids"]
        score = macro_f05(pred_df, val_gt)
        print(f"threshold {t:.2f}: F0.5 = {score:.4f}")
        if score > best_score:
            best_score, best_t = score, t
    print(f"BEST threshold: {best_t:.2f} -> F0.5 = {best_score:.4f}")
    return best_t, best_score
