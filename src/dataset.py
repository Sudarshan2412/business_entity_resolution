import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.scoring import parse_id_list
from src.data_split import make_split

def build_gt_lookup(gt_df):
    return {eid: parse_id_list(m) for eid, m in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"])}

def label_and_split(features_path, s1_df, gt_df, train_out, val_out,
                     batch_size=2_000_000, neg_per_pos=10, seed=42):
    train_gt, val_gt = make_split(s1_df, gt_df)
    train_entities = set(train_gt["source1_entity_id"])
    val_entities = set(val_gt["source1_entity_id"])
    gt_lookup = build_gt_lookup(gt_df)
    rng = np.random.default_rng(seed)

    pf = pq.ParquetFile(features_path)
    train_writer = val_writer = None
    train_pos = train_neg_kept = train_neg_seen = val_rows = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        df["label"] = [
            1 if cid in gt_lookup.get(eid, ()) else 0
            for eid, cid in zip(df["source1_entity_id"], df["candidate_entity_id"])
        ]

        vdf = df[df["source1_entity_id"].isin(val_entities)]
        if len(vdf):
            t = pa.Table.from_pandas(vdf, preserve_index=False)
            if val_writer is None:
                val_writer = pq.ParquetWriter(val_out, t.schema, compression="snappy")
            val_writer.write_table(t)
            val_rows += len(vdf)

        tdf = df[df["source1_entity_id"].isin(train_entities)]
        pos = tdf[tdf["label"] == 1]
        neg = tdf[tdf["label"] == 0]
        train_pos += len(pos)
        train_neg_seen += len(neg)
        keep_n = min(len(neg), len(pos) * neg_per_pos)
        neg_kept = neg.iloc[rng.choice(len(neg), size=keep_n, replace=False)] if keep_n > 0 else neg.iloc[0:0]
        train_neg_kept += len(neg_kept)

        out = pd.concat([pos, neg_kept], ignore_index=True)
        if len(out):
            t = pa.Table.from_pandas(out, preserve_index=False)
            if train_writer is None:
                train_writer = pq.ParquetWriter(train_out, t.schema, compression="snappy")
            train_writer.write_table(t)

    if train_writer: train_writer.close()
    if val_writer: val_writer.close()
    print(f"train: {train_pos} pos, {train_neg_kept}/{train_neg_seen} neg kept")
    print(f"val: {val_rows} pairs (unmodified — needed for honest threshold tuning)")
