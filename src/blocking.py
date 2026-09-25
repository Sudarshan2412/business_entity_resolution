import re
import pickle
from pathlib import Path
import pandas as pd
from src.config import OUTPUT

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"

def normalize_series(names: pd.Series) -> pd.Series:
    s = names.astype(str).str.lower()
    s = s.str.replace(LEGAL_SUFFIXES_PATTERN, "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def token_dataframe(df, banned=None):
    """entity_id, token pairs — one row per token occurrence, exploded."""
    normalized = normalize_series(df["business_name"])
    tok_lists = normalized.str.split()
    s = pd.Series(tok_lists.values, index=df["entity_id"].values).explode().dropna()
    tdf = s.reset_index()
    tdf.columns = ["entity_id", "token"]
    if banned:
        tdf = tdf[~tdf["token"].isin(banned)]
    return tdf

def get_banned_tokens(df, max_doc_freq_ratio=0.005):
    normalized = normalize_series(df["business_name"])
    tok_lists = normalized.str.split()
    s = pd.Series(tok_lists.values, index=df["entity_id"].values).explode().dropna()
    freq = s.value_counts()
    max_freq = max(1, int(len(df) * max_doc_freq_ratio))
    return set(freq[freq > max_freq].index)

def generate_candidates(s1_df, s2_df, s3_df, chunk_size=100_000, checkpoint_path=None):
    """Vectorized via merge, processed in chunks of s1 to bound memory,
    with optional checkpointing so a crash doesn't lose everything."""
    print("computing banned tokens...", flush=True)
    banned = get_banned_tokens(s2_df) | get_banned_tokens(s3_df)
    print(f"  {len(banned)} tokens banned total", flush=True)

    print("building token tables for source2/3...", flush=True)
    s2_tok = token_dataframe(s2_df, banned=banned)
    s3_tok = token_dataframe(s3_df, banned=banned)

    all_candidates = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            all_candidates = pickle.load(f)
        print(f"  resumed {len(all_candidates)} entities from checkpoint", flush=True)

    s1_ids_all = s1_df["entity_id"].values
    remaining_mask = ~pd.Series(s1_ids_all).isin(all_candidates.keys())
    s1_remaining = s1_df[remaining_mask.values]
    print(f"  {len(s1_remaining)} source1 entities left to process", flush=True)

    for start in range(0, len(s1_remaining), chunk_size):
        chunk = s1_remaining.iloc[start:start + chunk_size]
        s1_tok = token_dataframe(chunk, banned=banned)

        m2 = s1_tok.merge(s2_tok, on="token", suffixes=("_s1", "_s2"))
        m3 = s1_tok.merge(s3_tok, on="token", suffixes=("_s1", "_s3"))

        cand2 = m2.groupby("entity_id_s1")["entity_id_s2"].apply(lambda x: set(x))
        cand3 = m3.groupby("entity_id_s1")["entity_id_s3"].apply(lambda x: set(x))

        for eid in chunk["entity_id"].values:
            all_candidates[eid] = cand2.get(eid, set()) | cand3.get(eid, set())

        print(f"  processed {start + len(chunk)}/{len(s1_remaining)}", flush=True)
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                pickle.dump(all_candidates, f)

    return all_candidates

def blocking_recall(candidates: dict, gold_df) -> float:
    from src.scoring import parse_id_list
    hits, total = 0, 0
    for eid, matched in zip(gold_df["source1_entity_id"], gold_df["matched_entity_ids"]):
        true_ids = parse_id_list(matched)
        if not true_ids:
            continue
        total += len(true_ids)
        hits += len(true_ids & candidates.get(eid, set()))
    return hits / total if total else float("nan")
