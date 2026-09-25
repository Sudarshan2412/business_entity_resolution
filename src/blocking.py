import re
import pickle
from pathlib import Path
import pandas as pd

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"

def normalize_series(names: pd.Series) -> pd.Series:
    s = names.astype(str).str.lower()
    s = s.str.replace(LEGAL_SUFFIXES_PATTERN, "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def build_token_index(df, cache_path=None, max_doc_freq_ratio=0.005):
    if cache_path and Path(cache_path).exists():
        print(f"  loading cached index from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  normalizing {len(df)} names...", flush=True)
    normalized = normalize_series(df["business_name"])
    tok_lists = normalized.str.split()

    print("  exploding tokens (this is the heavy step)...", flush=True)
    exploded = pd.Series(tok_lists.values, index=df["entity_id"].values).explode()
    exploded = exploded.dropna()
    print(f"  {len(exploded)} total token occurrences", flush=True)

    print("  computing document frequency...", flush=True)
    freq = exploded.value_counts()
    max_freq = max(1, int(len(df) * max_doc_freq_ratio))
    banned = set(freq[freq > max_freq].index)
    print(f"  dropping {len(banned)} overly common tokens (freq > {max_freq})", flush=True)

    exploded = exploded[~exploded.isin(banned)]

    print("  grouping into index...", flush=True)
    idx = exploded.groupby(exploded.values).apply(lambda s: set(s.index)).to_dict()

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(idx, f)
        print(f"  cached index to {cache_path}", flush=True)

    return idx

def generate_candidates(s1_df, s2_df, s3_df, cache_dir="/content/business_entity_resolution/output/cache"):
    print("indexing source 2...", flush=True)
    idx2 = build_token_index(s2_df, cache_path=f"{cache_dir}/idx2.pkl")
    print("indexing source 3...", flush=True)
    idx3 = build_token_index(s3_df, cache_path=f"{cache_dir}/idx3.pkl")

    print("building candidates for source1...", flush=True)
    s1_ids = s1_df["entity_id"].values
    s1_tok_lists = normalize_series(s1_df["business_name"]).str.split()

    candidates = {}
    for i, (eid, toks) in enumerate(zip(s1_ids, s1_tok_lists)):
        cand = set()
        for tok in toks:
            cand |= idx2.get(tok, set())
            cand |= idx3.get(tok, set())
        candidates[eid] = cand
        if i % 500 == 0:
            print(f"  {i}/{len(s1_ids)} source1 entities processed", flush=True)
    return candidates

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
