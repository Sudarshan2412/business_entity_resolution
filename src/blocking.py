import re
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"

def normalize_series(names: pd.Series) -> pd.Series:
    """Vectorized normalization — pandas applies these as bulk operations,
    much faster than calling re.sub() per row in a Python loop."""
    s = names.astype(str).str.lower()
    s = s.str.replace(LEGAL_SUFFIXES_PATTERN, "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def normalize_name(name: str) -> str:
    """Single-string version, kept for one-off use elsewhere."""
    name = str(name).lower()
    name = re.sub(LEGAL_SUFFIXES_PATTERN, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def build_token_index(df, cache_path=None, max_doc_freq_ratio=0.005):
    if cache_path and Path(cache_path).exists():
        print(f"  loading cached index from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    ids = df["entity_id"].values
    normalized = normalize_series(df["business_name"])
    token_lists = normalized.str.split()

    doc_freq = defaultdict(int)
    for toks in token_lists:
        for tok in toks:
            doc_freq[tok] += 1

    max_freq = max(1, int(len(df) * max_doc_freq_ratio))
    banned = {tok for tok, freq in doc_freq.items() if freq > max_freq}
    print(f"  dropping {len(banned)} overly common tokens (freq > {max_freq})")

    idx = defaultdict(set)
    for eid, toks in zip(ids, token_lists):
        for tok in toks:
            if tok not in banned:
                idx[tok].add(eid)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(dict(idx), f)
        print(f"  cached index to {cache_path}")

    return idx

def generate_candidates(s1_df, s2_df, s3_df, cache_dir="/content/business_entity_resolution/output/cache"):
    print("indexing source 2...")
    idx2 = build_token_index(s2_df, cache_path=f"{cache_dir}/idx2.pkl")
    print("indexing source 3...")
    idx3 = build_token_index(s3_df, cache_path=f"{cache_dir}/idx3.pkl")

    s1_ids = s1_df["entity_id"].values
    s1_normalized = normalize_series(s1_df["business_name"])
    s1_token_lists = s1_normalized.str.split()

    candidates = {}
    for i, (eid, toks) in enumerate(zip(s1_ids, s1_token_lists)):
        cand = set()
        for tok in toks:
            cand |= idx2.get(tok, set())
            cand |= idx3.get(tok, set())
        candidates[eid] = cand
        if i % 1000 == 0:
            print(f"  {i}/{len(s1_ids)} source1 entities processed")
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
