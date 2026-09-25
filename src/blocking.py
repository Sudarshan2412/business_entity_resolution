import re
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd
from src.config import OUTPUT

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"

def normalize_series(names: pd.Series) -> pd.Series:
    s = names.astype(str).str.lower()
    s = s.str.replace(LEGAL_SUFFIXES_PATTERN, "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def get_banned_tokens(df, max_doc_freq_ratio=0.001, max_doc_freq_abs=500):
    """Ban a token if it's too common: either above max_doc_freq_ratio of
    the source's rows, OR above the absolute cap max_doc_freq_abs —
    whichever limit is stricter."""
    normalized = normalize_series(df["business_name"])
    tok_lists = normalized.str.split()
    s = pd.Series(tok_lists.values, index=df["entity_id"].values).explode().dropna()
    freq = s.value_counts()
    max_freq = max(1, min(int(len(df) * max_doc_freq_ratio), max_doc_freq_abs))
    return set(freq[freq > max_freq].index)

def build_inverted_index(df, banned=None):
    """token -> set(entity_id). No pandas merge, no cross-join."""
    normalized = normalize_series(df["business_name"])
    idx = defaultdict(set)
    for eid, name in zip(df["entity_id"].values, normalized.values):
        for tok in name.split():
            if banned and tok in banned:
                continue
            idx[tok].add(eid)
    return idx

def generate_candidates(s1_df, s2_df, s3_df, chunk_size=50_000, checkpoint_path=None):
    """Inverted-index based candidate generation. Memory stays flat no
    matter how much of s1 you feed in."""
    print("computing banned tokens...", flush=True)
    banned = get_banned_tokens(s2_df) | get_banned_tokens(s3_df)
    print(f"  {len(banned)} tokens banned total", flush=True)

    print("building inverted index for source2/3...", flush=True)
    idx2 = build_inverted_index(s2_df, banned=banned)
    idx3 = build_inverted_index(s3_df, banned=banned)

    all_candidates = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            all_candidates = pickle.load(f)
        print(f"  resumed {len(all_candidates)} entities from checkpoint", flush=True)

    s1_remaining = s1_df[~s1_df["entity_id"].isin(all_candidates.keys())]
    normalized = normalize_series(s1_remaining["business_name"])
    print(f"  {len(s1_remaining)} source1 entities left to process", flush=True)

    count = 0
    for eid, name in zip(s1_remaining["entity_id"].values, normalized.values):
        cands = set()
        for tok in name.split():
            cands |= idx2.get(tok, set())
            cands |= idx3.get(tok, set())
        all_candidates[eid] = cands
        count += 1
        if count % chunk_size == 0:
            print(f"  processed {count}/{len(s1_remaining)}", flush=True)
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(all_candidates, f)

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(all_candidates, f)

    return all_candidates

def generate_candidates_by_country(s1_df, s2_df, s3_df, chunk_size=50_000, checkpoint_dir=None):
    """Runs generate_candidates separately per country, then merges the
    results. A record from one country can never match a record from
    another, so this shrinks the comparison universe for free."""
    all_candidates = {}
    countries = sorted(s1_df["country"].dropna().unique())
    print(f"countries found: {countries}", flush=True)

    for country in countries:
        print(f"\n=== country: {country} ===", flush=True)
        s1_c = s1_df[s1_df["country"] == country]
        s2_c = s2_df[s2_df["country"] == country]
        s3_c = s3_df[s3_df["country"] == country]

        checkpoint_path = None
        if checkpoint_dir:
            safe_name = country.replace(" ", "_")
            checkpoint_path = Path(checkpoint_dir) / f"candidates_{safe_name}.pkl"

        country_candidates = generate_candidates(
            s1_c, s2_c, s3_c,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
        )
        all_candidates.update(country_candidates)

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