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

def get_banned_tokens(df, max_doc_freq_ratio=0.005, max_doc_freq_abs=2000):
    """Ban a token if it's too common: above max_doc_freq_ratio of the
    source's rows, OR above the absolute cap max_doc_freq_abs — whichever
    is stricter. The abs cap is what stops any single token from blowing
    up candidate-set sizes once the dataset gets huge."""
    normalized = normalize_series(df["business_name"])
    tok_lists = normalized.str.split()
    s = pd.Series(tok_lists.values, index=df["entity_id"].values).explode().dropna()
    freq = s.value_counts()
    max_freq = max(1, min(int(len(df) * max_doc_freq_ratio), max_doc_freq_abs))
    return set(freq[freq > max_freq].index)

def build_inverted_index(df, banned=None):
    normalized = normalize_series(df["business_name"])
    idx = defaultdict(set)
    for eid, name in zip(df["entity_id"].values, normalized.values):
        for tok in name.split():
            if banned and tok in banned:
                continue
            idx[tok].add(eid)
    return idx

def _already_written_ids(out_path):
    written = set()
    if out_path and Path(out_path).exists():
        with open(out_path, "r") as f:
            next(f, None)  # skip header
            for line in f:
                eid = line.split("\t", 1)[0]
                if eid:
                    written.add(eid)
    return written

def generate_candidates_stream(s1_df, s2_df, s3_df, out_path, chunk_size=50_000):
    """Writes source1_entity_id<TAB>candidate_entity_ids straight to
    out_path as it goes. Never holds more than one entity's candidates
    in memory at a time — this is the version to use on the full dataset."""
    print("computing banned tokens...", flush=True)
    banned = get_banned_tokens(s2_df) | get_banned_tokens(s3_df)
    print(f"  {len(banned)} tokens banned total", flush=True)

    print("building inverted index for source2/3...", flush=True)
    idx2 = build_inverted_index(s2_df, banned=banned)
    idx3 = build_inverted_index(s3_df, banned=banned)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = _already_written_ids(out_path)
    write_header = not out_path.exists()

    s1_remaining = s1_df[~s1_df["entity_id"].isin(already)]
    normalized = normalize_series(s1_remaining["business_name"])
    print(f"  {len(s1_remaining)} source1 entities left to process", flush=True)

    with open(out_path, "a") as f:
        if write_header:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
        count = 0
        for eid, name in zip(s1_remaining["entity_id"].values, normalized.values):
            cands = set()
            for tok in name.split():
                cands |= idx2.get(tok, set())
                cands |= idx3.get(tok, set())
            f.write(f"{eid}\t{','.join(sorted(cands))}\n")
            count += 1
            if count % chunk_size == 0:
                f.flush()
                print(f"  processed {count}/{len(s1_remaining)}", flush=True)
    print(f"done — wrote to {out_path}", flush=True)

def generate_candidates_by_country(s1_df, s2_df, s3_df, out_path, chunk_size=50_000):
    """Streams candidates per country into the SAME out_path file.
    Entities with a missing/blank country are matched against everything,
    so nothing gets silently dropped."""
    countries = sorted(s1_df["country"].dropna().unique())
    missing_mask = s1_df["country"].isna()
    print(f"countries found: {countries}", flush=True)
    if missing_mask.sum():
        print(f"WARNING: {missing_mask.sum()} source1 rows have no country "
              f"— matching them against ALL of source2/3 as a fallback.", flush=True)

    for country in countries:
        print(f"\n=== country: {country} ===", flush=True)
        s1_c = s1_df[s1_df["country"] == country]
        s2_c = s2_df[s2_df["country"] == country]
        s3_c = s3_df[s3_df["country"] == country]
        generate_candidates_stream(s1_c, s2_c, s3_c, out_path, chunk_size=chunk_size)

    if missing_mask.sum():
        print("\n=== country: (missing) ===", flush=True)
        generate_candidates_stream(s1_df[missing_mask], s2_df, s3_df, out_path, chunk_size=chunk_size)

def blocking_recall_from_file(path, gold_df) -> float:
    """Scores recall by streaming the candidates file line by line —
    works on the full dataset without loading it all into RAM."""
    from src.scoring import parse_id_list
    gt_lookup = {
        eid: parse_id_list(matched)
        for eid, matched in zip(gold_df["source1_entity_id"], gold_df["matched_entity_ids"])
    }
    hits, total = 0, 0
    with open(path, "r") as f:
        next(f)
        for line in f:
            eid, cand_str = line.rstrip("\n").split("\t")
            true_ids = gt_lookup.get(eid)
            if not true_ids:
                continue
            cand_ids = set(cand_str.split(",")) if cand_str else set()
            total += len(true_ids)
            hits += len(true_ids & cand_ids)
    return hits / total if total else float("nan")

def generate_candidates(s1_df, s2_df, s3_df, chunk_size=50_000, checkpoint_path=None):
    """In-memory version — ONLY use this for small samples (e.g. your
    5,000-row test). Do NOT use this on the full dataset — use
    generate_candidates_by_country instead."""
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

def blocking_recall(candidates: dict, gold_df) -> float:
    """Kept for the small in-memory (5,000-row sample) tests only."""
    from src.scoring import parse_id_list
    hits, total = 0, 0
    for eid, matched in zip(gold_df["source1_entity_id"], gold_df["matched_entity_ids"]):
        true_ids = parse_id_list(matched)
        if not true_ids:
            continue
        total += len(true_ids)
        hits += len(true_ids & candidates.get(eid, set()))
    return hits / total if total else float("nan")