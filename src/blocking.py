import re
import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

csv.field_size_limit(sys.maxsize)

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"
MAX_CANDIDATES_PER_ENTITY = 100  # tune down further once you check recall holds

def normalize_series(names: pd.Series) -> pd.Series:
    s = names.astype(str).str.lower()
    s = s.str.replace(LEGAL_SUFFIXES_PATTERN, "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def get_banned_tokens(df, max_doc_freq_ratio=0.005, max_doc_freq_abs=1_000_000):
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

def already_written_ids(out_path):
    if not Path(out_path).exists():
        return set()
    done = set()
    with open(out_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                done.add(row[0])
    return done

def top_k_candidates(name_tokens, idx2, idx3, k=MAX_CANDIDATES_PER_ENTITY):
    """Score each candidate by how many query tokens it shares (token-overlap
    count), keep the top-k. This is the ranking that actually matters — a
    candidate sharing 4 tokens with the query is far more likely to be a true
    match than one sharing 1, so this keeps the candidates most worth scoring
    later, not an arbitrary subset."""
    scores = Counter()
    for tok in name_tokens:
        for cid in idx2.get(tok, ()):
            scores[cid] += 1
        for cid in idx3.get(tok, ()):
            scores[cid] += 1
    if not scores:
        return set()
    return {cid for cid, _ in scores.most_common(k)}

def generate_candidates_to_file(s1_df, s2_df, s3_df, out_path, flush_every=2000,
                                 max_candidates=MAX_CANDIDATES_PER_ENTITY):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_ids = already_written_ids(out_path)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    print(f"resuming: {len(skip_ids)} rows already written, will skip those", flush=True)

    countries = sorted(s1_df["country"].dropna().unique())
    print(f"countries: {countries}", flush=True)

    total_candidates_written = 0
    entities_written = 0

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for country in countries:
            print(f"\n=== country: {country} ===", flush=True)
            s1_c = s1_df[s1_df["country"] == country]
            s2_c = s2_df[s2_df["country"] == country]
            s3_c = s3_df[s3_df["country"] == country]

            s1_c = s1_c[~s1_c["entity_id"].isin(skip_ids)]
            if len(s1_c) == 0:
                print("  already fully done, skipping", flush=True)
                continue

            print("  computing banned tokens...", flush=True)
            banned = get_banned_tokens(s2_c) | get_banned_tokens(s3_c)

            print("  building inverted index...", flush=True)
            idx2 = build_inverted_index(s2_c, banned=banned)
            idx3 = build_inverted_index(s3_c, banned=banned)

            normalized = normalize_series(s1_c["business_name"])
            total = len(s1_c)
            print(f"  {total} entities to process", flush=True)

            count = 0
            for eid, name in zip(s1_c["entity_id"].values, normalized.values):
                toks = name.split()
                cands = top_k_candidates(toks, idx2, idx3, k=max_candidates)
                writer.writerow([eid, ",".join(sorted(cands))])
                total_candidates_written += len(cands)
                entities_written += 1
                count += 1
                if count % flush_every == 0:
                    f.flush()
                    print(f"    {count}/{total} processed", flush=True)

            f.flush()
            del idx2, idx3

    avg = total_candidates_written / entities_written if entities_written else 0
    print(f"\ndone — output at {out_path}", flush=True)
    print(f"avg candidates/entity this run: {avg:.1f}", flush=True)

def blocking_recall_from_file(candidates_path, gold_df):
    from src.scoring import parse_id_list

    gold_map = {}
    for eid, matched in zip(gold_df["source1_entity_id"], gold_df["matched_entity_ids"]):
        true_ids = parse_id_list(matched)
        if true_ids:
            gold_map[eid] = true_ids

    hits, total = 0, 0
    with open(candidates_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            eid, cand_str = row[0], row[1] if len(row) > 1 else ""
            true_ids = gold_map.get(eid)
            if not true_ids:
                continue
            cand_ids = set(cand_str.split(",")) if cand_str else set()
            total += len(true_ids)
            hits += len(true_ids & cand_ids)

    return hits / total if total else float("nan")