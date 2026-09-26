import re
import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

csv.field_size_limit(sys.maxsize)

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"
ADDRESS_NOISE_PATTERN = r"\b(road|rd|street|st|avenue|ave|near|india|us|usa|united|states)\b"

NAME_TOP_K = 75
ADDR_TOP_K = 75

def normalize_name_series(names: pd.Series) -> pd.Series:
    s = names.astype(str).str.lower()
    s = s.str.replace(LEGAL_SUFFIXES_PATTERN, "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def normalize_address_series(addrs: pd.Series) -> pd.Series:
    s = addrs.astype(str).str.lower()
    s = s.str.replace(ADDRESS_NOISE_PATTERN, "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def get_banned_tokens(series: pd.Series, entity_ids, max_doc_freq_ratio=0.005, max_doc_freq_abs=1_000_000):
    tok_lists = series.str.split()
    s = pd.Series(tok_lists.values, index=entity_ids).explode().dropna()
    freq = s.value_counts()
    n = len(entity_ids)
    max_freq = max(1, min(int(n * max_doc_freq_ratio), max_doc_freq_abs))
    return set(freq[freq > max_freq].index)

def build_inverted_index(series: pd.Series, entity_ids, banned=None):
    idx = defaultdict(set)
    for eid, text in zip(entity_ids, series.values):
        for tok in text.split():
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

def top_k_scores(tokens, idx, k):
    scores = Counter()
    for tok in tokens:
        for cid in idx.get(tok, ()):
            scores[cid] += 1
    if not scores:
        return set()
    return {cid for cid, _ in scores.most_common(k)}

def generate_candidates_to_file(s1_df, s2_df, s3_df, out_path, flush_every=2000,
                                 name_k=NAME_TOP_K, addr_k=ADDR_TOP_K):
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

            print("  normalizing + banning tokens (name + address)...", flush=True)
            name2 = normalize_name_series(s2_c["business_name"])
            name3 = normalize_name_series(s3_c["business_name"])
            addr2 = normalize_address_series(s2_c["business_address"])
            addr3 = normalize_address_series(s3_c["business_address"])

            name_banned = get_banned_tokens(name2, s2_c["entity_id"].values) | \
                          get_banned_tokens(name3, s3_c["entity_id"].values)
            addr_banned = get_banned_tokens(addr2, s2_c["entity_id"].values) | \
                          get_banned_tokens(addr3, s3_c["entity_id"].values)

            print("  building inverted indexes...", flush=True)
            name_idx2 = build_inverted_index(name2, s2_c["entity_id"].values, name_banned)
            name_idx3 = build_inverted_index(name3, s3_c["entity_id"].values, name_banned)
            addr_idx2 = build_inverted_index(addr2, s2_c["entity_id"].values, addr_banned)
            addr_idx3 = build_inverted_index(addr3, s3_c["entity_id"].values, addr_banned)

            s1_name = normalize_name_series(s1_c["business_name"])
            s1_addr = normalize_address_series(s1_c["business_address"])
            total = len(s1_c)
            print(f"  {total} entities to process", flush=True)

            count = 0
            for eid, name, addr in zip(s1_c["entity_id"].values, s1_name.values, s1_addr.values):
                name_cands_2 = top_k_scores(name.split(), name_idx2, name_k)
                name_cands_3 = top_k_scores(name.split(), name_idx3, name_k)
                addr_cands_2 = top_k_scores(addr.split(), addr_idx2, addr_k)
                addr_cands_3 = top_k_scores(addr.split(), addr_idx3, addr_k)

                cands = name_cands_2 | name_cands_3 | addr_cands_2 | addr_cands_3
                writer.writerow([eid, ",".join(sorted(cands))])
                total_candidates_written += len(cands)
                entities_written += 1
                count += 1
                if count % flush_every == 0:
                    f.flush()
                    print(f"    {count}/{total} processed", flush=True)

            f.flush()
            del name_idx2, name_idx3, addr_idx2, addr_idx3

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