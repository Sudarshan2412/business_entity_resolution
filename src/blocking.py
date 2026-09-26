import re
import csv
import sys
from pathlib import Path
import pandas as pd

csv.field_size_limit(sys.maxsize)

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"
ADDRESS_NOISE_PATTERN = r"\b(road|rd|street|st|avenue|ave|near|india|us|usa|united|states)\b"

NAME_TOP_K = 75
ADDR_TOP_K = 75
CHUNK_SIZE = 20_000  # s1 rows processed per merge batch — tune down if RAM spikes, up if it stays low

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

def token_table(entity_ids, normalized_series, banned=None):
    """entity_id, token — one row per token occurrence."""
    tok_lists = normalized_series.str.split()
    s = pd.Series(tok_lists.values, index=entity_ids).explode().dropna()
    if banned:
        s = s[~s.isin(banned)]
    df = s.reset_index()
    df.columns = ["entity_id", "token"]
    return df

def get_banned_tokens(tok_df, n_entities, max_doc_freq_ratio=0.005, max_doc_freq_abs=1_000_000):
    freq = tok_df.groupby("token")["entity_id"].nunique()
    max_freq = max(1, min(int(n_entities * max_doc_freq_ratio), max_doc_freq_abs))
    return set(freq[freq > max_freq].index)

def top_k_via_merge(query_tok, corpus_tok, k):
    """query_tok, corpus_tok: [entity_id, token] tables. Returns a Series
    keyed by query entity_id, each value a set of top-k candidate ids by
    shared-token count. All heavy lifting is pandas merge/groupby (C-level),
    not a Python loop."""
    merged = query_tok.merge(corpus_tok, on="token", suffixes=("_q", "_c"))
    if merged.empty:
        return pd.Series(dtype=object)
    counts = merged.groupby(["entity_id_q", "entity_id_c"]).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=False)
    top = counts.groupby("entity_id_q").head(k)
    return top.groupby("entity_id_q")["entity_id_c"].apply(set)

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

def generate_candidates_to_file(s1_df, s2_df, s3_df, out_path,
                                 name_k=NAME_TOP_K, addr_k=ADDR_TOP_K, chunk_size=CHUNK_SIZE):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_ids = already_written_ids(out_path)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    print(f"resuming: {len(skip_ids)} rows already written, will skip those", flush=True)

    countries = sorted(s1_df["country"].dropna().unique())
    print(f"countries: {countries}", flush=True)

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
                print("  already done, skipping", flush=True)
                continue

            print("  building corpus token tables...", flush=True)
            name2_raw = token_table(s2_c["entity_id"].values, normalize_name_series(s2_c["business_name"]))
            name3_raw = token_table(s3_c["entity_id"].values, normalize_name_series(s3_c["business_name"]))
            addr2_raw = token_table(s2_c["entity_id"].values, normalize_address_series(s2_c["business_address"]))
            addr3_raw = token_table(s3_c["entity_id"].values, normalize_address_series(s3_c["business_address"]))

            name_banned = get_banned_tokens(name2_raw, len(s2_c)) | get_banned_tokens(name3_raw, len(s3_c))
            addr_banned = get_banned_tokens(addr2_raw, len(s2_c)) | get_banned_tokens(addr3_raw, len(s3_c))

            name2 = name2_raw[~name2_raw["token"].isin(name_banned)]
            name3 = name3_raw[~name3_raw["token"].isin(name_banned)]
            addr2 = addr2_raw[~addr2_raw["token"].isin(addr_banned)]
            addr3 = addr3_raw[~addr3_raw["token"].isin(addr_banned)]

            s1_name_series = normalize_name_series(s1_c["business_name"])
            s1_addr_series = normalize_address_series(s1_c["business_address"])
            total = len(s1_c)
            print(f"  {total} entities to process, in chunks of {chunk_size}", flush=True)

            for start in range(0, total, chunk_size):
                idx_slice = slice(start, start + chunk_size)
                chunk_ids = s1_c["entity_id"].values[idx_slice]
                chunk_name = s1_name_series.values[idx_slice]
                chunk_addr = s1_addr_series.values[idx_slice]

                q_name = pd.DataFrame({"entity_id": chunk_ids, "token": chunk_name})
                q_name = token_table(q_name["entity_id"].values, q_name["token"], banned=name_banned)
                q_addr_df = pd.DataFrame({"entity_id": chunk_ids, "token": chunk_addr})
                q_addr = token_table(q_addr_df["entity_id"].values, q_addr_df["token"], banned=addr_banned)

                r1 = top_k_via_merge(q_name, name2, name_k)
                r2 = top_k_via_merge(q_name, name3, name_k)
                r3 = top_k_via_merge(q_addr, addr2, addr_k)
                r4 = top_k_via_merge(q_addr, addr3, addr_k)

                for eid in chunk_ids:
                    cands = r1.get(eid, set()) | r2.get(eid, set()) | r3.get(eid, set()) | r4.get(eid, set())
                    writer.writerow([eid, ",".join(sorted(cands))])

                f.flush()
                print(f"    {min(start+chunk_size, total)}/{total} processed", flush=True)

    print(f"\ndone — output at {out_path}", flush=True)

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