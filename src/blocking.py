import re
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

csv.field_size_limit(sys.maxsize)

LEGAL_SUFFIXES_PATTERN = r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?"
ADDRESS_NOISE_PATTERN = r"\b(road|rd|street|st|avenue|ave|near|india|us|usa|united|states)\b"

NAME_TOP_K = 75
ADDR_TOP_K = 75
QUERY_CHUNK = 20_000  # s1 rows per matrix-multiply batch, bounds peak memory of the similarity matrix

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

def build_vectorizer(corpus_texts, max_df=0.005):
    vec = CountVectorizer(token_pattern=r"\S+", binary=True, max_df=max_df, min_df=1)
    matrix = vec.fit_transform(corpus_texts)
    return vec, matrix.tocsr()

def top_k_matches(query_matrix, corpus_matrix, corpus_ids, k):
    """query_matrix: sparse (n_query x vocab). corpus_matrix: sparse (n_corpus x vocab).
    Returns list of sets of corpus_ids, one per query row, top-k by shared-token count."""
    sim = query_matrix @ corpus_matrix.T  # sparse (n_query x n_corpus), values = shared token count
    sim = sim.tocsr()
    results = []
    for i in range(sim.shape[0]):
        row = sim.getrow(i)
        if row.nnz == 0:
            results.append(set())
            continue
        if row.nnz > k:
            top_idx_local = np.argpartition(-row.data, k)[:k]
        else:
            top_idx_local = np.arange(row.nnz)
        col_indices = row.indices[top_idx_local]
        results.append({corpus_ids[j] for j in col_indices})
    return results

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
                                 name_k=NAME_TOP_K, addr_k=ADDR_TOP_K, query_chunk=QUERY_CHUNK):
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

            print("  vectorizing corpus (name + address)...", flush=True)
            name2_txt = normalize_name_series(s2_c["business_name"]).values
            name3_txt = normalize_name_series(s3_c["business_name"]).values
            addr2_txt = normalize_address_series(s2_c["business_address"]).values
            addr3_txt = normalize_address_series(s3_c["business_address"]).values

            name2_vec, name2_mat = build_vectorizer(name2_txt)
            name3_vec, name3_mat = build_vectorizer(name3_txt)
            addr2_vec, addr2_mat = build_vectorizer(addr2_txt)
            addr3_vec, addr3_mat = build_vectorizer(addr3_txt)

            s2_ids = s2_c["entity_id"].values
            s3_ids = s3_c["entity_id"].values

            s1_name_txt = normalize_name_series(s1_c["business_name"]).values
            s1_addr_txt = normalize_address_series(s1_c["business_address"]).values
            s1_ids = s1_c["entity_id"].values
            total = len(s1_c)
            print(f"  {total} entities to process, in chunks of {query_chunk}", flush=True)

            for start in range(0, total, query_chunk):
                end = min(start + query_chunk, total)
                chunk_ids = s1_ids[start:end]
                chunk_name = s1_name_txt[start:end]
                chunk_addr = s1_addr_txt[start:end]

                qn2 = name2_vec.transform(chunk_name)
                qn3 = name3_vec.transform(chunk_name)
                qa2 = addr2_vec.transform(chunk_addr)
                qa3 = addr3_vec.transform(chunk_addr)

                r1 = top_k_matches(qn2, name2_mat, s2_ids, name_k)
                r2 = top_k_matches(qn3, name3_mat, s3_ids, name_k)
                r3 = top_k_matches(qa2, addr2_mat, s2_ids, addr_k)
                r4 = top_k_matches(qa3, addr3_mat, s3_ids, addr_k)

                for i, eid in enumerate(chunk_ids):
                    cands = r1[i] | r2[i] | r3[i] | r4[i]
                    writer.writerow([eid, ",".join(sorted(cands))])

                f.flush()
                print(f"    {end}/{total} processed", flush=True)

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