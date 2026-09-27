import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer

try:
    from rapidfuzz.distance import Levenshtein
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

from src.blocking import normalize_name_series as normalize_series

POSTAL_RE = re.compile(r"\b\d{4,7}\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

FEATURE_COLUMNS = [
    "name_char3_jaccard", "name_token_jaccard", "name_levenshtein",
    "name_len_diff", "name_token_count_diff",
    "addr_token_jaccard", "addr_levenshtein", "addr_len_diff",
    "addr_token_count_diff", "postal_match", "house_no_match", "country_match",
]


def normalize_address(addr: pd.Series) -> pd.Series:
    s = addr.astype(str).str.lower()
    s = s.str.replace(NON_ALNUM_RE, " ", regex=True)
    s = s.str.replace(MULTI_SPACE_RE, " ", regex=True).str.strip()
    return s


def extract_postal_code(addr_norm: str) -> Optional[str]:
    m = POSTAL_RE.findall(addr_norm)
    return m[-1] if m else None


def extract_house_number(addr_norm: str) -> Optional[str]:
    for tok in addr_norm.split():
        if tok.isdigit():
            return tok
    return None


def scan_referenced_ids(candidate_pairs_path, chunksize=200_000):
    s1_ids, cand_ids = set(), set()
    for chunk in pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, chunksize=chunksize):
        chunk["candidate_entity_ids"] = chunk["candidate_entity_ids"].fillna("")
        s1_ids.update(chunk["source1_entity_id"].tolist())
        for cell in chunk["candidate_entity_ids"]:
            if cell:
                cand_ids.update(cell.split(","))
    return s1_ids, cand_ids


def build_entity_lookup(df: pd.DataFrame, keep_ids: set) -> dict:
    df = df[df["entity_id"].isin(keep_ids)]
    name_norm = normalize_series(df["business_name"].fillna(""))
    addr_norm = normalize_address(df["business_address"].fillna(""))
    lookup = {}
    for eid, name, addr, country in zip(
        df["entity_id"].values, name_norm.values, addr_norm.values,
        df["country"].astype(str).values,
    ):
        lookup[eid] = {
            "name": name, "addr": addr,
            "postal": extract_postal_code(addr),
            "house_no": extract_house_number(addr),
            "country": country.strip().lower(),
        }
    return lookup


def explode_candidates(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["candidate_entity_ids"] = chunk["candidate_entity_ids"].fillna("")
    chunk = chunk[chunk["candidate_entity_ids"] != ""]
    if chunk.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    chunk["candidate_entity_id"] = chunk["candidate_entity_ids"].str.split(",")
    chunk = chunk.explode("candidate_entity_id")
    return chunk[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)


def fit_global_vectorizers(s1_lookup, s23_lookup):
    all_names = [r["name"] for r in s1_lookup.values()] + [r["name"] for r in s23_lookup.values()]
    all_addrs = [r["addr"] for r in s1_lookup.values()] + [r["addr"] for r in s23_lookup.values()]

    name_char_vec = CountVectorizer(analyzer="char", ngram_range=(3, 3), binary=True, min_df=1)
    name_char_vec.fit(all_names)
    name_word_vec = CountVectorizer(analyzer="word", binary=True, min_df=1)
    name_word_vec.fit(all_names)
    addr_word_vec = CountVectorizer(analyzer="word", binary=True, min_df=1)
    addr_word_vec.fit(all_addrs)

    return name_char_vec, name_word_vec, addr_word_vec


def _jaccard_from_vectors(A, B):
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    sizes_a = np.asarray(A.sum(axis=1)).ravel()
    sizes_b = np.asarray(B.sum(axis=1)).ravel()
    union = sizes_a + sizes_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    return jac.astype(np.float32)


def compute_features_for_pairs(pairs, s1_lookup, s23_lookup, name_char_vec, name_word_vec, addr_word_vec):
    ids1 = pairs["source1_entity_id"].values
    ids2 = pairs["candidate_entity_id"].values

    valid = np.array([e1 in s1_lookup and e2 in s23_lookup for e1, e2 in zip(ids1, ids2)])
    if not valid.any():
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS)

    ids1, ids2 = ids1[valid], ids2[valid]
    r1 = [s1_lookup[e] for e in ids1]
    r2 = [s23_lookup[e] for e in ids2]

    names1 = [r["name"] for r in r1]
    names2 = [r["name"] for r in r2]
    addrs1 = [r["addr"] for r in r1]
    addrs2 = [r["addr"] for r in r2]

    A_char = name_char_vec.transform(names1); B_char = name_char_vec.transform(names2)
    A_word = name_word_vec.transform(names1); B_word = name_word_vec.transform(names2)
    A_addr = addr_word_vec.transform(addrs1); B_addr = addr_word_vec.transform(addrs2)

    name_char3_jac = _jaccard_from_vectors(A_char, B_char)
    name_token_jac = _jaccard_from_vectors(A_word, B_word)
    addr_token_jac = _jaccard_from_vectors(A_addr, B_addr)

    if _HAS_RAPIDFUZZ:
        name_lev = np.array([Levenshtein.normalized_similarity(a, b) if a and b else 0.0
                              for a, b in zip(names1, names2)], dtype=np.float32)
        addr_lev = np.array([Levenshtein.normalized_similarity(a, b) if a and b else 0.0
                              for a, b in zip(addrs1, addrs2)], dtype=np.float32)
    else:
        import difflib
        name_lev = np.array([difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0
                              for a, b in zip(names1, names2)], dtype=np.float32)
        addr_lev = np.array([difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0
                              for a, b in zip(addrs1, addrs2)], dtype=np.float32)

    name_len_diff = np.abs(np.array([len(a) for a in names1]) - np.array([len(b) for b in names2])).astype(np.float32)
    name_tok_diff = np.abs(np.array([len(a.split()) for a in names1]) - np.array([len(b.split()) for b in names2])).astype(np.float32)
    addr_len_diff = np.abs(np.array([len(a) for a in addrs1]) - np.array([len(b) for b in addrs2])).astype(np.float32)
    addr_tok_diff = np.abs(np.array([len(a.split()) for a in addrs1]) - np.array([len(b.split()) for b in addrs2])).astype(np.float32)

    postal_match = np.array([
        1.0 if a["postal"] and b["postal"] and a["postal"] == b["postal"]
        else (0.0 if a["postal"] and b["postal"] else -1.0)
        for a, b in zip(r1, r2)
    ], dtype=np.float32)
    house_match = np.array([
        1.0 if a["house_no"] and b["house_no"] and a["house_no"] == b["house_no"]
        else (0.0 if a["house_no"] and b["house_no"] else -1.0)
        for a, b in zip(r1, r2)
    ], dtype=np.float32)
    country_match = np.array([1.0 if a["country"] == b["country"] else 0.0 for a, b in zip(r1, r2)], dtype=np.float32)

    return pd.DataFrame({
        "source1_entity_id": ids1, "candidate_entity_id": ids2,
        "name_char3_jaccard": name_char3_jac, "name_token_jaccard": name_token_jac,
        "name_levenshtein": name_lev, "name_len_diff": name_len_diff, "name_token_count_diff": name_tok_diff,
        "addr_token_jaccard": addr_token_jac, "addr_levenshtein": addr_lev,
        "addr_len_diff": addr_len_diff, "addr_token_count_diff": addr_tok_diff,
        "postal_match": postal_match, "house_no_match": house_match, "country_match": country_match,
    })


def run_feature_pipeline(candidate_pairs_path, s1_df, s2_df, s3_df, out_path,
                          row_chunksize: int = 20_000, max_pairs_per_batch: int = 300_000):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print("scanning candidate_pairs.tsv for referenced entity ids...", flush=True)
    s1_ids, cand_ids = scan_referenced_ids(candidate_pairs_path)
    print(f"  {len(s1_ids)} source1 ids, {len(cand_ids)} unique candidate ids referenced", flush=True)

    print("building entity lookups...", flush=True)
    s1_lookup = build_entity_lookup(s1_df, s1_ids)
    s23_lookup = build_entity_lookup(s2_df, cand_ids)
    s23_lookup.update(build_entity_lookup(s3_df, cand_ids))
    print(f"  s1_lookup: {len(s1_lookup)}, s23_lookup: {len(s23_lookup)}", flush=True)

    print("fitting vectorizers ONCE over full referenced corpus...", flush=True)
    t_fit = time.time()
    name_char_vec, name_word_vec, addr_word_vec = fit_global_vectorizers(s1_lookup, s23_lookup)
    print(f"  vectorizer fit done in {time.time()-t_fit:.1f}s", flush=True)

    writer = None
    total_pairs = 0
    t0 = time.time()
    pending, pending_count = [], 0

    def flush_pending():
        nonlocal writer, total_pairs, pending
        if not pending:
            return
        pairs = pd.concat(pending, ignore_index=True)
        pending = []
        feats = compute_features_for_pairs(pairs, s1_lookup, s23_lookup, name_char_vec, name_word_vec, addr_word_vec)
        if feats.empty:
            return
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
        writer.write_table(table)
        total_pairs += len(feats)
        elapsed = time.time() - t0
        print(f"  {total_pairs} pairs written, {elapsed:.1f}s elapsed ({total_pairs/elapsed:.0f} pairs/sec)", flush=True)

    for chunk in pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, chunksize=row_chunksize):
        exploded = explode_candidates(chunk)
        if exploded.empty:
            continue
        pending.append(exploded)
        pending_count += len(exploded)
        if pending_count >= max_pairs_per_batch:
            flush_pending()
            pending_count = 0

    flush_pending()
    if writer is not None:
        writer.close()
    print(f"done — {total_pairs} pairs written to {out_path}", flush=True)
