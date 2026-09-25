import re
from collections import defaultdict

LEGAL_SUFFIXES = re.compile(
    r"\b(corp|corporation|pvt|private|ltd|limited|llc|inc|incorporated|co)\b\.?",
    flags=re.IGNORECASE,
)

def normalize_name(name: str) -> str:
    name = str(name).lower()
    name = LEGAL_SUFFIXES.sub("", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def name_tokens(name: str) -> set:
    return set(normalize_name(name).split())

def build_token_index(df, max_doc_freq_ratio=0.005):
    """Token -> set of entity_ids, dropping tokens that appear in too many records
    (they're useless for blocking and cause candidate-set explosions)."""
    ids = df["entity_id"].values
    names = df["business_name"].values

    token_lists = [name_tokens(n) for n in names]  # still O(n) but no iterrows overhead

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
    return idx

def generate_candidates(s1_df, s2_df, s3_df, max_doc_freq_ratio=0.005):
    print("indexing source 2...")
    idx2 = build_token_index(s2_df, max_doc_freq_ratio)
    print("indexing source 3...")
    idx3 = build_token_index(s3_df, max_doc_freq_ratio)

    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["business_name"].values

    candidates = {}
    for i, (eid, name) in enumerate(zip(s1_ids, s1_names)):
        toks = name_tokens(name)
        cand = set()
        for tok in toks:
            cand |= idx2.get(tok, set())
            cand |= idx3.get(tok, set())
        candidates[eid] = cand
        if i % 200000 == 0:
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
