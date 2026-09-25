import re

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

def generate_candidates(s1_df, s2_df, s3_df):
    def index_by_token(df):
        idx = {}
        for _, row in df.iterrows():
            for tok in name_tokens(row["business_name"]):
                idx.setdefault(tok, set()).add(row["entity_id"])
        return idx

    idx2, idx3 = index_by_token(s2_df), index_by_token(s3_df)
    candidates = {}
    for _, row in s1_df.iterrows():
        eid = row["entity_id"]
        toks = name_tokens(row["business_name"])
        cand = set()
        for tok in toks:
            cand |= idx2.get(tok, set())
            cand |= idx3.get(tok, set())
        candidates[eid] = cand
    return candidates

def blocking_recall(candidates: dict, gold_df) -> float:
    from src.scoring import parse_id_list
    hits, total = 0, 0
    for _, row in gold_df.iterrows():
        true_ids = parse_id_list(row["matched_entity_ids"])
        if not true_ids:
            continue
        total += len(true_ids)
        hits += len(true_ids & candidates.get(row["source1_entity_id"], set()))
    return hits / total if total else float("nan")
