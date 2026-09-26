"""
blocking.py — Business Entity Resolution Challenge

Candidate generation ("blocking"): for every Source-1 entity, find a SMALL
set of Source-2/Source-3 records worth comparing in detail, instead of
comparing every record to every other record.

Strategy (simple, explainable, and works across an OPEN country set —
including France, which never appears in training):

  1. Normalize every record's name into tokens (via normalize.py), dropping
     common legal-suffix words ("inc", "ltd", "pvt", ...) since they're too
     frequent to be useful signal.
  2. Build an inverted index: token -> list of entity_ids, SEPARATELY per
     (source, country). Country is used as-is from the data (never a fixed
     {US, India} list), so France or anything else "just works" the same way.
  3. For each Source-1 entity, look up every candidate that shares at least
     one name token AND the same country. Rank candidates by token overlap
     count (+ a light address-token overlap bonus) and keep only the top-K.

This keeps the candidate set small (recall-vs-size is controlled by K and by
MIN_TOKEN_LEN) while remaining generous enough to catch typo'd / reordered
names, since only ONE shared token is required to enter the candidate pool.

Usage:
    from blocking import build_candidates
    candidates = build_candidates(df_s1, df_s2, df_s3, top_k=20)
    # candidates: {source1_entity_id: [(candidate_entity_id, score), ...]}
"""

from collections import defaultdict

from normalize import normalize_name, normalize_address

MIN_TOKEN_LEN = 2  # ignore 1-letter tokens as blocking keys, too noisy


def _prep_records(df):
    """Attach normalized name tokens + address tokens to every row.

    Returns a list of dicts: entity_id, country, name_tokens (set),
    addr_tokens (set).
    """
    records = []
    for row in df.itertuples(index=False):
        name_info = normalize_name(getattr(row, "business_name", ""))
        addr_info = normalize_address(getattr(row, "business_address", ""))
        name_tokens = {t for t in name_info["tokens"] if len(t) >= MIN_TOKEN_LEN}
        addr_tokens = {
            t for t in addr_info["clean_address_lower"].replace(",", " ").split()
            if len(t) >= MIN_TOKEN_LEN
        }
        records.append({
            "entity_id": row.entity_id,
            "country": str(getattr(row, "country", "")).strip(),
            "name_tokens": name_tokens,
            "addr_tokens": addr_tokens,
        })
    return records


def _build_inverted_index(records):
    """token -> country -> list of record indices."""
    index = defaultdict(lambda: defaultdict(list))
    for i, rec in enumerate(records):
        for tok in rec["name_tokens"]:
            index[tok][rec["country"]].append(i)
    return index


def build_candidates(df_s1, df_s2, df_s3, top_k=20):
    """Return {source1_entity_id: [(candidate_entity_id, score), ...]}.

    `score` = (# shared name tokens) + 0.5 * (# shared address tokens),
    used only to rank/trim to top_k — NOT a final match probability.
    """
    s1_records = _prep_records(df_s1)
    s2_records = _prep_records(df_s2)
    s3_records = _prep_records(df_s3)

    s2_index = _build_inverted_index(s2_records)
    s3_index = _build_inverted_index(s3_records)

    results = {}
    for s1 in s1_records:
        scores = defaultdict(float)
        for source_records, source_index in ((s2_records, s2_index), (s3_records, s3_index)):
            seen_idxs = set()
            for tok in s1["name_tokens"]:
                for idx in source_index.get(tok, {}).get(s1["country"], []):
                    seen_idxs.add(idx)
            for idx in seen_idxs:
                cand = source_records[idx]
                name_overlap = len(s1["name_tokens"] & cand["name_tokens"])
                addr_overlap = len(s1["addr_tokens"] & cand["addr_tokens"])
                scores[cand["entity_id"]] = name_overlap + 0.5 * addr_overlap

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results[s1["entity_id"]] = ranked

    return results


def candidates_to_dataframe(candidates):
    """Turn the {s1: [(cand, score), ...]} dict into the candidate_pairs.tsv shape."""
    import pandas as pd
    rows = []
    for s1_id, ranked in candidates.items():
        ids = ",".join(cid for cid, _ in ranked)
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ids})
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])


if __name__ == "__main__":
    import pandas as pd
    import sys

    s1_path, s2_path, s3_path = sys.argv[1:4] if len(sys.argv) > 3 else (
        "sample_source1.tsv", "sample_source2.tsv", "sample_source3.tsv"
    )
    df_s1 = pd.read_csv(s1_path, sep="\t")
    df_s2 = pd.read_csv(s2_path, sep="\t")
    df_s3 = pd.read_csv(s3_path, sep="\t")

    candidates = build_candidates(df_s1, df_s2, df_s3, top_k=20)

    total_candidates = sum(len(v) for v in candidates.values())
    n_zero = sum(1 for v in candidates.values() if len(v) == 0)
    print(f"S1 entities: {len(candidates)}")
    print(f"Avg candidates per S1 entity: {total_candidates / max(len(candidates),1):.2f}")
    print(f"S1 entities with ZERO candidates: {n_zero}")
    print()
    print("Sample rows:")
    out_df = candidates_to_dataframe(candidates)
    print(out_df.head(10).to_string())
