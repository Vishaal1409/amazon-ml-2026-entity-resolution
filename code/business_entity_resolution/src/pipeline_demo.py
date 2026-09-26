"""
pipeline_demo.py — Business Entity Resolution Challenge

Wires normalize.py -> blocking.py -> a PLACEHOLDER matcher -> the two
required output files, so you have a working, correctly-formatted,
end-to-end pipeline TONIGHT. Swap `placeholder_match()` for Abhinav's real
trained classifier once it's ready — everything else (I/O, formatting,
candidate generation) stays the same.

Run:
    python3 pipeline_demo.py \
        --s1 dataset/test/test_source1.tsv \
        --s2 dataset/test/test_source2.tsv \
        --s3 dataset/test/test_source3.tsv \
        --out-dir output

Writes output/matching_results.tsv and output/candidate_pairs.tsv in the
exact format the validator + leaderboard expect.
"""

import argparse
import os

import pandas as pd

from blocking import build_candidates, candidates_to_dataframe

# Placeholder decision rule: a candidate becomes a MATCH only if it shares
# enough tokens with the S1 record to look like strong evidence. This is
# deliberately conservative (favours precision) because F0.5 punishes false
# merges twice as hard as missed matches, and because doing nothing smart is
# safer than doing something wrong. REPLACE with a trained model's predict()
# once the matching model exists — keep this function's signature the same
# so the rest of the pipeline doesn't need to change.
MATCH_SCORE_THRESHOLD = 2.0


def placeholder_match(ranked_candidates, threshold=MATCH_SCORE_THRESHOLD):
    """ranked_candidates: [(candidate_id, score), ...] from blocking.

    Returns the list of candidate_ids to treat as final matches.
    """
    return [cid for cid, score in ranked_candidates if score >= threshold]


def run(s1_path, s2_path, s3_path, out_dir, top_k=20):
    df_s1 = pd.read_csv(s1_path, sep="\t")
    df_s2 = pd.read_csv(s2_path, sep="\t")
    df_s3 = pd.read_csv(s3_path, sep="\t")

    print(f"Loaded: S1={len(df_s1)} S2={len(df_s2)} S3={len(df_s3)} rows")

    candidates = build_candidates(df_s1, df_s2, df_s3, top_k=top_k)

    match_rows = []
    for s1_id, ranked in candidates.items():
        matched_ids = placeholder_match(ranked)
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids),
        })
    matching_df = pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"])
    candidate_df = candidates_to_dataframe(candidates)

    os.makedirs(out_dir, exist_ok=True)
    matching_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")

    matching_df.to_csv(matching_path, sep="\t", index=False)
    candidate_df.to_csv(candidate_path, sep="\t", index=False)

    n_with_match = (matching_df["matched_entity_ids"] != "").sum()
    print(f"Wrote {matching_path} ({len(matching_df)} rows, {n_with_match} with >=1 match)")
    print(f"Wrote {candidate_path} ({len(candidate_df)} rows)")
    return matching_path, candidate_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    run(args.s1, args.s2, args.s3, args.out_dir, top_k=args.top_k)
