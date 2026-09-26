"""Inference on the test split -> output/matching_results.tsv + output/candidate_pairs.tsv"""
import json
import sys
import time

import lightgbm as lgb
import polars as pl

import pipeline as P
from blocking import load
from common import WORK, OUT, N_JOBS


def write_lists(pairs, s1_ids, id_col, path):
    agg = pairs.group_by("entity_id").agg(pl.col("cand_id").unique().sort().str.join(",").alias(id_col))
    out = (s1_ids.join(agg, on="entity_id", how="left").fill_null("")
                 .rename({"entity_id": "source1_entity_id"}).select("source1_entity_id", id_col))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main(split="test"):
    t0 = time.time()
    cfg = json.load(open(WORK / "model_config.json"))
    m2 = lgb.Booster(model_file=str(WORK / "stage2.txt"))
    m3 = lgb.Booster(model_file=str(WORK / "stage3.txt"))
    sc = pl.read_parquet(WORK / f"{split}_scored.parquet")
    surv = P.stage2_filter(split, sc, m2, cfg["thr2"], cfg.get("s2_topn", P.S2_TOPN))
    del sc
    feats = P.stage3_frame(split, surv)
    feats = feats.with_columns(pl.Series("p3", m3.predict(feats.select(P.stage3_cols()).to_numpy(), num_threads=N_JOBS)))
    pred = P.decide(feats, "p3", cfg["thr3"])

    s1 = load(split, 1).select("eid", "entity_id")
    cand = pl.concat([load(split, 2), load(split, 3)]).select(pl.col("eid").alias("cid"),
                                                              pl.col("entity_id").alias("cand_id"))
    name = lambda d: d.select("eid", "cid").join(s1, on="eid").join(cand, on="cid")
    OUT.mkdir(parents=True, exist_ok=True)
    c = write_lists(name(surv), s1.select("entity_id"), "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    m = write_lists(name(pred), s1.select("entity_id"), "matched_entity_ids", OUT / "matching_results.tsv")
    print(f"S1 {s1.height:,}; candidates {surv.height:,} ({surv.height/s1.height:.2f}/S1); "
          f"matches {pred.height:,} ({pred.height/s1.height:.2f}/S1); "
          f"empty rows {(m['matched_entity_ids']=='').sum():,} ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
