"""Part B: self-training for country labels that never appear in training (France).

1. Pseudo-label the pass-2 pair universe of unseen-country test S1 entities:
   positive = one-to-one winner with p_final >= POS, negative = p_final <= NEG;
   entities with any pair in between are dropped entirely.
2. Retrain the pass-2 model on fold-C1 training rows + pseudo-labelled rows (weight W_PSEUDO).
3. Accept the adapted model only if hold-out (fold-C2, US/India) F0.5 does not drop by more
   than MAX_DROP; otherwise keep the original model.
4. Re-score the unseen-country pairs with the adapted model and rewrite their rows in the
   output files (candidate set unchanged).
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import collective as C
import pipeline as P
import train as T
from blocking import load
from common import N_JOBS, OUT, WORK

POS, NEG, W_PSEUDO, MAX_DROP = 0.97, 0.03, 0.5, 0.0005


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def unseen_eids(split="test"):
    seen = set(load("train", 1)["country"].unique().to_list())
    s1 = load(split, 1).select("eid", "country")
    return s1.filter(~pl.col("country").is_in(list(seen)))["eid"]


def main():
    cfg = json.load(open(WORK / "model_config.json"))
    cols, thr = cfg["pass2_cols"], cfg["pass2_thr"]
    eids = unseen_eids()
    log(f"unseen-country S1: {len(eids):,}")

    # ---- pseudo-labels on the unseen-country universe ----
    pass1 = pl.read_parquet(WORK / "test_pass1.parquet").filter(pl.col("eid").is_in(eids.implode()))
    sc = pl.scan_parquet(WORK / "test_scored.parquet").filter(pl.col("eid").is_in(eids.implode())).collect()
    d = C.pass2_frame("test", pass1, sc)
    del sc
    m0 = lgb.Booster(model_file=str(WORK / "pass2.txt"))
    d = d.with_columns(pl.Series("p_final", m0.predict(d.select(cols).to_numpy(), num_threads=N_JOBS)))
    d = d.filter((pl.col("is_new") == 0) | (
        (pl.col("sib_bscore").rank("ordinal", descending=True).over("eid") <= C.SIB_TOPN_PRED)
        & (pl.col("sib_bscore") >= C.SIB_MIN_PRED)))
    win = pl.col("p_final").rank("ordinal", descending=True).over("cid") == 1
    unsure = ((pl.col("p_final") > NEG) & (pl.col("p_final") < POS)).any().over("eid")
    pl_rows = (d.filter(~unsure)
                .with_columns(pl.when((pl.col("p_final") >= POS) & win).then(1)
                                .when(pl.col("p_final") <= NEG).then(0).otherwise(None).alias("y"))
                .drop_nulls("y"))
    log(f"pseudo-labelled rows {pl_rows.height:,} (pos {pl_rows['y'].sum():,}) "
        f"from {pl_rows['eid'].n_unique():,} of {len(eids):,} S1")

    # ---- retrain pass 2 with pseudo-labels ----
    tr_all = pl.read_parquet(WORK / "train_pass2_C.parquet")
    c1 = pl.col("eid").hash(seed=C.SEED) % 5 < 3
    tr, va = tr_all.filter(c1), tr_all.filter(~c1)
    inner = tr["eid"].hash(seed=3) % 10 == 0
    X = np.vstack([tr.filter(~inner).select(cols).to_numpy(), pl_rows.select(cols).to_numpy()])
    y = np.concatenate([tr.filter(~inner)["y"].to_numpy(), pl_rows["y"].to_numpy()])
    w = np.concatenate([np.ones(tr.filter(~inner).height), np.full(pl_rows.height, W_PSEUDO)])
    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  verbose=-1, num_threads=N_JOBS, seed=C.SEED)
    dtr = lgb.Dataset(X, y, weight=w)
    dva = lgb.Dataset(tr.filter(inner).select(cols).to_numpy(), tr.filter(inner)["y"].to_numpy(), reference=dtr)
    m1 = lgb.train(params, dtr, 3000, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])

    # ---- acceptance check on hold-out (seen countries) ----
    s1ids, pos, nt = T.truth()
    eC = s1ids.with_columns(T.fold_of("eid").alias("f")).filter(pl.col("f") >= 7)["eid"]
    ev = eC.filter(eC.hash(seed=C.SEED) % 5 >= 3)
    ntv = nt.filter(pl.col("eid").is_in(ev.implode()))
    va = va.filter((pl.col("is_new") == 0) | (
        (pl.col("sib_bscore").rank("ordinal", descending=True).over("eid") <= C.SIB_TOPN_PRED)
        & (pl.col("sib_bscore") >= C.SIB_MIN_PRED)))
    Xv = va.select(cols).to_numpy()
    f0 = P.f05_macro(P.decide(va.with_columns(pl.Series("p", m0.predict(Xv, num_threads=N_JOBS))), "p", thr), ntv, ev)
    f1 = P.f05_macro(P.decide(va.with_columns(pl.Series("p", m1.predict(Xv, num_threads=N_JOBS))), "p", thr), ntv, ev)
    log(f"hold-out F0.5: original {f0:.5f}  adapted {f1:.5f}")
    if f1 < f0 - MAX_DROP:
        log("adapted model rejected (hold-out drop too large); outputs unchanged")
        return
    m1.save_model(str(WORK / "pass2_adapted.txt"))

    # ---- re-score unseen-country pairs and rewrite their output rows ----
    p_old = d["p_final"].to_numpy()
    d = d.with_columns(pl.Series("p_final", m1.predict(d.select(cols).to_numpy(), num_threads=N_JOBS)))
    pred = P.decide(d, "p_final", thr)
    s1 = load("test", 1).select("eid", "entity_id")
    cand = pl.concat([load("test", 2), load("test", 3)]).select(pl.col("eid").alias("cid"),
                                                                pl.col("entity_id").alias("cand_id"))
    new = (pred.select("eid", "cid").join(s1, on="eid").join(cand, on="cid")
               .group_by("entity_id").agg(pl.col("cand_id").unique().sort().str.join(",").alias("new_ids")))
    uns_ids = s1.filter(pl.col("eid").is_in(eids.implode())).select("entity_id")
    mr = pl.read_csv(OUT / "matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
    mr = (mr.join(new, left_on="source1_entity_id", right_on="entity_id", how="left")
            .with_columns(pl.when(pl.col("source1_entity_id").is_in(uns_ids["entity_id"].implode()))
                            .then(pl.col("new_ids").fill_null("")).otherwise(pl.col("matched_entity_ids"))
                            .alias("matched_entity_ids"))
            .select("source1_entity_id", "matched_entity_ids"))
    mr.write_csv(OUT / "matching_results.tsv", separator="\t", quote_style="never")
    unsure_old = ((p_old > 0.3) & (p_old < 0.7)).mean()
    unsure_new = ((d["p_final"].to_numpy() > 0.3) & (d["p_final"].to_numpy() < 0.7)).mean()
    log(f"unseen-country: matches {pred.height:,} ({pred.height/len(eids):.2f}/S1); "
        f"uncertain share {unsure_old:.4f} -> {unsure_new:.4f}")


if __name__ == "__main__":
    main()
