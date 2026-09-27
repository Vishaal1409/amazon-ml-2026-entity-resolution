"""Train the candidate filter (stage 2) and matcher (stage 3) on the training split.

S1 entities are split into disjoint folds so that each stage is trained on data
it has never been fit on:
    fold A (20%) -> stage-2 filter training
    fold B (50%) -> stage-3 matcher training (on stage-2 survivors of B)
    fold C (30%) -> hold-out validation of the whole cascade (F0.5 macro)
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import pipeline as P
from blocking import load
from cheap_features import CHEAP_FEATS
from common import DATA, WORK, N_JOBS, read_tsv

SEED = 7
S2_RECALL = 0.995    # stage-2 threshold keeps this fraction of reachable positives (on A-holdout)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def truth(split="train"):
    s1 = load(split, 1).select("eid", "entity_id")
    cand = pl.concat([load(split, 2), load(split, 3)]).select(pl.col("eid").alias("cid"),
                                                              pl.col("entity_id").alias("cand_id"))
    gt = (read_tsv(DATA / split / f"{split}_ground_truth.tsv")
          .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .rename({"source1_entity_id": "entity_id", "matched_entity_ids": "cand_id"}))
    pos = (gt.filter(pl.col("cand_id") != "").join(s1, on="entity_id").join(cand, on="cand_id")
             .select("eid", "cid").with_columns(pl.lit(1, pl.Int8).alias("y")))
    nt = pos.group_by("eid").agg(pl.len().alias("nt"))
    return s1, pos, nt


def fold_of(eid):
    return (pl.col(eid).hash(seed=SEED) % 10)


def lgb_train(X, y, Xv, yv, params, rounds):
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])


def main():
    t0 = time.time()
    s1, pos, nt = truth()
    def get(fold_filter):
        return (pl.scan_parquet(WORK / "train_scored.parquet")
                  .with_columns(fold_of("eid").alias("fold")).filter(fold_filter).drop("fold")
                  .join(pos.lazy(), on=["eid", "cid"], how="left")
                  .with_columns(pl.col("y").fill_null(0).cast(pl.Int8)).collect())

    # ---------------- stage 2: candidate filter ----------------
    if (WORK / "stage2.txt").exists() and (WORK / "stage2.json").exists():
        m2 = lgb.Booster(model_file=str(WORK / "stage2.txt"))
        thr2 = json.load(open(WORK / "stage2.json"))["thr2"]
        log(f"reusing stage2 model, thr {thr2:.4f}")
    else:
        m2, thr2 = train_stage2(get, t0)
    stage3(get, m2, thr2, s1, pos, nt, t0)


def train_stage2(get, t0):
    A = get(pl.col("fold") < 2)
    log(f"fold A pairs {A.height:,}")
    frames = [f for f in P.stage2_frames("train", A)]
    del A
    fa = pl.concat(frames)
    tr = fa.filter(pl.col("eid").hash(seed=99) % 5 != 0)
    va = fa.filter(pl.col("eid").hash(seed=99) % 5 == 0)
    # negative subsampling for speed (ranking quality is what matters here)
    tr = pl.concat([tr.filter(pl.col("y") == 1), tr.filter(pl.col("y") == 0).sample(fraction=0.35, seed=SEED)])
    params2 = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=0, seed=SEED)
    m2 = lgb_train(tr.select(CHEAP_FEATS).to_numpy(), tr["y"].to_numpy(),
                   va.select(CHEAP_FEATS).to_numpy(), va["y"].to_numpy(), params2, 600)
    p = m2.predict(va.select(CHEAP_FEATS).to_numpy())
    va = va.with_columns(pl.Series("p2", p))
    # threshold that keeps S2_RECALL of the reachable positives (within top-N)
    va_top = va.filter(pl.col("p2").rank("ordinal", descending=True).over("eid") <= P.S2_TOPN)
    pp = np.sort(va_top.filter(pl.col("y") == 1)["p2"].to_numpy())
    n_pos_va = va["y"].sum()
    k = int((1 - S2_RECALL) * n_pos_va) - (n_pos_va - len(pp))
    thr2 = float(pp[max(k, 0)]) if k > 0 else float(pp[0])
    n_s1_va = va["eid"].n_unique()
    for t in sorted({thr2, 0.005, 0.01, 0.02, 0.05, 0.1}):
        kept = va_top.filter(pl.col("p2") >= t)
        log(f"  stage2 thr={t:.4f}: recall(of reachable)={kept['y'].sum()/n_pos_va:.4f} "
            f"cands/S1={kept.height/n_s1_va:.2f}")
    log(f"stage2 threshold {thr2:.4f} ({time.time()-t0:.0f}s)")
    m2.save_model(str(WORK / "stage2.txt"))
    json.dump({"thr2": thr2}, open(WORK / "stage2.json", "w"))
    return m2, thr2


def stage3(get, m2, thr2, s1, pos, nt, t0):
    def survivors(fold_filter, name):
        path = WORK / f"train_surv_{name}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        s = P.stage2_filter("train", get(fold_filter), m2, thr2)
        s.write_parquet(path)
        return s

    B = survivors((pl.col("fold") >= 2) & (pl.col("fold") < 7), "B")
    C = survivors(pl.col("fold") >= 7, "C")
    fb = P.stage3_frame("train", B)
    fc = P.stage3_frame("train", C)
    cols = P.stage3_cols()
    fb.write_parquet(WORK / "train_stage3_B.parquet")
    fc.write_parquet(WORK / "train_stage3_C.parquet")
    trb = fb.filter(pl.col("eid").hash(seed=5) % 10 != 0)
    vab = fb.filter(pl.col("eid").hash(seed=5) % 10 == 0)
    params3 = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                   verbose=-1, num_threads=0, seed=SEED)
    m3 = lgb_train(trb.select(cols).to_numpy(), trb["y"].to_numpy(),
                   vab.select(cols).to_numpy(), vab["y"].to_numpy(), params3, 3000)
    m3.save_model(str(WORK / "stage3.txt"))
    imp = sorted(zip(cols, m3.feature_importance("gain")), key=lambda x: -x[1])
    log("top features:", [(c, int(g)) for c, g in imp[:25]])

    # ---------------- validation on fold C ----------------
    fc = fc.with_columns(pl.Series("p3", m3.predict(fc.select(cols).to_numpy(), num_threads=N_JOBS)))
    eids_c = s1.with_columns(fold_of("eid").alias("fold")).filter(pl.col("fold") >= 7)["eid"]
    nt_c = nt.filter(pl.col("eid").is_in(eids_c.implode()))
    best = (0, 0.5)
    for t in np.arange(0.2, 0.9, 0.025):
        f = P.f05_macro(P.decide(fc, "p3", t), nt_c, eids_c)
        if f > best[0]:
            best = (f, float(t))
        log(f"  thr={t:.3f} F0.5={f:.5f}")
    pred = P.decide(fc, "p3", best[1])
    n_true_c = pos.filter(pl.col("eid").is_in(eids_c.implode())).height
    prec, rec = pred["y"].sum() / max(pred.height, 1), pred["y"].sum() / n_true_c
    log(f"VALIDATION pair precision={prec:.4f} recall={rec:.4f}")
    cand_f = P.f05_macro(fc.select("eid", "cid", "y"), nt_c, eids_c)
    reach = fc["y"].sum() / pos.filter(pl.col("eid").is_in(eids_c.implode())).height
    log(f"VALIDATION (fold C, {len(eids_c):,} S1): best F0.5={best[0]:.5f} at thr={best[1]:.3f}; "
        f"candidate recall={reach:.4f}; cands/S1={fc.height/len(eids_c):.2f}; F0.5 if all cands={cand_f:.4f}")
    json.dump({"thr2": thr2, "thr3": best[1], "f05_valid": best[0], "precision": float(prec), "recall": float(rec), "s2_topn": P.S2_TOPN},
              open(WORK / "model_config.json", "w"), indent=1)
    log(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
