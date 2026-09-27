"""Pass 2: collective matching with sibling evidence.

After pass 1, every S1 entity has a few *confident* matches (p3 >= CONF).  Each
S2/S3 record belongs to one real entity, so those confident siblings are extra
views of the same business (other name variant, other address fragment).

  1. sibling probe: each confident sibling is used as a query against the same
     blocking index (compound keys, prefix filtering) -> new candidate pairs
     (S1, record) that the S1 record alone could not reach.
  2. pair universe = pass-1 candidate set  U  sibling-probe top pairs.
  3. features = rich similarity to the S1 record  +  rich similarity to the
     best sibling  +  sibling-probe scores  +  pass-1 probabilities
     (missing for new pairs)  +  competition for the record across S1s.
  4. LightGBM pass-2 model -> one-to-one assignment -> threshold.

The candidate set written to candidate_pairs.tsv is the pass-2 pair universe
(the exact set the pass-2 model scores).
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import features as F
import pipeline as P
from blocking import add_skeleton, build_shard, load, score_shard
from common import N_JOBS, WORK

CONF = 0.9          # pass-1 probability for a sibling to be trusted
SIB_TOPN = 4        # new pairs per S1 taken from the sibling probe (training universe)
SIB_TOPN_PRED = 2   # at inference: keep the top-2 sibling pairs ...
SIB_MIN_PRED = 0.5  # ... with sibling cosine >= 0.5 (4.7 vs 7.5 cands/S1 for -0.0003 F0.5)
SEED = 11


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sibling_probe(split, conf, cand):
    """conf: (eid, sib) confident pairs. Returns (eid, cid, sib, sib_ps, sib_bscore)."""
    t0 = time.time()
    q = cand.join(conf.select(pl.col("sib").alias("eid")).unique(), on="eid")
    out = []
    for country in sorted(q["country"].unique().to_list()):
        a = q.filter(pl.col("country") == country)
        b = cand.filter(pl.col("country") == country)
        ix = build_shard(a, b)
        sc = score_shard(ix, k_probe=30)
        del ix
        out.append(sc.select(pl.col("eid").alias("sib"), "cid", pl.col("ps").alias("sib_ps"),
                             pl.col("bscore").alias("sib_bscore"))
                     .filter(pl.col("sib") != pl.col("cid")))
        log(f"  sibling probe [{country}] queries {a.height:,} ({time.time()-t0:.0f}s)")
    sp = pl.concat(out)
    return sp.join(conf, on="sib")


def build_universe(split, pass1, sc):
    """pass1: (eid, cid, p2, p3) over pass-1 candidates. sc: stage-1 scored pairs."""
    cand = add_skeleton(pl.concat([load(split, 2), load(split, 3)]))
    top = pass1.filter(pl.col("p3").rank("ordinal", descending=True).over("cid") == 1)
    conf = top.filter(pl.col("p3") >= CONF).select("eid", pl.col("cid").alias("sib"),
                                                    pl.col("p3").alias("sib_p3"))
    log(f"confident siblings {conf.height:,} for {conf['eid'].n_unique():,} S1")
    sp = sibling_probe(split, conf, cand)
    # aggregate sibling evidence per (eid, cid); remember the best sibling
    sp = sp.filter(pl.col("cid") != pl.col("sib"))
    agg = (sp.sort("sib_bscore", descending=True)
             .group_by("eid", "cid").agg(pl.col("sib_bscore").max(), pl.col("sib_ps").max(),
                                        pl.len().alias("sib_hits"), pl.col("sib").first().alias("best_sib")))
    # do not propose records that are already confident siblings of the same S1
    agg = agg.join(conf.select("eid", pl.col("sib").alias("cid")), on=["eid", "cid"], how="anti")
    new = (agg.join(pass1.select("eid", "cid"), on=["eid", "cid"], how="anti")
              .filter(pl.col("sib_bscore").rank("ordinal", descending=True).over("eid") <= SIB_TOPN))
    uni = pl.concat([pass1.select("eid", "cid"), new.select("eid", "cid")]).unique()
    uni = (uni.join(pass1, on=["eid", "cid"], how="left")
              .join(agg, on=["eid", "cid"], how="left")
              .join(sc.select("eid", "cid", "ps", "bscore", "bscore_name", "bscore_num", "rank_s1",
                              "rank_c", "prank_s1"), on=["eid", "cid"], how="left"))
    # pass-1 view of the record across S1s, and of the S1 across records
    uni = uni.with_columns(
        pl.col("p3").is_null().cast(pl.Int8).alias("is_new"),
        pl.col("sib_hits").fill_null(0),
        pl.len().over("eid").alias("u_n_s1"),
        pl.len().over("cid").alias("u_n_c"),
        (pl.col("p3").fill_null(0).max().over("cid") - pl.col("p3").fill_null(0)).alias("c_best_other_gap"),
        pl.col("p3").fill_null(0).max().over("cid").alias("c_best_p3"),
    )
    nconf = conf.group_by("eid").agg(pl.len().alias("n_conf"),
                                     (pl.col("sib") >= 200_000_000).sum().alias("n_conf_s3"))
    uni = uni.join(nconf, on="eid", how="left").with_columns(
        pl.col("n_conf").fill_null(0), pl.col("n_conf_s3").fill_null(0),
        (pl.col("cid") >= 200_000_000).cast(pl.Int8).alias("is_s3"))
    uni = uni.with_columns(
        pl.when(pl.col("is_s3") == 1).then(pl.col("n_conf_s3"))
          .otherwise(pl.col("n_conf") - pl.col("n_conf_s3")).alias("n_conf_same_src"))
    # competitor confidence: is this record a confident sibling of *another* S1?
    other = conf.select(pl.col("sib").alias("cid"), pl.col("eid").alias("_o"), pl.col("sib_p3").alias("other_conf_p3"))
    uni = (uni.join(other, on="cid", how="left")
              .with_columns(pl.when(pl.col("_o") == pl.col("eid")).then(None)
                              .otherwise(pl.col("other_conf_p3")).alias("other_conf_p3"))
              .drop("_o").unique(["eid", "cid"]))
    return uni, cand


def rich_vs_sibling(uni, cand):
    """String features between the candidate and the S1's best sibling that found it."""
    has = uni.filter(pl.col("best_sib").is_not_null()).select("eid", "cid", pl.col("best_sib"))
    q = has.select(pl.col("best_sib").alias("eid"), "cid")
    d = F.compute(q, cand, cand)
    d = d.rename({c: "s" + c[1:] for c in d.columns if c.startswith("r_")})
    d = has.join(d.rename({"eid": "best_sib"}), on=["best_sib", "cid"], how="left").drop("best_sib")
    return uni.join(d.unique(["eid", "cid"]), on=["eid", "cid"], how="left")


def pass2_frame(split, pass1, sc):
    t0 = time.time()
    uni, cand = build_universe(split, pass1, sc)
    log(f"pass-2 universe {uni.height:,} pairs ({(uni['is_new']==1).sum():,} new) ({time.time()-t0:.0f}s)")
    s1 = load(split, 1)
    d = F.compute(uni, s1, cand)
    d = rich_vs_sibling(d, cand)
    return d


def pass2_cols(d):
    drop = {"eid", "cid", "y", "best_sib", "p_final"}
    return [c for c in d.columns if c not in drop and d[c].dtype != pl.String]


def train_and_validate():
    import train as T
    s1ids, pos, nt = T.truth()
    fc = pl.read_parquet(WORK / "train_stage3_C.parquet")
    m3 = lgb.Booster(model_file=str(WORK / "stage3.txt"))
    fc = fc.with_columns(pl.Series("p3", m3.predict(fc.select(P.stage3_cols()).to_numpy(), num_threads=N_JOBS)))
    pass1 = fc.select("eid", "cid", "p2", "p3")
    cfg = json.load(open(WORK / "model_config.json"))
    sc = (pl.scan_parquet(WORK / "train_scored.parquet")
            .filter(pl.col("eid").is_in(pass1["eid"].unique().implode())).collect())
    d = pass2_frame("train", pass1, sc)
    d = d.join(pos, on=["eid", "cid"], how="left").with_columns(pl.col("y").fill_null(0))
    d.write_parquet(WORK / "train_pass2_C.parquet")
    cols = pass2_cols(d)
    eidsC = s1ids.with_columns(T.fold_of("eid").alias("f")).filter(pl.col("f") >= 7)["eid"]
    split = pl.col("eid").hash(seed=SEED) % 5 < 3
    tr, va = d.filter(split), d.filter(~split)
    eids_va = eidsC.filter(eidsC.hash(seed=SEED) % 5 >= 3)
    nt_va = nt.filter(pl.col("eid").is_in(eids_va.implode()))
    inner = tr["eid"].hash(seed=3) % 10 == 0
    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  verbose=-1, num_threads=N_JOBS, seed=SEED)
    m = T.lgb_train(tr.filter(~inner).select(cols).to_numpy(), tr.filter(~inner)["y"].to_numpy(),
                    tr.filter(inner).select(cols).to_numpy(), tr.filter(inner)["y"].to_numpy(), params, 3000)
    m.save_model(str(WORK / "pass2.txt"))
    va = va.with_columns(pl.Series("p_final", m.predict(va.select(cols).to_numpy(), num_threads=N_JOBS)))
    base = P.f05_macro(P.decide(va.filter(pl.col("is_new") == 0), "p3", cfg["thr3"]), nt_va, eids_va)
    best = (0, 0.5)
    for t in np.arange(0.3, 0.95, 0.025):
        f = P.f05_macro(P.decide(va, "p_final", t), nt_va, eids_va)
        best = max(best, (f, float(t)))
    pred = P.decide(va, "p_final", best[1])
    ntrue = pos.filter(pl.col("eid").is_in(eids_va.implode())).height
    prec, rec = pred["y"].sum() / pred.height, pred["y"].sum() / ntrue
    ceil = va["y"].sum() / ntrue
    upper = P.f05_macro(va.filter(pl.col("y") == 1).select("eid", "cid", "y"), nt_va, eids_va)
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])[:20]
    log("top features:", [(c, int(g)) for c, g in imp])
    log(f"PASS2 VALIDATION ({len(eids_va):,} S1): pass-1 F0.5={base:.5f} -> pass-2 F0.5={best[0]:.5f} "
        f"at thr={best[1]:.3f}; precision={prec:.4f} recall={rec:.4f}; candidate recall={ceil:.4f}; "
        f"cands/S1={va.height/len(eids_va):.2f}; perfect-matcher bound={upper:.5f}")
    cfg.update({"pass2_thr": best[1], "pass2_f05_valid": best[0], "pass2_precision": prec,
                "pass2_recall": rec, "pass1_f05_same_split": base, "pass2_conf": CONF,
                "pass2_cols": cols})
    json.dump(cfg, open(WORK / "model_config.json", "w"), indent=1)


def predict(split="test"):
    from predict import write_lists
    from common import OUT
    cfg = json.load(open(WORK / "model_config.json"))
    m = lgb.Booster(model_file=str(WORK / "pass2.txt"))
    pass1 = pl.read_parquet(WORK / f"{split}_pass1.parquet")
    sc = pl.read_parquet(WORK / f"{split}_scored.parquet")
    d = pass2_frame(split, pass1, sc)
    del sc
    d = d.filter((pl.col("is_new") == 0) | (
        (pl.col("sib_bscore").rank("ordinal", descending=True).over("eid") <= SIB_TOPN_PRED)
        & (pl.col("sib_bscore") >= SIB_MIN_PRED)))
    d = d.with_columns(pl.Series("p_final", m.predict(d.select(cfg["pass2_cols"]).to_numpy(), num_threads=N_JOBS)))
    pred = P.decide(d, "p_final", cfg["pass2_thr"])
    s1 = load(split, 1).select("eid", "entity_id")
    cand = pl.concat([load(split, 2), load(split, 3)]).select(pl.col("eid").alias("cid"),
                                                              pl.col("entity_id").alias("cand_id"))
    name = lambda x: x.select("eid", "cid").join(s1, on="eid").join(cand, on="cid")
    write_lists(name(d), s1.select("entity_id"), "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    mm = write_lists(name(pred), s1.select("entity_id"), "matched_entity_ids", OUT / "matching_results.tsv")
    log(f"PASS2 TEST: S1 {s1.height:,}; candidates {d.height:,} ({d.height/s1.height:.2f}/S1); "
        f"matches {pred.height:,} ({pred.height/s1.height:.2f}/S1); empty {(mm['matched_entity_ids']=='').sum():,}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_and_validate()
    else:
        predict(sys.argv[2] if len(sys.argv) > 2 else "test")
