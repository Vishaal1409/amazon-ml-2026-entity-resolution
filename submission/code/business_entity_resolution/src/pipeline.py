"""Shared cascade logic used by training and inference.

stage 1  blocking.py        -> scored pairs (~55 / S1)
stage 2  candidate filter   -> LightGBM on cheap vectorised features; survivors
                               are the candidate set (candidate_pairs.tsv)
stage 3  matcher            -> LightGBM on rich string features + context
                               over the candidate set, then decision rule
"""
import sys
import time

import numpy as np
import polars as pl

import features as F
from blocking import load
from cheap_features import CHEAP_FEATS, cheap_features, side_frame
from common import WORK, N_JOBS

S2_TOPN = 8          # at most this many candidates per S1 after the filter
RICH_FEATS = None    # filled lazily


def stage2_frames(split, sc, chunk_rows=12_000_000):
    """Yield cheap-feature frames for the blocking pairs, chunked by S1 id."""
    s1side = side_frame(split, 1)
    cside = pl.concat([side_frame(split, 2), side_frame(split, 3)])
    eids = sc["eid"].unique().sort()
    per = max(1, int(chunk_rows / max(1, sc.height / max(1, len(eids)))))
    for i in range(0, len(eids), per):
        lo, hi = eids[i], eids[min(i + per, len(eids)) - 1]
        yield cheap_features(sc.filter(pl.col("eid").is_between(lo, hi)), s1side, cside)


def stage2_filter(split, sc, model, thr, topn=S2_TOPN, log=True):
    """Run the candidate filter; returns survivors with p2 (and y if present)."""
    t0 = time.time()
    out = []
    for fr in stage2_frames(split, sc):
        p = model.predict(fr.select(CHEAP_FEATS).to_numpy(), num_threads=N_JOBS)
        fr = fr.select(["eid", "cid"] + CHEAP_FEATS + [c for c in ("y",) if c in fr.columns]).with_columns(
            pl.Series("p2", p, dtype=pl.Float32))
        fr = fr.filter(pl.col("p2") >= thr)
        out.append(fr)
    s = pl.concat(out)
    s = s.filter(pl.col("p2").rank("ordinal", descending=True).over("eid") <= topn)
    if log:
        print(f"stage2: {s.height:,} survivors ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    return s


def context(df, score="p2"):
    """Group context over the candidate set: competition for the same record."""
    s = pl.col(score)
    return df.with_columns(
        s.rank("ordinal", descending=True).over("eid").alias("c_rank_s1"),
        s.rank("ordinal", descending=True).over("cid").alias("c_rank_c"),
        (s - s.max().over("eid")).alias("c_gap_best_s1"),
        (s - pl.when(pl.len().over("cid") > 1)
            .then(s.top_k(2).min().over("cid")).otherwise(0.0)).alias("c_gap_second_c"),
        (s - s.max().over("cid")).alias("c_gap_best_c"),
        pl.len().over("eid").alias("c_n_s1"),
        pl.len().over("cid").alias("c_n_c"),
        s.sum().over("eid").alias("c_sum_s1"),
    )


CTX_FEATS = ["c_rank_s1", "c_rank_c", "c_gap_best_s1", "c_gap_second_c", "c_gap_best_c",
             "c_n_s1", "c_n_c", "c_sum_s1"]


def stage3_frame(split, surv):
    """Rich features over the candidate set."""
    s1 = load(split, 1)
    cand = pl.concat([load(split, 2), load(split, 3)])
    d = F.compute(surv, s1, cand)
    return context(d, "p2")


def stage3_cols():
    return CHEAP_FEATS + ["p2"] + [f"r_{c}" for c in F.STR_FEATS] + CTX_FEATS


def decide(df, score="p3", thr=0.5):
    """One-to-one assignment (each S2/S3 record goes to its best S1) + threshold."""
    return df.filter((pl.col(score) >= thr)
                     & (pl.col(score).rank("ordinal", descending=True).over("cid") == 1))


def f05_macro(pred, truth_counts, all_eids):
    """pred: (eid, cid, y) predicted pairs; truth_counts: (eid, nt); all_eids: Series of eids.
    Per-entity F0.5 = 1.25*tp / (0.25*nt + np); both empty -> 1."""
    agg = pred.group_by("eid").agg(pl.col("y").sum().alias("tp"), pl.len().alias("np"))
    e = (pl.DataFrame({"eid": all_eids}).join(truth_counts, on="eid", how="left")
           .join(agg, on="eid", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    return e.select(f.mean()).item()
