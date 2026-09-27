"""Pairwise features for candidate (S1, S2/S3) pairs.

All features are script/language agnostic string similarities, so they transfer
to countries unseen in training (France).  Country itself is NOT a feature.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from common import WORK, N_JOBS
from normalize import skeleton

REC_COLS = ["n_norm", "n_core", "n_legal", "a_norm", "a_words", "a_nums", "indic"]

STR_FEATS = [
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "core_ratio", "core_tsort", "core_tset",
    "core_partial", "core_jw", "core_lev", "concat_ratio", "concat_eq", "concat_partial",
    "core_jacc", "core_ovl", "skel_jacc", "skel_ovl", "first_tok_eq", "first_tok_jw", "tok_cover1", "tok_cover2",
    "n_core1", "n_core2", "len_core1", "len_core2",
    "legal1", "legal2", "legal_eq", "legal_conflict",
    "ad_ratio", "ad_tset", "ad_tsort", "ad_partial", "adw_tset", "adw_jacc", "adw_ovl",
    "num_jacc", "num_ovl", "num_any", "hn_eq", "zip_eq", "zip_conflict", "n_nums1", "n_nums2",
    "addr_empty2", "addr_len1", "addr_len2", "indic2",
]


def _set_stats(a, b):
    if not a or not b:
        return 0.0, 0.0
    i = len(a & b)
    return i / len(a | b), i / min(len(a), len(b))


def _fuzzy_cover(a, b):
    """fraction of tokens in a that have a JW>=0.9 match in b"""
    if not a:
        return 0.0
    hit = 0
    for t in a:
        for u in b:
            if t == u or JaroWinkler.similarity(t, u) >= 0.9:
                hit += 1
                break
    return hit / len(a)


def _zips(nums):
    return {n for n in nums if len(n) >= 5}


def pair_feats(r1, r2):
    n1, c1, l1, a1, aw1, an1, _ = r1
    n2, c2, l2, a2, aw2, an2, ind2 = r2
    t1, t2 = c1.split(), c2.split()
    s1, s2 = set(t1), set(t2)
    k1 = {skeleton(t) for t in t1}
    k2 = {skeleton(t) for t in t2}
    cc1, cc2 = c1.replace(" ", ""), c2.replace(" ", "")
    lg1, lg2 = set(l1.split()), set(l2.split())
    nums1, nums2 = an1.split(), an2.split()
    ns1, ns2 = set(nums1), set(nums2)
    z1, z2 = _zips(ns1), _zips(ns2)
    w1, w2 = set(aw1.split()), set(aw2.split())
    cj, co = _set_stats(s1, s2)
    kj, ko = _set_stats(k1, k2)
    wj, wo = _set_stats(w1, w2)
    nj, no = _set_stats(ns1, ns2)
    out = [
        fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2),
        fuzz.partial_ratio(n1, n2),
        fuzz.ratio(c1, c2), fuzz.token_sort_ratio(c1, c2), fuzz.token_set_ratio(c1, c2),
        fuzz.partial_ratio(c1, c2), JaroWinkler.similarity(c1, c2), Levenshtein.distance(c1, c2),
        fuzz.ratio(cc1, cc2), float(cc1 == cc2 and cc1 != ""), fuzz.partial_ratio(cc1, cc2),
        cj, co, kj, ko,
        float(bool(t1) and bool(t2) and t1[0] == t2[0]),
        JaroWinkler.similarity(t1[0], t2[0]) if t1 and t2 else 0.0,
        _fuzzy_cover(t1, t2), _fuzzy_cover(t2, t1),
        len(t1), len(t2), len(cc1), len(cc2),
        float(bool(lg1)), float(bool(lg2)), float(lg1 == lg2), float(bool(lg1) and bool(lg2) and not (lg1 & lg2)),
        fuzz.ratio(a1, a2), fuzz.token_set_ratio(a1, a2), fuzz.token_sort_ratio(a1, a2),
        fuzz.partial_ratio(a1, a2) if a1 and a2 else 0.0,
        fuzz.token_set_ratio(aw1, aw2), wj, wo,
        nj, no, float(bool(ns1 & ns2)),
        float(bool(nums1) and bool(nums2) and nums1[0] == nums2[0]),
        float(bool(z1 & z2)), float(bool(z1) and bool(z2) and not (z1 & z2)),
        len(nums1), len(nums2),
        float(a2 == ""), len(a1), len(a2), float(ind2),
    ]
    return out


def _work(args):
    left, right = args
    return np.array([pair_feats(a, b) for a, b in zip(left, right)], dtype=np.float32)


def compute(pairs, s1, cand, chunk=20000):
    """pairs: DataFrame with eid, cid (+ blocking cols). s1/cand: normalised frames with eid."""
    t0 = time.time()
    d = (pairs.join(s1.select(["eid"] + REC_COLS), on="eid")
              .join(cand.select([pl.col("eid").alias("cid")] + [pl.col(c).alias(c + "_2") for c in REC_COLS]),
                    on="cid"))
    left = list(zip(*[d[c].to_list() for c in REC_COLS]))
    right = list(zip(*[d[c + "_2"].to_list() for c in REC_COLS]))
    jobs = [(left[i:i + chunk], right[i:i + chunk]) for i in range(0, len(left), chunk)]
    with Pool(N_JOBS) as pool:
        mats = pool.map(_work, jobs)
    X = np.vstack(mats) if mats else np.zeros((0, len(STR_FEATS)), np.float32)
    feats = pl.DataFrame(X, schema=[f"r_{c}" for c in STR_FEATS])
    base = d.drop([c for c in d.columns if c in REC_COLS or c.endswith("_2")])
    print(f"features: {len(left):,} pairs in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    return pl.concat([base, feats], how="horizontal")


def context_feats(df):
    """Group-level features computed from the blocking scores."""
    return df.with_columns(
        (pl.col("bscore") / pl.col("best_s1")).alias("rel_s1"),
        (pl.col("bscore") / pl.col("best_c")).alias("rel_c"),
        (pl.col("bscore") - pl.when(pl.col("rank_c") == 1).then(pl.col("second_c"))
         .otherwise(pl.col("best_c"))).alias("margin_c"),
        (pl.col("bscore") - pl.when(pl.col("rank_s1") == 1).then(pl.col("second_s1"))
         .otherwise(pl.col("best_s1"))).alias("margin_s1"),
        pl.len().over("eid").alias("n_cands"),
        (pl.col("cid") >= 200_000_000).cast(pl.Int8).alias("is_s3"),
    )
