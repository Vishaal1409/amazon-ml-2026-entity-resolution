"""Vectorised (polars-only) pair features for the candidate-filter stage.

These are cheap enough to evaluate on every blocking pair (~55 per S1 record).
"""
import polars as pl

from blocking import add_skeleton, load

SIDE_COLS = ["eid", "n_core", "n_skel", "n_legal", "a_words", "a_nums", "a_norm", "indic"]

BLOCK_FEATS = ["ps", "bscore", "bscore_name", "bscore_num", "rank_s1", "rank_c", "prank_s1",
               "best_s1", "best_c", "second_c", "second_s1", "n_s1_for_c", "n_c_for_s1"]
CHEAP_FEATS = BLOCK_FEATS + [
    "rel_s1", "rel_c", "margin_c", "margin_s1", "is_s3",
    "nm_jacc", "nm_ovl", "sk_jacc", "sk_ovl", "aw_jacc", "aw_ovl", "num_jacc", "num_ovl",
    "hn_eq", "concat_eq", "legal_eq", "legal_conflict", "addr_empty2", "indic2",
    "n_tok1", "n_tok2", "n_aw1", "n_aw2", "n_num1", "n_num2",
]


def side_frame(split, s):
    df = load(split, s)
    df = add_skeleton(df)
    sp = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return df.select(
        "eid",
        sp("n_core").alias("tn"), sp("n_skel").alias("tk"), sp("a_words").alias("ta"),
        sp("a_nums").alias("tu"), sp("n_legal").alias("tl"),
        pl.col("n_core").str.replace_all(" ", "").alias("cc"),
        (pl.col("a_norm") == "").alias("addr_empty"), "indic")


def _jo(a, b, pre):
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    la, lb = pl.col(a).list.len(), pl.col(b).list.len()
    union = la + lb - inter
    return [
        pl.when(union > 0).then(inter / union).otherwise(0.0).alias(pre + "_jacc"),
        pl.when(pl.min_horizontal(la, lb) > 0).then(inter / pl.min_horizontal(la, lb)).otherwise(0.0).alias(pre + "_ovl"),
    ]


def cheap_features(sc, s1side, cside):
    d = sc.join(s1side, on="eid").join(cside.rename(lambda c: c if c == "eid" else c + "2").rename({"eid": "cid"}),
                                       on="cid")
    d = d.with_columns(
        *_jo("tn", "tn2", "nm"), *_jo("tk", "tk2", "sk"), *_jo("ta", "ta2", "aw"), *_jo("tu", "tu2", "num"),
        (pl.col("tu").list.first() == pl.col("tu2").list.first()).fill_null(False).cast(pl.Int8).alias("hn_eq"),
        ((pl.col("cc") == pl.col("cc2")) & (pl.col("cc") != "")).cast(pl.Int8).alias("concat_eq"),
        (pl.col("tl").list.sort() == pl.col("tl2").list.sort()).cast(pl.Int8).alias("legal_eq"),
        ((pl.col("tl").list.len() > 0) & (pl.col("tl2").list.len() > 0)
         & (pl.col("tl").list.set_intersection(pl.col("tl2")).list.len() == 0)).cast(pl.Int8).alias("legal_conflict"),
        pl.col("addr_empty2").cast(pl.Int8), pl.col("indic2").cast(pl.Int8).alias("indic2"),
        pl.col("tn").list.len().alias("n_tok1"), pl.col("tn2").list.len().alias("n_tok2"),
        pl.col("ta").list.len().alias("n_aw1"), pl.col("ta2").list.len().alias("n_aw2"),
        pl.col("tu").list.len().alias("n_num1"), pl.col("tu2").list.len().alias("n_num2"),
        (pl.col("bscore") / pl.col("best_s1")).fill_nan(0).alias("rel_s1"),
        (pl.col("bscore") / pl.col("best_c")).fill_nan(0).alias("rel_c"),
        (pl.col("bscore") - pl.when(pl.col("rank_c") == 1).then(pl.col("second_c"))
         .otherwise(pl.col("best_c"))).alias("margin_c"),
        (pl.col("bscore") - pl.when(pl.col("rank_s1") == 1).then(pl.col("second_s1"))
         .otherwise(pl.col("best_s1"))).alias("margin_s1"),
        (pl.col("cid") >= 200_000_000).cast(pl.Int8).alias("is_s3"),
    )
    keep = ["eid", "cid"] + CHEAP_FEATS + [c for c in ("y",) if c in d.columns]
    return d.select(keep)
