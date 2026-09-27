"""Scalable candidate generation (blocking).

The index is sharded by the `country` label (an open set: unseen labels such as
France simply become their own shard), and within a shard every record emits
hashed blocking keys:

  single keys   n|<core name token>  k|<phonetic skeleton>  c|<concatenated name>
                a|<address word>     h|<address number>
  compound keys (name-token x address-word), (number x address-word),
                (name-token x number) built from each record's rarest tokens.
                Individually common tokens ("douglas", "722") become highly
                selective in combination, while remaining robust to typos in
                any single field.

Candidate generation per S1 record:
  1. probe: look up only its M rarest keys with document-frequency <= DF_MAX
     (prefix filtering) -> work per record is bounded by M * DF_MAX regardless
     of corpus size; aggregate IDF of shared probe keys, keep top K_PROBE.
  2. re-score: IDF-weighted cosine over all *single* keys of both records.
  3. prune: top-K per S1, relative-score cut, and a mutual-rank constraint
     (the S1 must be among the top-R S1 records for that candidate, since each
     S2/S3 record belongs to at most one S1 entity).
"""
import math
import sys
import time

import polars as pl

from common import WORK

DF_MAX = 500
M_PROBE = 16
K_PROBE = 60
K_FINAL = 10
REL_MIN = 0.3
MUTUAL_R = 2
TOP_NAME, TOP_ADDR, TOP_NUM = 3, 4, 3

ADDR_STOP = ["st", "rd", "ave", "dr", "blvd", "ln", "ct", "pl", "unit", "no", "the", "and", "of",
             "de", "du", "des", "la", "le", "nr", "opp", "flr", "floor", "road", "near", "box",
             "cnty", "city", "door", "shop", "office", "plot", "bldg", "sec", "vill", "dist"]
OFFSETS = {1: 0, 2: 100_000_000, 3: 200_000_000}


def load(split, s):
    df = pl.read_parquet(WORK / f"{split}_s{s}.parquet")
    return df.with_row_index("eid", offset=OFFSETS[s])


def add_skeleton(df):
    from normalize import skeleton
    return df.with_columns(pl.col("n_core").map_elements(
        lambda s: " ".join(skeleton(t) for t in s.split() if len(t) >= 3), return_dtype=pl.String
    ).alias("n_skel"))


def _tok(col, prefix, minlen, stop=None):
    e = pl.element()
    cond = e.str.len_chars() >= minlen
    if stop:
        cond = cond & ~e.is_in(stop)
    return pl.col(col).str.split(" ").list.eval(pl.lit(prefix) + e.filter(cond))


def single_keys(df):
    """(eid, key:u64, kt:str) one row per distinct single key."""
    lst = pl.concat_list(
        _tok("n_core", "n|", 2), _tok("n_skel", "k|", 3),
        pl.concat_list(pl.lit("c|") + pl.col("n_core").str.replace_all(" ", "")),
        _tok("a_words", "a|", 3, ADDR_STOP), _tok("a_nums", "h|", 1))
    return (df.select("eid", lst.alias("key"))
              .explode("key")
              .filter(pl.col("key").str.len_chars() > 2)
              .select("eid", pl.col("key").str.slice(0, 1).alias("kt"),
                      pl.col("key").hash(seed=7).alias("key"))
              .unique(["eid", "key"]))


def compound_keys(sk):
    """sk: single keys with df. Cross the rarest tokens of each record."""
    def top(kt, n):
        return (sk.filter(pl.col("kt") == kt).sort(["eid", "df"])
                  .group_by("eid", maintain_order=True).head(n).select("eid", pl.col("key").alias(kt)))
    nt, aw, hn = top("n", TOP_NAME), top("a", TOP_ADDR), top("h", TOP_NUM)
    parts = []
    for a, b, salt in ((nt, aw, 1), (hn, aw, 2), (nt, hn, 3)):
        x, y = a.columns[1], b.columns[1]
        parts.append(a.join(b, on="eid").select(
            "eid", pl.struct(pl.col(x), pl.col(y), pl.lit(salt)).hash(seed=11).alias("key")))
    return pl.concat(parts).unique(["eid", "key"])


def build_shard(s1, cand):
    """Key tables for one country shard."""
    n_c = max(cand.height, 1)
    k1, kc = single_keys(s1), single_keys(cand)
    dfk = kc.group_by("key").len("df")
    kc = kc.join(dfk, on="key")
    k1 = k1.join(dfk, on="key", how="left").with_columns(pl.col("df").fill_null(0))
    c1, cc = compound_keys(k1), compound_keys(kc)
    dfc = cc.group_by("key").len("df")
    cc = cc.join(dfc, on="key")
    c1 = c1.join(dfc, on="key", how="left").with_columns(pl.col("df").fill_null(0))
    idf = pl.lit(math.log(n_c + 1)) - pl.col("df").add(1).log()
    k1 = k1.with_columns(idf.cast(pl.Float32).alias("w"))
    kc = kc.with_columns(idf.cast(pl.Float32).alias("w"))
    c1 = c1.with_columns(idf.cast(pl.Float32).alias("w"))
    cc = cc.with_columns(idf.cast(pl.Float32).alias("w"))
    return dict(k1=k1, kc=kc.rename({"eid": "cid"}), c1=c1, cc=cc.rename({"eid": "cid"}))


def score_shard(ix, df_max=DF_MAX, m_probe=M_PROBE, k_probe=K_PROBE, chunk=100_000):
    k1, kc, c1, cc = ix["k1"], ix["kc"], ix["c1"], ix["cc"]
    probe_src = pl.concat([k1.select("eid", "key", "df", "w"), c1.select("eid", "key", "df", "w")])
    probe = (probe_src.filter((pl.col("df") >= 1) & (pl.col("df") <= df_max))
                      .sort(["eid", "df"]).group_by("eid", maintain_order=True).head(m_probe)
                      .select("eid", "key"))
    idx = pl.concat([kc.filter(pl.col("df") <= df_max).select("cid", "key", "w"),
                     cc.filter(pl.col("df") <= df_max).select("cid", "key", "w")])
    norm1 = k1.group_by("eid").agg(pl.col("w").pow(2).sum().sqrt().alias("n1"))
    normc = kc.group_by("cid").agg(pl.col("w").pow(2).sum().sqrt().alias("nc"))
    kc_all = kc.select("cid", "key")
    ids = k1["eid"].unique().sort()
    parts = []
    for i in range(0, len(ids), chunk):
        lo, hi = ids[i], ids[min(i + chunk, len(ids)) - 1]
        rng = pl.col("eid").is_between(lo, hi)
        pairs = (probe.filter(rng).join(idx, on="key")
                      .group_by("eid", "cid").agg(pl.col("w").sum().alias("ps"))
                      .filter(pl.col("ps").rank("ordinal", descending=True).over("eid") <= k_probe))
        mine = k1.filter(rng).select("eid", "key", "w", "kt")
        shared = (pairs.select("eid", "cid").join(mine, on="eid").join(kc_all, on=["cid", "key"])
                       .group_by("eid", "cid").agg(
                           pl.col("w").pow(2).sum().alias("dot"),
                           pl.col("w").filter(pl.col("kt").is_in(["n", "k", "c"])).pow(2).sum().alias("dot_name"),
                           pl.col("w").filter(pl.col("kt") == "h").pow(2).sum().alias("dot_num")))
        parts.append(pairs.join(shared, on=["eid", "cid"], how="left").with_columns(pl.col("dot", "dot_name", "dot_num").fill_null(0.0)))
    sc = pl.concat(parts)
    sc = (sc.join(norm1, on="eid").join(normc, on="cid")
            .with_columns((pl.col("dot") / (pl.col("n1") * pl.col("nc"))).alias("bscore"),
                          (pl.col("dot_name") / (pl.col("n1") * pl.col("nc"))).alias("bscore_name"),
                          (pl.col("dot_num") / (pl.col("n1") * pl.col("nc"))).alias("bscore_num"))
            .drop("dot", "dot_name", "dot_num", "n1", "nc"))
    return sc


def add_ranks(sc):
    sc = sc.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over("eid").alias("rank_s1"),
        pl.col("bscore").rank("ordinal", descending=True).over("cid").alias("rank_c"),
        pl.col("ps").rank("ordinal", descending=True).over("eid").alias("prank_s1"),
        pl.col("bscore").max().over("eid").alias("best_s1"),
        pl.col("bscore").max().over("cid").alias("best_c"),
        pl.col("bscore").top_k(2).min().over("cid").alias("second_c"),
        pl.col("bscore").top_k(2).min().over("eid").alias("second_s1"),
        pl.len().over("cid").alias("n_s1_for_c"),
        pl.len().over("eid").alias("n_c_for_s1"),
    )
    return sc.with_columns(
        pl.when(pl.col("n_s1_for_c") > 1).then(pl.col("second_c")).otherwise(0.0).alias("second_c"),
        pl.when(pl.col("n_c_for_s1") > 1).then(pl.col("second_s1")).otherwise(0.0).alias("second_s1"))


def prune(sc, k_final=K_FINAL, rel_min=REL_MIN, mutual_r=MUTUAL_R):
    return sc.filter((pl.col("rank_s1") <= k_final)
                     & (pl.col("bscore") >= rel_min * pl.col("best_s1"))
                     & (pl.col("rank_c") <= mutual_r))


def block(split, s1_limit=None, log=True, **kw):
    """Returns (s1, cand, scored_pairs) where scored pairs carry blocking scores/ranks."""
    t0 = time.time()
    s1 = load(split, 1)
    if s1_limit:
        s1 = s1.head(s1_limit)
    s1 = add_skeleton(s1)
    cand = add_skeleton(pl.concat([load(split, 2), load(split, 3)]))
    out = []
    for country in sorted(s1["country"].unique().to_list()):
        a = s1.filter(pl.col("country") == country)
        b = cand.filter(pl.col("country") == country)
        if a.height == 0 or b.height == 0:
            continue
        ix = build_shard(a, b)
        if log:
            print(f"[{country}] s1 {a.height:,} cand {b.height:,} keys: s1 {ix['k1'].height+ix['c1'].height:,}"
                  f" cand {ix['kc'].height+ix['cc'].height:,} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        sc = score_shard(ix, **kw)
        del ix
        out.append(sc)
        if log:
            print(f"[{country}] scored pairs {sc.height:,} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    sc = add_ranks(pl.concat(out))
    return s1, cand, sc


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    s1, cand, sc = block(split)
    sc.write_parquet(WORK / f"{split}_scored.parquet")
