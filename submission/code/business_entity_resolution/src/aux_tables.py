"""Auxiliary lookup tables used by the pair features.

addr_common.json: per country label, the most frequent address words (cities,
regions, departments, state codes).  Removing them isolates street-level
agreement.  Built from the (unlabelled) source files of both splits, so an
unseen country such as France gets its own list automatically.
"""
import json
import sys

import polars as pl

from common import WORK

TOP_N = 400


def build_addr_common(top_n=TOP_N):
    frames = [pl.scan_parquet(WORK / f"{sp}_s{s}.parquet").select("country", "a_words")
              for sp in ("train", "test") for s in (1, 2, 3)]
    w = (pl.concat(frames).with_columns(pl.col("a_words").str.split(" ").list.unique())
           .explode("a_words").filter(pl.col("a_words").str.len_chars() > 0)
           .group_by("country", "a_words").len()
           .sort(["country", "len"], descending=[False, True])
           .group_by("country", maintain_order=True).head(top_n).collect())
    out = {c: w.filter(pl.col("country") == c)["a_words"].to_list() for c in w["country"].unique().to_list()}
    json.dump(out, open(WORK / "addr_common.json", "w"))
    print({c: v[:15] for c, v in out.items()}, file=sys.stderr)


if __name__ == "__main__":
    build_addr_common()
