"""Normalise all source files (train + test) into parquet with cleaned fields."""
import json
import sys
from multiprocessing import Pool

import polars as pl

from common import read_tsv, DATA, WORK, N_JOBS
import normalize as nz


def _init(d, v):
    nz.set_translit_dict(d)
    nz.set_vocab(v)


def _work(chunk):
    names, addrs = chunk
    out = {k: [] for k in ("n_norm", "n_core", "n_legal", "a_norm", "a_words", "a_nums", "indic")}
    for n, a in zip(names, addrs):
        f = nz.name_features(n)
        g = nz.addr_features(a)
        for k, v in f.items():
            out[k].append(v)
        for k, v in g.items():
            out[k].append(v)
        out["indic"].append(nz.has_indic(n) or nz.has_indic(a))
    return out


def process(df, pool, chunk=20000):
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    cols = {k: [] for k in ("n_norm", "n_core", "n_legal", "a_norm", "a_words", "a_nums", "indic")}
    for res in pool.imap(_work, chunks):
        for k in cols:
            cols[k].extend(res[k])
    return df.select("entity_id", pl.col("country").str.strip_chars().str.to_lowercase()).with_columns(
        **{k: pl.Series(v) for k, v in cols.items()})


def main(splits=("train", "test")):
    d = json.load(open(WORK / "translit_dict.json"))
    v = json.load(open(WORK / "name_vocab.json"))
    with Pool(N_JOBS, initializer=_init, initargs=(d, v)) as pool:
        for split in splits:
            for s in (1, 2, 3):
                src = DATA / split / f"{split}_source{s}.tsv"
                df = read_tsv(src)
                out = process(df, pool)
                out.write_parquet(WORK / f"{split}_s{s}.parquet")
                print(split, s, out.shape, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("train", "test"))
