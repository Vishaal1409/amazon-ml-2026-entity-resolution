"""Learn a transliteration-token -> Latin-token dictionary from training matches.

For every matched pair where the S2/S3 record is written in an Indic script, each
rule-transliterated token is aligned to its most similar token in the Source 1
record (Latin).  Votes are aggregated and confident mappings are kept, e.g.
  praivet -> private, limitet -> limited, hariyana -> haryana
Only the provided training data is used.
"""
import json
import sys
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

from common import read_tsv, DATA, WORK
from normalize import fold, has_indic, _NON_ALNUM


def toks(s):
    return _NON_ALNUM.sub(" ", fold(s, use_dict=False)).split()


def main():
    s1 = read_tsv(DATA / "train/train_source1.tsv")
    cands = pl.concat([read_tsv(DATA / "train/train_source2.tsv"), read_tsv(DATA / "train/train_source3.tsv")])
    ind = cands.filter(pl.col("business_name").str.contains(r"[ऀ-ൿ]")
                       | pl.col("business_address").str.contains(r"[ऀ-ൿ]"))
    gt = read_tsv(DATA / "train/train_ground_truth.tsv")
    gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .rename({"matched_entity_ids": "cid"}))
    pairs = (ind.join(gt, left_on="entity_id", right_on="cid")
               .join(s1, left_on="source1_entity_id", right_on="entity_id", suffix="_1"))
    print("indic pairs", pairs.height, file=sys.stderr)

    votes = defaultdict(Counter)
    for field in ("business_name", "business_address"):
        for a, b in zip(pairs[field].to_list(), pairs[field + "_1"].to_list()):
            if not a or not b or not has_indic(a):
                continue
            ta, tb = toks(a), toks(b)
            if not tb:
                continue
            for t in ta:
                if t.isdigit() or len(t) < 2:
                    continue
                best, bs = None, 0.0
                for u in tb:
                    sc = fuzz.ratio(t, u)
                    if sc > bs:
                        best, bs = u, sc
                if bs >= 55:
                    votes[t][best] += 1
                else:
                    votes[t]["<none>"] += 1
    mapping = {}
    for t, c in votes.items():
        u, n = c.most_common(1)[0]
        tot = sum(c.values())
        if u != "<none>" and u != t and n >= 3 and n / tot >= 0.6:
            mapping[t] = u
    print("learned", len(mapping), "token mappings", file=sys.stderr)
    WORK.mkdir(exist_ok=True)
    with open(WORK / "translit_dict.json", "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=0, sort_keys=True)
    for k in list(mapping)[:40]:
        print(k, "->", mapping[k], file=sys.stderr)


def learn_vocab():
    """Unigram costs of name tokens in Source 1 (clean reference) for word segmentation."""
    import math
    import normalize as nz
    cnt = Counter()
    for split in ("train", "test"):
        names = read_tsv(DATA / f"{split}/{split}_source1.tsv")["business_name"].to_list()
        for n in names:
            toks, _ = nz.name_tokens(n)
            cnt.update(t for t in toks if t.isalpha() and len(t) >= 2)
    for w in list(nz.LEGAL) + [v for x in nz.NAME_CANON.values() for v in x.split()]:
        if len(w) >= 2:
            cnt[w] += 50
    cnt = {w: c for w, c in cnt.items() if c >= 3}
    tot = sum(cnt.values())
    vocab = {w: round(-math.log(c / tot), 3) for w, c in cnt.items()}
    with open(WORK / "name_vocab.json", "w") as f:
        json.dump(vocab, f)
    print("vocab", len(vocab), file=sys.stderr)


if __name__ == "__main__":
    main()
    import normalize as nz
    nz.set_translit_dict(json.load(open(WORK / "translit_dict.json")))
    learn_vocab()
