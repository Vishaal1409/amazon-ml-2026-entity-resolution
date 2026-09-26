# Business Entity Resolution — reproduction guide

Three-stage cascade: **blocking** (compound-key inverted index) → **candidate filter**
(LightGBM on vectorised features) → **matcher** (LightGBM on rich string features +
competition context) → one-to-one assignment + F0.5-tuned threshold.

Held-out validation (30% of train S1 entities, 662,834 records, never seen by either model):
**macro F0.5 = 0.9750**, pair precision 0.9954, pair recall 0.9394, **3.84 candidates per S1**.

## Environment
Python 3.13, CPU only (developed on a 10-core / 24 GB Mac). No external data, APIs or pretrained models.
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Layout expected
```
<ROOT>/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<ROOT>/dataset/test/test_source{1,2,3}.tsv
```
`ROOT` defaults to three levels above `src/` (i.e. the folder containing `code/`); override with
`BER_ROOT=/path`. Intermediate files go to `$BER_ROOT/work`, outputs to `$BER_ROOT/output`
(override with `BER_WORK`, `BER_OUT`; threads with `BER_JOBS`).

## Run end-to-end (~45 min)
```bash
BER_ROOT=/path/to/root bash run_all.sh
```
or step by step from `src/`:

| step | script | what it does | time |
|---|---|---|---|
| 1 | `learn_dict.py` | learns Indic-transliteration token dictionary from train matches + name vocabulary for word segmentation | ~1 min |
| 2 | `prep.py` | normalises all 6 source files (transliteration, accents, homoglyphs, legal suffixes, address abbreviations) → parquet | ~2 min |
| 3 | `blocking.py train` / `blocking.py test` | candidate generation, scored pairs | ~4 min each |
| 4 | `train.py` | trains stage-2 filter (fold A) and stage-3 matcher (fold B), validates on fold C, picks thresholds → `work/model_config.json` | ~25 min |
| 5 | `predict.py test` | writes `output/candidate_pairs.tsv` and `output/matching_results.tsv` | ~10 min |

Validate: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Source files
- `normalize.py` – text normalisation, rule-based Brahmic→Latin transliteration, phonetic skeleton, word segmentation
- `learn_dict.py` – learned transliteration dictionary + vocabulary
- `prep.py` – parallel normalisation
- `blocking.py` – country-sharded inverted index, single + compound keys, prefix-filtered probing, scoring/ranks
- `cheap_features.py` – vectorised pair features for the candidate filter
- `features.py` – rich rapidfuzz features for the matcher
- `pipeline.py` – cascade glue, one-to-one decision rule, F0.5 metric
- `train.py`, `predict.py` – training/validation and inference
