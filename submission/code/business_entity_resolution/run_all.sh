#!/usr/bin/env bash
# End-to-end: data -> normalisation -> blocking -> training -> inference -> outputs
# Usage: BER_ROOT=/path/to/student_resource bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PY:-python3}
$PY learn_dict.py          # transliteration dictionary + name vocabulary (train data only / S1 names)
$PY prep.py                # normalise all 6 source files -> work/*.parquet
$PY aux_tables.py          # frequent address words per country (street-level features)
$PY blocking.py train      # candidate generation (train)
$PY blocking.py test       # candidate generation (test)
$PY train.py               # stage-2 candidate filter + stage-3 matcher, validation F0.5
$PY predict.py test        # pass-1 predictions (also writes work/test_pass1.parquet)
$PY collective.py train    # pass-2 collective model (sibling evidence), validation F0.5
$PY collective.py predict test   # output/matching_results.tsv + output/candidate_pairs.tsv
$PY selftrain.py           # self-training for countries unseen in training (France); rewrites their rows
