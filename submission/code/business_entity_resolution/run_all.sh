#!/usr/bin/env bash
# End-to-end: data -> normalisation -> blocking -> training -> inference -> outputs
# Usage: BER_ROOT=/path/to/student_resource bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PY:-python3}
$PY learn_dict.py          # transliteration dictionary + name vocabulary (train data only / S1 names)
$PY prep.py                # normalise all 6 source files -> work/*.parquet
$PY blocking.py train      # candidate generation (train)
$PY blocking.py test       # candidate generation (test)
$PY train.py               # stage-2 candidate filter + stage-3 matcher, validation F0.5
$PY predict.py test        # output/matching_results.tsv + output/candidate_pairs.tsv
