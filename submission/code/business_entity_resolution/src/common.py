import os
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[3]))
DATA = Path(os.environ.get("BER_DATA", ROOT / "dataset"))
WORK = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUT = Path(os.environ.get("BER_OUT", ROOT / "output"))
N_JOBS = int(os.environ.get("BER_JOBS", os.cpu_count() or 4))


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       null_values=None).fill_null("")
