"""
normalize.py — Business Entity Resolution Challenge

Cleans up business_name / business_address text BEFORE any similarity scoring
or blocking. Handles the noise patterns confirmed by EDA on the sample data:

  - text in 8 different Indian scripts (Devanagari, Tamil, Kannada, Telugu,
    Gujarati, Bengali, Gurmukhi, Malayalam) mixed into otherwise-Latin fields
  - French accented characters
  - garbage lead-in tokens ("--", "<<")
  - the literal word "null" appearing INSIDE the text (not just a missing value)
  - legal suffixes wrapped in brackets/parens: "[Limited]", "(Limited)"
  - legal suffixes as a PREFIX as often as a suffix: "LLC Foo", "Foo LLC"
  - dba clauses and appended websites: "X dba Y", "... | www.site.com"

This is pure offline text normalization (unidecode is a static Unicode-to-ASCII
transliteration table, not an external lookup service), so it does not touch
the "no external data/API lookup" rule.

Usage:
    from normalize import normalize_name, normalize_address

    clean_name, name_tokens = normalize_name(raw_name)
    clean_address = normalize_address(raw_address)
"""

import re
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Config: stopword-style legal suffix/entity-type tokens (after normalization,
# lowercase). Used both to unwrap bracket/paren'd suffixes and to exclude
# from blocking-key tokens (they're too common to be a useful signal).
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "ltd",
    "limited", "pvt", "private", "co", "company", "sarl", "sas", "sasu",
    "sa", "plc", "group", "holdings", "enterprises", "enterprise",
    "services", "solutions", "consultants", "consultancy", "ventures",
    "trust", "society", "association", "foundation", "public",
    "partners", "partnership", "associates", "agency", "agencies",
    "industries", "international", "global", "national",
}

LEADIN_RE = re.compile(r"^\s*(--+|<<+|>>+|\*+)\s*")
NULL_TOKEN_RE = re.compile(r"(?i)\bnull\b")
MULTISPACE_RE = re.compile(r"\s+")
STRAY_PUNCT_EDGE_RE = re.compile(r"^[\s,._\-|]+|[\s,._\-|]+$")
REPEATED_DELIM_RE = re.compile(r"\s*,\s*(,\s*)+")
BRACKETED_SUFFIX_RE = re.compile(r"[\[\(]\s*([A-Za-z .]+?)\s*[\]\)]")
DBA_SPLIT_RE = re.compile(r"(?i)\bd/?b/?a\b")
WEBSITE_RE = re.compile(
    r"(?i)(https?://\S+|www\.\S+|\b[a-z0-9\-]+\.(com|net|org|in|co)\b)"
)
BARE_DOMAIN_RE = re.compile(r"(?i)^([a-z0-9\-]+)\.(com|net|org|in|co)$")
PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")


def detect_scripts(text):
    """Return the set of non-Latin Unicode block names present in `text`.

    Useful for tagging records so error analysis can check specifically how
    non-Latin-script entities are scoring (per the plan's Day-3 task).
    """
    scripts = set()
    if not isinstance(text, str):
        return scripts
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            scripts.add("DEVANAGARI")
        elif 0x0980 <= cp <= 0x09FF:
            scripts.add("BENGALI")
        elif 0x0A00 <= cp <= 0x0A7F:
            scripts.add("GURMUKHI")
        elif 0x0A80 <= cp <= 0x0AFF:
            scripts.add("GUJARATI")
        elif 0x0B00 <= cp <= 0x0B7F:
            scripts.add("ORIYA")
        elif 0x0B80 <= cp <= 0x0BFF:
            scripts.add("TAMIL")
        elif 0x0C00 <= cp <= 0x0C7F:
            scripts.add("TELUGU")
        elif 0x0C80 <= cp <= 0x0CFF:
            scripts.add("KANNADA")
        elif 0x0D00 <= cp <= 0x0D7F:
            scripts.add("MALAYALAM")
    return scripts


def _strip_leadin(text):
    return LEADIN_RE.sub("", text)


def _strip_null_token(text):
    text = NULL_TOKEN_RE.sub(" ", text)
    return text


def _unwrap_bracketed_suffix(text):
    """Turn "Foo [Limited]" / "Foo (Ltd)" into "Foo Limited" / "Foo Ltd".

    Only unwraps when the bracketed content looks like a short legal-suffix
    phrase (<=3 words), so we don't accidentally unwrap an unrelated
    parenthetical note elsewhere in the name.
    """
    def _repl(m):
        inner = m.group(1).strip()
        if 0 < len(inner.split()) <= 3:
            return " " + inner
        return m.group(0)
    return BRACKETED_SUFFIX_RE.sub(_repl, text)


def _split_dba_and_website(text):
    """Return (primary_name, extra_fragments) after removing dba/website noise.

    `extra_fragments` (dba alias, website) are returned separately in case
    you want to also index them for blocking recall, but the primary name is
    what should be used for the main normalized field.
    """
    extras = []

    # Split on a literal pipe first ("NAME | www.site.com")
    parts = PIPE_SPLIT_RE.split(text)
    primary = parts[0]
    extras.extend(p for p in parts[1:] if p.strip())

    # Pull out dba clauses: "X dba Y" -> primary "X", extra "Y"
    dba_parts = DBA_SPLIT_RE.split(primary)
    if len(dba_parts) > 1:
        primary = dba_parts[0]
        extras.extend(p for p in dba_parts[1:] if p.strip())

    # Strip any remaining website-looking fragments out of the primary name.
    # If the name IS just a bare domain ("wilfordhancock.com"), keep the
    # domain's own label instead of erasing the name entirely.
    bare = BARE_DOMAIN_RE.match(primary.strip())
    if bare:
        primary = bare.group(1).replace("-", " ")
    elif WEBSITE_RE.search(primary):
        primary = WEBSITE_RE.sub(" ", primary)

    return primary, extras


def _clean_common(text):
    if text is None:
        return ""
    text = str(text)
    if text.strip().lower() in ("", "nan", "none", "null"):
        return ""
    text = unidecode(text)          # transliterate any non-Latin script
    text = _strip_leadin(text)
    text = _strip_null_token(text)
    text = REPEATED_DELIM_RE.sub(", ", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = STRAY_PUNCT_EDGE_RE.sub("", text)
    return text.strip()


def normalize_name(raw_name):
    """Clean a business_name. Returns (clean_name, token_list).

    token_list excludes common legal-suffix words so it's ready to use
    directly as a blocking signal (see blocking.py).
    """
    scripts = detect_scripts(raw_name)
    text = _clean_common(raw_name)
    text = _unwrap_bracketed_suffix(text)
    primary, extras = _split_dba_and_website(text)
    primary = MULTISPACE_RE.sub(" ", primary).strip()

    lower = primary.lower()
    # normalize punctuation that separates words for tokenization
    lower_for_tokens = re.sub(r"[^a-z0-9 ]", " ", lower)
    tokens = [t for t in lower_for_tokens.split() if t and t not in LEGAL_SUFFIX_TOKENS]

    return {
        "clean_name": primary,
        "clean_name_lower": lower,
        "tokens": tokens,
        "extras": [MULTISPACE_RE.sub(" ", e).strip() for e in extras],
        "had_scripts": scripts,
    }


def normalize_address(raw_address):
    """Clean a business_address. Returns (clean_address, clean_address_lower)."""
    scripts = detect_scripts(raw_address)
    text = _clean_common(raw_address)
    lower = text.lower()
    return {
        "clean_address": text,
        "clean_address_lower": lower,
        "had_scripts": scripts,
    }


if __name__ == "__main__":
    # Quick self-test against a handful of the confirmed noise patterns.
    samples = [
        "-- Holloway Peak Inc Seafood",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com",
        "Ectolumdrex dba X+ Madison Inc",
        "wilfordhancock.com",
        "LLC Moncada Léarning Center",
        "Pvt. EFS Print Ventures Ltd.",
        "Clm Agro [Limited]",
        "<< Team Ecole",
        "ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட்",
    ]
    for s in samples:
        r = normalize_name(s)
        print(f"{s!r:60} -> {r['clean_name']!r:45} tokens={r['tokens']}")

    print()
    addr_samples = [
        "MA, 25 NIAGARA STREET, NULL, SPRINGFIELD",
        "South Bangalore Nursing, ಕರ್ನಾಟಕ",
        None,
        "Door No 543 139G/A, NULL, Mettupalayam, Coimbatore",
    ]
    for a in addr_samples:
        r = normalize_address(a)
        print(f"{a!r:55} -> {r['clean_address']!r}")
