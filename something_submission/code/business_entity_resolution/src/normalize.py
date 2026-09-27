"""Text normalisation for business names and addresses.

Everything here is language-agnostic string processing (no external lookups):
  * rule-based transliteration of the 9 Brahmic scripts to Latin
  * accent folding, homoglyph repair (8ig -> big, Sta1der -> stalder)
  * canonicalisation of legal suffixes and address abbreviations
  * a phonetic "skeleton" used to compare transliteration variants
"""
import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------
# Indic -> Latin transliteration
# All Brahmic Unicode blocks share the ISCII layout, so we fold every block
# onto Devanagari (offset within the 128-char block) and transliterate once.
# --------------------------------------------------------------------------
_INDIC_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
           0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d",
         0x5D: "dh", 0x5E: "f", 0x5F: "y"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x62: "li", 0x63: "li"}
_VIRAMA = 0x4D
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}


def _indic_offset(ch):
    cp = ord(ch)
    if 0x0900 <= cp < 0x0D80:
        return cp & 0x7F
    return None


def has_indic(s):
    return any(0x0900 <= ord(c) < 0x0D80 for c in s)


def translit_indic(s):
    """Transliterate Brahmic-script text to Latin; non-Indic chars pass through."""
    out = []
    pending = False  # a consonant whose inherent 'a' is not yet emitted/suppressed
    for ch in s:
        off = _indic_offset(ch)
        if off is None:
            if pending:
                # schwa deletion at word end
                pending = False
            out.append(ch)
            continue
        if off in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[off])
            pending = True
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending = False
        elif off == _VIRAMA:
            pending = False
        elif off in _VOWELS:
            if pending:
                out.append("a")
                pending = False
            out.append(_VOWELS[off])
        elif off in _NASAL:
            if pending:
                out.append("a")
                pending = False
            out.append(_NASAL[off])
        elif 0x66 <= off <= 0x6F:
            if pending:
                pending = False
            out.append(str(off - 0x66))
        else:  # nukta, avagraha, stress marks ...
            continue
    return "".join(out)


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
_HOMO = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
                       "@": "a", "$": "s", "|": "l"})
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th|er|e|eme|bis|ter|[a-z])?$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|us|biz|info|co)\b")


TRANSLIT_DICT = {}


def set_translit_dict(d):
    TRANSLIT_DICT.clear()
    TRANSLIT_DICT.update(d)


def fold(s, use_dict=True):
    """NFKC + Indic transliteration (+ learned token dict) + accent folding + lowercase."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if has_indic(s):
        s = translit_indic(s).lower()
        if use_dict and TRANSLIT_DICT:
            s = re.sub(r"[a-z]+", lambda m: TRANSLIT_DICT.get(m.group(0), m.group(0)), s)
    s = unidecode(s)
    return s.lower()


def fix_homoglyphs(tok):
    """8ig -> big, sta1der -> stalder; leaves real numbers/ordinals alone."""
    if tok.isdigit() or _ORDINAL.match(tok):
        return tok
    if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
        n_alpha = sum(c.isalpha() for c in tok)
        if n_alpha >= len(tok) / 2:
            return tok.translate(_HOMO)
    return tok


def merge_initials(toks):
    """['e','u','r','l'] -> ['eurl'] (dotted acronyms)."""
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf))
    return out


_VOWEL_RE = re.compile(r"[aeiouy]")


def skeleton(tok):
    """Phonetic skeleton: robust to vowel noise, aspiration and transliteration."""
    if not tok:
        return ""
    if tok.isdigit():
        return tok
    t = tok.replace("ph", "f").replace("w", "v").replace("z", "j").replace("q", "k")
    t = t.replace("ck", "k").replace("c", "k").replace("x", "ks")
    head, rest = t[0], t[1:]
    rest = rest.replace("h", "")
    rest = _VOWEL_RE.sub("", rest)
    t = head + rest
    out = [t[0]]
    for c in t[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
NAME_CANON = {
    "pvt": "private", "pvtltd": "private limited", "prvt": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited", "limted": "limited",
    "corp": "corporation", "corpn": "corporation", "co": "company", "cos": "company",
    "inc": "incorporated", "incorp": "incorporated", "incorporation": "incorporated",
    "intl": "international", "int": "international", "mfg": "manufacturing",
    "assn": "association", "assoc": "association", "bros": "brothers", "svcs": "services",
    "svc": "services", "ent": "enterprises", "ents": "enterprises", "enterprise": "enterprises",
    "mgmt": "management", "tech": "technologies", "technology": "technologies",
    "dev": "development", "grp": "group", "ctr": "center", "centre": "center",
    "&": "and", "et": "and", "l.l.c": "llc", "st": "saint", "ste": "saint", "sainte": "saint",
    "elaelapi": "llp", "elelpi": "llp", "elalpi": "llp",
    "ets": "etablissements", "cie": "compagnie", "ste.": "societe", "soc": "societe",
}
# tokens that carry legal form / honorifics / filler rather than identity
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "pllc",
    "plc", "pc", "pa", "ltd", "gmbh", "ag", "nv", "bv", "sarl", "sas", "sasu", "eurl", "sa",
    "sci", "snc", "scs", "scop", "selarl", "and", "the", "of", "de", "du", "des", "la", "le",
    "les", "d", "l", "et", "a", "an", "aka", "dba", "fka", "ta", "m", "s", "ms", "mr", "mrs",
    "dr", "smt", "shri", "sri", "shree", "sree", "opc", "india", "france", "usa", "us",
    "com", "www", "net", "org", "co",
}
_ALIAS_SPLIT = re.compile(r"\b(?:aka|a/k/a|dba|d/b/a|fka|f/k/a|t/a|trading as|doing business as)\b")


# --------------------------------------------------------------------------
# Word segmentation for glued / domain-style names ("megaadvisors")
# --------------------------------------------------------------------------
VOCAB = {}  # word -> cost (-log p), learned from Source 1 names


def set_vocab(d):
    VOCAB.clear()
    VOCAB.update(d)


def segment(tok, max_word=20):
    """Min-cost split of an unknown glued token into known words; None if impossible."""
    n = len(tok)
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word), i - 1):  # words of length >= 2
            w = tok[j:i]
            c = VOCAB.get(w)
            if c is not None and best[j] + c < best[i]:
                best[i] = best[j] + c
                back[i] = j
    if best[n] == float("inf"):
        return None
    out, i = [], n
    while i > 0:
        out.append(tok[back[i]:i])
        i = back[i]
    return out[::-1] if len(out) > 1 else None


def _maybe_segment(toks):
    if not VOCAB:
        return toks
    out = []
    for t in toks:
        if len(t) >= 7 and t.isalpha() and t not in VOCAB:
            seg = segment(t)
            # only trust clean splits: short fragments are usually typos or transliteration debris
            if seg and len(seg) <= 4 and all(len(w) >= 3 for w in seg):
                out.extend(seg)
                continue
        out.append(t)
    return out


def name_tokens(raw):
    indic = has_indic(raw)
    s = fold(raw)
    m = _DOMAIN.search(s.strip())
    if m:
        s = m.group(1)
    s = s.replace("&", " and ").replace("'", "").replace("`", "")
    s = _ALIAS_SPLIT.sub(" | ", s)
    parts = [p for p in s.split("|")]
    all_toks = []
    alias_toks = []
    for i, p in enumerate(parts):
        toks = [fix_homoglyphs(t) for t in _NON_ALNUM.sub(" ", p).split()]
        toks = merge_initials(toks)
        if not indic:
            toks = _maybe_segment(toks)
        exp = []
        for t in toks:
            c = NAME_CANON.get(t)
            exp.extend(c.split() if c else [t])
        all_toks.extend(exp)
        alias_toks.append(exp)
    return all_toks, alias_toks


def name_features(raw):
    toks, aliases = name_tokens(raw)
    core = [t for t in toks if t not in LEGAL]
    legal = sorted(set(t for t in toks if t in LEGAL and t not in {"and", "the", "of", "a", "an"}))
    return {
        "n_norm": " ".join(toks),
        "n_core": " ".join(core),
        "n_legal": " ".join(legal),
    }


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------
ADDR_CANON = {
    "street": "st", "str": "st", "saint": "st", "sainte": "st", "ste": "st", "stree": "st",
    "road": "rd", "avenue": "ave", "av": "ave", "aven": "ave", "anenue": "ave", "drive": "dr",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "lane": "ln", "court": "ct",
    "place": "pl", "highway": "hwy", "parkway": "pkwy", "circle": "cir", "terrace": "ter",
    "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "trail": "trl", "way": "way",
    "mount": "mt", "fort": "ft", "point": "pt", "county": "cnty", "suite": "unit",
    "apartment": "unit", "apt": "unit", "ste.": "unit", "rue": "r", "allee": "all",
    "impasse": "imp", "chemin": "ch", "route": "rte", "nagar": "ngr", "colony": "col",
    "sector": "sec", "near": "nr", "opposite": "opp", "opp": "opp", "building": "bldg",
    "floor": "flr", "number": "no", "num": "no", "house": "h", "plot": "plot",
    "village": "vill", "vpo": "vill", "district": "dist", "dt": "dist", "tehsil": "teh",
    # common Indian city aliases
    "bombay": "mumbai", "poona": "pune", "madras": "chennai", "calcutta": "kolkata",
    "bengaluru": "bangalore", "gurugram": "gurgaon", "baroda": "vadodara", "cochin": "kochi",
    "mysuru": "mysore", "trivandrum": "thiruvananthapuram", "benares": "varanasi",
    "banaras": "varanasi", "allahabad": "prayagraj", "cawnpore": "kanpur", "belgaum": "belagavi",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
_STATE_RE = {
    c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b")
    for c, d in (("us", US_STATES), ("in", IN_STATES))
}
_NULLS = re.compile(r"<null>|\bnull\b|\bnone\b|\bn/?a\b")
_NUM = re.compile(r"\d+")


def addr_features(raw):
    s = fold(raw)
    s = _NULLS.sub(" ", s)
    s = s.replace("'", "").replace("&", " and ")
    # full state names -> codes (applied for both tables; country is an open set,
    # so we don't branch on it - codes rarely collide with real address words)
    s = _STATE_RE["us"].sub(lambda m: " " + US_STATES[m.group(1)] + " ", s)
    s = _STATE_RE["in"].sub(lambda m: " " + IN_STATES[m.group(1)] + " ", s)
    toks = [fix_homoglyphs(t) for t in _NON_ALNUM.sub(" ", s).split()]
    toks = [ADDR_CANON.get(t, t) for t in toks]
    nums = []
    for t in toks:
        for n in _NUM.findall(t):
            n2 = n.lstrip("0") or "0"
            nums.append(n2)
    words = [t for t in toks if not t.isdigit()]
    return {
        "a_norm": " ".join(toks),
        "a_words": " ".join(words),
        "a_nums": " ".join(nums),
    }


if __name__ == "__main__":
    for x in ["राम मार्केटिंग प्राइवेट लिमिटेड", "ராஜ் எனர்ஜி பிரைவேட் லிமிடெட்",
              "*** RASA PURE [PRÍVATE-LIMITED]", "8ig 64 Grill", "LLC Sta1der Ínterprivate",
              "colonialfoods.com", "Cirazeta aka Children's Program III", "TENNIS  GAVROCHES LYCEE E.U.R.L.",
              "Burgos & Smith Pimco LLC", "sárkdesigncom", "Hotel Innovative Infra Private (Limited)"]:
        print(x, "->", name_features(x))
    for x in ["2348- Gaebler Ave, # APT 4, Saint Louis, Missouri", "Monroville, Main Saint, Ohio",
              "Currie Road, <NULL>, Howrah, WB", "2209, Luxmi Nagar Village Israna, Panipat, हरियाणा",
              "27 AV DU PRESIDET JOHN FITZGERALD KENNEDY, PESSAC", "No 00409, D-1, Shreenathji Park"]:
        print(x, "->", addr_features(x))
    print([skeleton(t) for t in "praivet private limited limitad marketing enarji energy".split()])
