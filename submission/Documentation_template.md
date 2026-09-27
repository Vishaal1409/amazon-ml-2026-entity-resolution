# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** something
**Team Members:** Balaa Ts, Abhinava Krishna R, Vishaal S  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
A three-stage, fully CPU-based cascade: (1) a country-sharded inverted-index blocker whose
*compound keys* (name-token × address-word, number × address-word, name-token × number) are
selective yet noise-robust; (2) a LightGBM candidate filter on vectorised features that shrinks
the candidate set to **3.84 records per Source-1 entity** (true average is 3.46) while keeping
99.5% of reachable matches; (3) a LightGBM matcher on rich string-similarity and "competition"
features, followed by one-to-one assignment and an F0.5-optimised threshold.
A second, *collective* pass uses each entity's confident matches as extra queries to recover records the S1 record alone cannot reach.
Hold-out validation (265,078 S1 entities unseen by every model): **macro F0.5 = 0.9819, precision 0.9968, recall 0.9562**.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: 2.2M S1, 5.0M S2, 5.3M S3; test: 1.73M S1, 4.9M S2, 5.1M S3 (adds France, unseen in train).
- Every S2/S3 record matches **at most one** S1 entity (7.64M matched ids, all unique); ~27% of
  S2/S3 records are distractors. 5.6% of S1 entities are singletons; mean 3.46 matches/S1.
- Country labels always agree between matched records → safe partition key (open set of labels).
- Distractors are mostly *same/similar name at a different address* (e.g. "Helios Ltd" in OR vs CT),
  so address agreement is the main precision signal.
- Name noise: ~9.5% of names in 9 Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali,
  Gujarati, Malayalam, Gurmukhi, Oriya); homoglyphs (`8ig`, `Sta1der`); accents; token
  reordering; legal-suffix changes; glued/domain names (`colonialfoods.com`, `megaadvisors`);
  prefixes/suffix noise (`--`, `***`, `Dr`, `Services`); DBA/aka names.
- Address noise: component reordering, abbreviations, state name ↔ code, Indic-script state names,
  truncation to "number + city", missing addresses (~3.4% empty), city aliases (Bombay/Mumbai).

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned candidate filter + gradient-boosted matcher (cascade)
**Core Innovation:** compound-key prefix-filtered blocking + a learned filter stage that yields a
candidate set barely larger than the true match set; mutual/one-to-one constraints exploiting the
"each S2/S3 record belongs to ≤1 S1" structure.

Normalisation (`normalize.py`, language-agnostic):
- Rule-based transliteration of all Brahmic scripts: every script block is folded onto the
  Devanagari layout (shared ISCII offsets) and transliterated with inherent-vowel / virama /
  schwa-deletion handling.
- A **learned transliteration dictionary** (1,730 entries, e.g. `praivet→private`,
  `entarpraijes→enterprises`, `hariyana→haryana`) obtained by aligning transliterated tokens with
  the most similar token of the matched S1 record in *training* pairs.
- Unicode NFKC + accent folding, homoglyph repair in mixed alpha-numeric tokens, merging of dotted
  initials (`E.U.R.L.`→`eurl`), canonical legal forms (Pvt→private, Ltd→limited, SARL, SAS, …),
  split into *core* tokens vs legal/filler tokens.
- Word segmentation of glued tokens with a unigram vocabulary built from S1 names (DP min-cost);
  only clean splits (≤4 pieces, each ≥3 letters) are accepted and transliterated text is never
  segmented (error analysis showed segmentation was fragmenting transliterations such as `elaelapi`).
- Address: US/Indian state names→codes, abbreviation canonicalisation (street→st, rue→r, …),
  number extraction with leading-zero stripping, a few Indian city aliases.

---

## 3. Candidate Generation (Blocking)
- **Sharding:** by the `country` string (open set; France forms its own shard automatically).
- **Blocking keys used (64-bit hashed):**
  single keys – core name tokens, phonetic skeletons of name tokens (robust to vowel noise and
  transliteration), concatenated core name, address words, address numbers;
  compound keys – rarest 3 name tokens × rarest 4 address words, rarest 3 numbers × rarest 4
  address words, name tokens × numbers.
- **Prefix filtering:** each S1 record probes only its 16 rarest keys with document frequency ≤ 500,
  so work per record is bounded (≤ 8,000 postings) independent of corpus size → linear scaling.
  Probed pairs are ranked by summed IDF (top 60), then re-scored with an IDF-weighted cosine
  (overall, name-only and number-only), plus ranks in both directions and runner-up margins.
- **Learned candidate filter (final candidate stage):** LightGBM (600 trees) over 38 vectorised
  features (blocking scores, both-direction ranks, margins, token/skeleton/address/number
  Jaccard and overlap, house-number and legal-form agreement). Pairs with p ≥ 0.12 and within
  the top 8 per S1 form `candidate_pairs.tsv` — exactly the set the matcher scores.
- **Candidate pairs generated (test):** see §5 (≈3.8 per S1 entity).
- **How true matches were not lost:** compound keys recover matches whose individual tokens are
  common; phonetic skeletons and learned transliteration handle script/typo variants; the filter
  threshold is set to keep 99.5% of reachable positives. Recall ceiling of the final candidate
  set on hold-out: **95.6%** of all true pairs (many residual misses have an empty address and a
  generic name).
- Reduction ratio: 3.84 candidates vs ~10M possible → > 99.9999%.

---

- **Name-evidence probing (final version):** the rarest 5 name keys are always probed, 10 candidate
  slots per S1 are reserved for the best name-only probe score, and name-token-pair compound keys are
  added. Before this, records with long addresses spent their whole probe budget on address keys, so
  candidates with an empty address were unreachable (recall on them 57% → 75%; overall blocking
  recall 96.4% → 97.2% at the same pairs per S1; perfect-matcher bound 0.988 → 0.991).

### 3.1 Pass 2 — collective candidate expansion
Error analysis on the hold-out showed 4.4% of true pairs were lost at blocking and that 95% of those
belong to an S1 entity that already had at least one confident match. Every S2/S3 record belongs to a
single entity, so a confident match (pass-1 p ≥ 0.9, one-to-one) is another view of the same business
with its own name/address variant. Each confident match is used as an additional query against the same
blocking index; for each S1 the top-2 newly found records with sibling cosine ≥ 0.5 are added.
The final `candidate_pairs.tsv` is this pass-2 universe (pass-1 candidates ∪ sibling-found records),
i.e. exactly the set scored by the final model: **4.72 candidates per S1**, candidate recall 96.4%.

## 4. Matching Model

**Features used (88):**
- Name: rapidfuzz ratio / token-sort / token-set / partial ratio on full and core names,
  Jaro-Winkler, Levenshtein, concatenated-name ratio/equality (handles glued names), token and
  phonetic-skeleton Jaccard/overlap, fuzzy token coverage both ways, first-token match, lengths,
  legal-form presence / equality / conflict.
- Address: ratio / token-set / token-sort / partial ratio, word Jaccard/overlap, number
  Jaccard/overlap, house-number equality, postcode (5–6 digit) equality and conflict, empty flags.
- Other: all blocking scores and ranks, stage-2 probability, source (S2 vs S3), Indic-script flag,
  and **competition context** over the candidate set: rank of the pair among the S1's candidates
  and among the candidate's S1s, gap to the best/second-best competing S1, candidate counts.
  Country is deliberately *not* a feature (France generalisation).

**Model type:** LightGBM binary classifier (255 leaves, lr 0.05, early stopping), trained on the
stage-2 survivors of a disjoint fold of S1 entities (no leakage between stages).
**Decision rule:** each S2/S3 record is assigned only to its highest-scoring S1 (one-to-one),
then kept if p ≥ τ.
**Threshold selection method:** τ chosen by maximising macro F0.5 (singletons included) on hold-out data.

**Pass-2 model:** LightGBM over pass-1 probabilities (missing for new pairs), rich similarity to the S1
record, rich similarity to the best sibling that found the record, sibling-probe scores and hit counts,
number of confident matches of the S1 (overall and in the same source), and competition features
(is the record a confident match of another S1, best pass-1 probability of the record elsewhere).
Pass-2 is trained on 60% of fold C (whose pass-1 scores are out-of-sample) and validated on the other 40%.
Final decision: one-to-one assignment, threshold 0.70.

Data split: S1 entities hashed into 10 folds — A (20%) trains the filter, B (50%) trains the
matcher, C (30%) is the untouched validation set for the whole cascade.

---

### 4.1 Generalisation to France (unseen country)
Leaderboard analysis (US 0.982 / India 0.972 on validation vs 0.966 overall) implied France ≈ 0.90.
Inspecting uncertain French test pairs showed distractors on the *same street* with nearby house
numbers whose names differ by a single generic word ("CV Amis SARL" vs "CV Sportive SARL"), acronym
aliases ("Calais Garage SAS" vs "CG"), and region/department names diluting address similarity.
Country-agnostic features added: IDF-weighted cost of name words unmatched on either side, acronym
match, house-number distance, street-level address similarity after removing each country's 400 most
frequent address words (learned from the unlabelled files), extended legal forms (EI, EIRL, SCM, GIE…).
Effect: validation +0.002; French predicted matches/S1 3.43 → 3.17 and uncertain French pairs 4.9% → 2.5%.
Finally the pass-2 model is adapted to unseen countries by self-training: confident French test
pairs (p ≥ 0.97 positives, ≤ 0.03 negatives, entities with any uncertain pair excluded; 721k rows)
are added with weight 0.5; the adapted model is accepted only if US/India hold-out F0.5 does not drop
(0.97999 → 0.97999). It changed 2.7% of French rows (uncertain share 2.5% → 1.9%).

## 5. Results & Error Analysis

| Stage (hold-out) | F0.5 | Precision | Recall | Cands/S1 |
|---|---|---|---|---|
| Pass 1 (initial) | 0.9750 | 0.9954 | 0.9394 | 3.84 |
| Pass 1 + segmentation fix | 0.9761 | 0.9958 | 0.9414 | 3.83 |
| Pass 2 collective | 0.9781 | 0.9961 | 0.9485 | 4.72 |
| + France-oriented generic features | 0.9800 | 0.9972 | 0.9518 | 4.72 |
| **+ name-evidence blocking (final)** | **0.9819** | **0.9968** | **0.9562** | **~5** |

- Upper bound with a perfect matcher on the final candidate set ≈ 0.988: the remaining gap is
  dominated by blocking losses.
- Loss breakdown (share of all true pairs, pass 1): empty address + generic name 1.9% (practically
  unresolvable), empty address with unique name 0.7%, completely different trade name 0.75%,
  Indic-script records 0.8%, other noise 1.4%.
- Test run (final): 1,732,544 S1 entities; 9,728,804 candidate pairs (5.62 per S1); 5,711,907 predicted matches (3.30 per S1). Validator (with --check-ids): PASS.
- **Common false positives:** same generic name ("Global Technologies Pvt Ltd") where the
  candidate address is truncated to "door no + city" and matches another entity in that city.
- **Common false negatives:** candidate records with empty address and generic names; DBA /
  completely different trade names; heavily truncated Indian addresses.

---

## 6. Conclusion
Careful normalisation (transliteration learned from data, segmentation, homoglyphs) plus
compound-key blocking gives a high-recall, linearly scalable candidate generator; a learned filter
shrinks candidates to near the true-match count, and a context-aware GBDT with one-to-one
assignment yields 0.982 macro F0.5 on unseen entities. All models are LightGBM (MIT licence),
far below the 8B parameter limit; no external data or services are used.
