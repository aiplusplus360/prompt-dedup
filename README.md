# prompt-dedup

**Find the forty prompts that are almost the same prompt — with MinHash for speed, exact Jaccard for the final call, and a stable hash so the answer does not change between runs.**

[![CI](https://github.com/aiplusplus360/prompt-dedup/actions/workflows/ci.yml/badge.svg)](https://github.com/aiplusplus360/prompt-dedup/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

```console
$ prompt-dedup scan prompts/ --threshold 0.8
6 document(s), 2 group(s) at threshold 0.80
  duplicate group:
    prompts/summarise-v1.txt
    prompts/summarise-v2.txt
  duplicate group:
    prompts/classify-ticket.txt
    prompts/classify-ticket-copy.txt
closest pairs:
  0.94  prompts/summarise-v1.txt  <->  prompts/summarise-v2.txt
```

---

## Why this exists

A prompt library grows by copy-paste. Two prompts that differ by one sentence get maintained separately, drift apart, and one of them quietly loses the instruction that mattered. Exact-match deduplication finds nothing; manual review does not scale past thirty files.

Two techniques, and one correctness detail that makes the difference between a tool and a demo.

**MinHash** estimates Jaccard similarity from a fixed-length signature: 128 integers stand in for the whole set of shingles, so millions of pairwise comparisons become cheap.

**LSH** buckets signatures so only plausible pairs are compared at all. The band/row split is chosen so the probability curve is steepest at your threshold.

**Stable hashing.** MinHash implementations that use Python's `hash()` are non-deterministic: `PYTHONHASHSEED` is randomised per process, so two runs on the same corpus produce different signatures, different buckets, and different clusters. This uses BLAKE2b, and the test suite runs the same comparison under three different hash seeds in subprocesses to prove it.

## Normalisation does the real work

Before shingling:

- lowercase, collapse whitespace;
- strip edge punctuation, so `"Summarise this."` and `"Summarise this!"` agree;
- mask numbers and quoted literals, so `"Order 1001 status?"` and `"Order 2002 status?"` are recognised as the same prompt template.

The last one is the difference between finding duplicates and finding exact copies. It is also a judgement call, so `--no-mask` turns it off.

## LSH proposes, exact Jaccard decides

A MinHash candidate is not a duplicate — it is a pair worth checking. Every candidate is verified with exact Jaccard on the shingle sets, and only pairs above the threshold are reported. Naive MinHash tools report candidates and drown you in false positives.

Clustering is union-find over verified pairs, so chains group ({A≈B, B≈C} yields one group of three) and singletons are excluded.

## Install and use

No dependencies. Python 3.9+.

```bash
pip install git+https://github.com/aiplusplus360/prompt-dedup
```

```bash
prompt-dedup scan prompts/ --threshold 0.8 --fail-on-dupes
prompt-dedup scan prompts/ --json
prompt-dedup pair a.prompt b.prompt --json
```

A `.json` file containing an array of strings is treated as one document per string, which is how most prompt libraries export.

Exit codes: `0` clean, `1` duplicates found (with `--fail-on-dupes`), `2` no readable input.

## Library use

```python
from promptdedup import clusters, find_duplicates, compare_documents

groups = clusters(documents, threshold=0.8)
for group in groups:
    print([names[i] for i in group])

print(compare_documents(a, b))   # {'jaccard': ..., 'minhash': ..., 'shared_shingles': ...}
```

## Tests

```bash
python -m unittest discover -s tests -v
```

34 core tests and 6 CLI tests: normalisation and masking, shingle edge cases, exact Jaccard, MinHash accuracy against exact similarity, band selection, candidate collision and non-collision, transitive clustering, singleton exclusion, pair ordering, JSON-array ingestion, and a subprocess test that the signature is identical under three different `PYTHONHASHSEED` values.
