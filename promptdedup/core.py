"""Near-duplicate prompt detection with MinHash and LSH.

A prompt library grows by copy-paste. Forty variants of "summarise this email"
differ in one sentence, get maintained separately, and drift. Finding them by
eye does not happen; finding them by exact match finds nothing.

MinHash estimates Jaccard similarity from a fixed-size signature, and LSH turns
that into candidate pairs without comparing every prompt to every other. The
hash is BLAKE2b, not ``hash()``: Python randomises string hashing per process,
which would make the buckets different on every run and the tool useless.

>>> docs = ["Summarise the email in one sentence.",
...         "Summarise the email in one sentence!",
...         "Return the current stock price as JSON."]
>>> clusters(docs, threshold=0.6)
[[0, 1]]
>>> round(jaccard(shingles(docs[0]), shingles(docs[1])), 2) > 0.5
True
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "normalise",
    "shingles",
    "jaccard",
    "MinHasher",
    "Signature",
    "optimal_bands",
    "lsh_candidates",
    "clusters",
    "compare_documents",
]

_WS = re.compile(r"\s+")
# Values that vary between otherwise identical prompts: ids, numbers, quoted
# examples. Masking them is what makes the near-duplicate visible.
_VOLATILE = re.compile(r"\b\d+(?:\.\d+)?\b|'[^']{0,40}'|\"[^\"]{0,40}\"")
# Trailing punctuation differs between "in one sentence." and "in one sentence!"
# while the prompt is the same prompt.
_EDGE_PUNCT = re.compile(r"^[^\w<>]+|[^\w<>]+$")


def normalise(text: str, mask_values: bool = True) -> str:
    """Lowercase, collapse whitespace, strip edge punctuation, mask literals.

    >>> normalise("Hello   WORLD")
    'hello world'
    >>> normalise("Order 1234 for Bob", mask_values=True)
    'order <n> for bob'
    >>> normalise("Summarise this!")
    'summarise this'
    """
    text = text.strip().lower()
    if mask_values:
        text = _VOLATILE.sub("<n>", text)
    words = [_EDGE_PUNCT.sub("", word) for word in _WS.split(text)]
    text = " ".join(word for word in words if word)
    return _WS.sub(" ", text)


def shingles(text: str, k: int = 3) -> Set[str]:
    """Word k-shingles of normalised text.

    Short texts fall back to the whole normalised string so two one-word
    prompts are still comparable.

    >>> sorted(shingles("a b c d", k=2))
    ['a b', 'b c', 'c d']
    """
    if k < 1:
        raise ValueError("k must be positive")
    words = normalise(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(left: Set[str], right: Set[str]) -> float:
    """Exact Jaccard similarity.

    >>> jaccard({'a', 'b'}, {'a', 'b'}) == 1.0
    True
    >>> jaccard({'a'}, {'b'})
    0.0
    """
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


class MinHasher:
    """MinHash signatures from a fixed permutation family."""

    _MERSENNE = (1 << 61) - 1

    def __init__(self, num_perm: int = 128, seed: int = 0):
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        self.seed = seed
        self._a: List[int] = []
        self._b: List[int] = []
        state = int(hashlib.blake2b(str(seed).encode(), digest_size=8).hexdigest(), 16)
        for index in range(num_perm):
            state = (6364136223846793005 * state + 1442695040888963407) % (1 << 64)
            self._a.append((state % (self._MERSENNE - 1)) + 1)
            state = (6364136223846793005 * state + 1442695040888963407) % (1 << 64)
            self._b.append(state % self._MERSENNE)

    def signature(self, shingle_set: Iterable[str]) -> Tuple[int, ...]:
        """Return a ``num_perm``-length signature.

        >>> hasher = MinHasher(num_perm=16, seed=1)
        >>> len(hasher.signature({"a b", "b c"}))
        16
        """
        values = [_stable_hash(item) for item in shingle_set]
        if not values:
            return tuple([self._MERSENNE] * self.num_perm)
        signature = []
        for a, b in zip(self._a, self._b):
            signature.append(min((a * value + b) % self._MERSENNE for value in values))
        return tuple(signature)


Signature = Tuple[int, ...]


def estimate_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    """Estimate similarity as the fraction of matching signature positions.

    >>> hasher = MinHasher(num_perm=64)
    >>> a = hasher.signature(shingles("one two three four"))
    >>> estimate_jaccard(a, a)
    1.0
    """
    if len(left) != len(right):
        raise ValueError("signatures must be the same length")
    if not left:
        return 1.0
    return sum(1 for x, y in zip(left, right) if x == y) / len(left)


def _band_keys(signature: Sequence[int], bands: int, rows: int) -> List[Tuple[int, str]]:
    keys = []
    for band in range(bands):
        chunk = signature[band * rows:(band + 1) * rows]
        digest = hashlib.blake2b(repr(chunk).encode(), digest_size=8).hexdigest()
        keys.append((band, digest))
    return keys


def optimal_bands(threshold: float, num_perm: int = 128) -> Tuple[int, int]:
    """Choose the (bands, rows) that puts the LSH S-curve steepest at ``threshold``.

    >>> bands, rows = optimal_bands(0.8, 128)
    >>> bands * rows <= 128
    True
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    best = (1, num_perm)
    best_score = float("inf")
    for bands in range(1, num_perm + 1):
        rows = num_perm // bands
        if rows < 1:
            continue
        # Probability a pair at the threshold shares at least one band.
        probability = 1 - (1 - threshold ** rows) ** bands
        score = abs(probability - 0.5)
        if score < best_score:
            best_score = score
            best = (bands, rows)
    return best


@dataclass
class DuplicatePair:
    """Two documents that look like the same prompt."""

    left: int
    right: int
    similarity: float
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {"left": self.left, "right": self.right,
                "similarity": round(self.similarity, 4), "reason": self.reason}


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def lsh_candidates(signatures: Sequence[Signature], bands: int, rows: int) -> List[Tuple[int, int]]:
    """Candidate index pairs that share at least one LSH band."""
    buckets: Dict[Tuple[int, str], List[int]] = {}
    for index, signature in enumerate(signatures):
        for key in _band_keys(signature, bands, rows):
            buckets.setdefault(key, []).append(index)
    pairs: Set[Tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                pairs.add((left, right) if left < right else (right, left))
    return sorted(pairs)


def compare_documents(left: str, right: str, k: int = 3, mask_values: bool = True) -> Dict[str, float]:
    """Exact similarity between two prompts, for the pair you already suspect.

    >>> compare_documents("summarise this", "summarise this")["jaccard"]
    1.0
    """
    left_set = shingles(normalise(left, mask_values), k)
    right_set = shingles(normalise(right, mask_values), k)
    exact = jaccard(left_set, right_set)
    hasher = MinHasher(num_perm=64)
    estimated = estimate_jaccard(hasher.signature(left_set), hasher.signature(right_set))
    return {"jaccard": exact, "minhash": estimated,
            "shared_shingles": float(len(left_set & right_set))}


def clusters(documents: Sequence[str], threshold: float = 0.8, k: int = 3,
             num_perm: int = 128, mask_values: bool = True,
             bands: Optional[int] = None) -> List[List[int]]:
    """Group near-duplicate documents, verified with exact Jaccard.

    LSH proposes candidates; the exact score decides. A candidate pair below
    the threshold is discarded, so the output does not contain the LSH
    false positives that make naive MinHash tools noisy.

    >>> clusters(["a b c d e", "a b c d e", "x y z"], threshold=0.9)
    [[0, 1]]
    """
    if not documents:
        return []
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if bands is None:
        bands, _ = optimal_bands(threshold, num_perm)
    rows = max(1, num_perm // bands)
    normalised = [normalise(doc, mask_values) for doc in documents]
    shingle_sets = [shingles(text, k) for text in normalised]
    hasher = MinHasher(num_perm=num_perm)
    signatures = [hasher.signature(items) for items in shingle_sets]

    union = _UnionFind(len(documents))
    for left, right in lsh_candidates(signatures, bands, rows):
        if jaccard(shingle_sets[left], shingle_sets[right]) >= threshold:
            union.union(left, right)

    groups: Dict[int, List[int]] = {}
    for index in range(len(documents)):
        groups.setdefault(union.find(index), []).append(index)
    return sorted((sorted(group) for group in groups.values() if len(group) > 1),
                  key=lambda group: group[0])


def find_duplicates(documents: Sequence[str], threshold: float = 0.8, k: int = 3,
                    num_perm: int = 128, mask_values: bool = True) -> List[DuplicatePair]:
    """All candidate pairs at or above ``threshold``, with exact scores."""
    shingle_sets = [shingles(normalise(doc, mask_values), k) for doc in documents]
    hasher = MinHasher(num_perm=num_perm)
    signatures = [hasher.signature(items) for items in shingle_sets]
    bands, _ = optimal_bands(threshold, num_perm)
    rows = max(1, num_perm // bands)
    pairs = []
    for left, right in lsh_candidates(signatures, bands, rows):
        score = jaccard(shingle_sets[left], shingle_sets[right])
        if score >= threshold:
            pairs.append(DuplicatePair(left, right, score, "lsh+verify"))
    pairs.sort(key=lambda pair: (-pair.similarity, pair.left, pair.right))
    return pairs
