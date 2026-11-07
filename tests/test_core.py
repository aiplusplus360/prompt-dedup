import subprocess
import sys
import unittest

from promptdedup.core import (
    MinHasher,
    clusters,
    compare_documents,
    estimate_jaccard,
    find_duplicates,
    jaccard,
    lsh_candidates,
    normalise,
    optimal_bands,
    shingles,
)


class NormaliseTests(unittest.TestCase):
    def test_lowercase_and_collapse(self):
        self.assertEqual(normalise("Hello   WORLD"), "hello world")

    def test_masks_numbers(self):
        self.assertEqual(normalise("model 4 for user 8821"), "model <n> for user <n>")

    def test_masks_quoted_examples(self):
        self.assertEqual(normalise('say "hi" now'), "say <n> now")

    def test_masking_can_be_disabled(self):
        self.assertEqual(normalise("order 1234", mask_values=False), "order 1234")


class ShingleTests(unittest.TestCase):
    def test_k2(self):
        self.assertEqual(sorted(shingles("a b c d", k=2)), ["a b", "b c", "c d"])

    def test_short_text_is_one_shingle(self):
        self.assertEqual(shingles("only two", k=3), {"only two"})

    def test_empty(self):
        self.assertEqual(shingles("", k=3), set())

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            shingles("a", k=0)


class JaccardTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint(self):
        self.assertEqual(jaccard({"a"}, {"b"}), 0.0)

    def test_half(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_both_empty(self):
        self.assertEqual(jaccard(set(), set()), 1.0)


class MinHashTests(unittest.TestCase):
    def test_signature_length(self):
        hasher = MinHasher(num_perm=32)
        self.assertEqual(len(hasher.signature({"a", "b"})), 32)

    def test_estimate_identity(self):
        hasher = MinHasher(num_perm=128)
        sig = hasher.signature(shingles("one two three four five"))
        self.assertEqual(estimate_jaccard(sig, sig), 1.0)

    def test_estimate_tracks_exact(self):
        hasher = MinHasher(num_perm=256, seed=3)
        left = shingles("the quick brown fox jumps over the lazy dog")
        right = shingles("the quick brown fox leaps over the lazy dog")
        exact = jaccard(left, right)
        estimate = estimate_jaccard(hasher.signature(left), hasher.signature(right))
        self.assertLess(abs(exact - estimate), 0.15, (exact, estimate))

    def test_disjoint_is_zero(self):
        hasher = MinHasher(num_perm=64)
        left = hasher.signature({"a b c"})
        right = hasher.signature({"x y z"})
        self.assertEqual(estimate_jaccard(left, right), 0.0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            estimate_jaccard((1, 2), (1,))

    def test_empty_set_is_sentinel(self):
        hasher = MinHasher(num_perm=8)
        sig = hasher.signature(set())
        self.assertEqual(len(set(sig)), 1)

    def test_hash_is_stable_across_processes(self):
        # PYTHONHASHSEED changes hash() but must not change the signature.
        script = (
            "import sys;"
            "sys.path.insert(0, %r);"
            "from promptdedup.core import MinHasher, shingles;"
            "print(MinHasher(num_perm=16, seed=1).signature(shingles('a b c d e')))"
        )
        import os
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        outputs = set()
        for seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run([sys.executable, "-c", script % root],
                                    capture_output=True, text=True, env=env, check=True)
            outputs.add(result.stdout.strip())
        self.assertEqual(len(outputs), 1)


class BandTests(unittest.TestCase):
    def test_bands_times_rows_within_budget(self):
        for threshold in (0.3, 0.5, 0.7, 0.9):
            bands, rows = optimal_bands(threshold, 128)
            self.assertLessEqual(bands * rows, 128)

    def test_higher_threshold_uses_more_rows(self):
        _, low = optimal_bands(0.3, 128)
        _, high = optimal_bands(0.9, 128)
        self.assertLessEqual(low, high)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            optimal_bands(0.0)


class CandidateTests(unittest.TestCase):
    def test_identical_signatures_collide(self):
        hasher = MinHasher(num_perm=64)
        sig = hasher.signature(shingles("a b c d e f"))
        self.assertIn((0, 1), lsh_candidates([sig, sig], *optimal_bands(0.8, 64)))

    def test_different_signatures_do_not_collide(self):
        hasher = MinHasher(num_perm=64)
        left = hasher.signature(shingles("alpha beta gamma delta"))
        right = hasher.signature(shingles("one two three four"))
        self.assertEqual(lsh_candidates([left, right], *optimal_bands(0.8, 64)), [])


class ClusterTests(unittest.TestCase):
    def test_identical_prompts_group(self):
        self.assertEqual(clusters(["a b c d e", "a b c d e"], threshold=0.9), [[0, 1]])

    def test_unrelated_prompts_do_not(self):
        self.assertEqual(clusters(["a b c d e", "v w x y z"], threshold=0.6), [])

    def test_masking_makes_numeric_variants_match(self):
        docs = ["Order 1001 status?", "Order 2002 status?"]
        self.assertEqual(clusters(docs, threshold=0.9), [[0, 1]])

    def test_transitive_group(self):
        docs = ["a b c d e f g", "a b c d e f g h", "a b c d e f g h i"]
        groups = clusters(docs, threshold=0.5)
        self.assertEqual(groups, [[0, 1, 2]])

    def test_empty_input(self):
        self.assertEqual(clusters([]), [])

    def test_singletons_excluded(self):
        docs = ["one two three", "four five six", "seven eight nine"]
        self.assertEqual(clusters(docs, threshold=0.9), [])


class FindTests(unittest.TestCase):
    def test_pairs_sorted_by_similarity(self):
        docs = ["a b c d e", "a b c d e", "a b c d e f"]
        pairs = find_duplicates(docs, threshold=0.5)
        self.assertTrue(pairs)
        self.assertGreaterEqual(pairs[0].similarity, pairs[-1].similarity)

    def test_pair_as_dict(self):
        pair = find_duplicates(["a b c d e", "a b c d e"], threshold=0.9)[0]
        payload = pair.as_dict()
        self.assertEqual(set(payload), {"left", "right", "similarity", "reason"})


class CompareTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(compare_documents("summarise this", "summarise this")["jaccard"], 1.0)

    def test_counts_shared_shingles(self):
        result = compare_documents("a b c d", "a b c d")
        self.assertGreater(result["shared_shingles"], 0)


if __name__ == "__main__":
    unittest.main()
