import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from promptdedup.cli import main


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_scan_groups_duplicates(self):
        self.write("a.txt", "summarise the email in one sentence")
        self.write("b.txt", "summarise the email in one sentence!")
        code, out, _ = run(["scan", str(self.root), "--threshold", "0.6", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload["groups"]), 1)
        self.assertEqual(len(payload["groups"][0]["files"]), 2)

    def test_scan_fail_on_dupes(self):
        self.write("a.txt", "one two three four five")
        self.write("b.txt", "one two three four five")
        code, _, _ = run(["scan", str(self.root), "--threshold", "0.9", "--fail-on-dupes"])
        self.assertEqual(code, 1)

    def test_scan_clean_exits_zero(self):
        self.write("a.txt", "alpha beta gamma delta")
        self.write("b.txt", "one two three four")
        code, _, _ = run(["scan", str(self.root), "--fail-on-dupes"])
        self.assertEqual(code, 0)

    def test_json_array_file_becomes_documents(self):
        path = self.write("prompts.json", json.dumps([
            "classify this ticket", "classify this ticket", "summarise this email",
        ]))
        code, out, _ = run(["scan", path, "--threshold", "0.9", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["documents"], 3)

    def test_pair(self):
        left = self.write("l.txt", "the quick brown fox")
        right = self.write("r.txt", "the quick brown fox")
        code, out, _ = run(["pair", left, right, "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["jaccard"], 1.0)

    def test_empty_path_is_error(self):
        empty = self.root / "empty"
        empty.mkdir()
        code, _, err = run(["scan", str(empty)])
        self.assertEqual(code, 2)
        self.assertIn("no documents", err)


if __name__ == "__main__":
    unittest.main()
