"""``prompt-dedup`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .core import clusters, compare_documents, find_duplicates

__version__ = "0.1.0"

SUFFIXES = {".txt", ".md", ".prompt", ".tmpl", ".jinja", ".yaml", ".yml", ".json"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-dedup",
        description="Find near-duplicate prompts with MinHash and LSH.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="group near-duplicate documents")
    scan.add_argument("paths", nargs="+", help="files or directories")
    scan.add_argument("--threshold", type=float, default=0.8)
    scan.add_argument("--shingle-size", type=int, default=3)
    scan.add_argument("--num-perm", type=int, default=128)
    scan.add_argument("--no-mask", action="store_true",
                      help="do not mask numbers and quoted examples")
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--fail-on-dupes", action="store_true")

    pair = sub.add_parser("pair", help="compare two documents directly")
    pair.add_argument("left")
    pair.add_argument("right")
    pair.add_argument("--shingle-size", type=int, default=3)
    pair.add_argument("--json", action="store_true")
    return parser


def _documents(paths: Sequence[str]) -> List[Tuple[str, str]]:
    documents: List[Tuple[str, str]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUFFIXES:
                    documents.append((str(child), child.read_text(encoding="utf-8",
                                                                  errors="replace")))
        elif path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(text)
                except ValueError:
                    payload = None
                if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
                    for index, item in enumerate(payload):
                        documents.append(("%s[%d]" % (path, index), item))
                    continue
            documents.append((str(path), text))
        else:
            print("skipping missing path: %s" % raw, file=sys.stderr)
    return documents


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "pair":
        left = Path(args.left).read_text(encoding="utf-8", errors="replace")
        right = Path(args.right).read_text(encoding="utf-8", errors="replace")
        result = compare_documents(left, right, k=args.shingle_size,
                                   mask_values=not getattr(args, "no_mask", False))
        print(json.dumps(result, indent=2) if args.json
              else "jaccard %.3f  minhash %.3f  shared %d"
                   % (result["jaccard"], result["minhash"], result["shared_shingles"]))
        return 0

    documents = _documents(args.paths)
    if not documents:
        print("no documents found", file=sys.stderr)
        return 2
    names = [name for name, _ in documents]
    texts = [text for _, text in documents]
    groups = clusters(texts, threshold=args.threshold, k=args.shingle_size,
                      num_perm=args.num_perm, mask_values=not args.no_mask)
    pairs = find_duplicates(texts, threshold=args.threshold, k=args.shingle_size,
                            num_perm=args.num_perm, mask_values=not args.no_mask)

    if args.json:
        print(json.dumps({
            "documents": len(documents),
            "threshold": args.threshold,
            "groups": [{"indices": group, "files": [names[i] for i in group]}
                       for group in groups],
            "pairs": [pair.as_dict() for pair in pairs],
        }, indent=2))
    else:
        print("%d document(s), %d group(s) at threshold %.2f"
              % (len(documents), len(groups), args.threshold))
        for group in groups:
            print("  duplicate group:")
            for index in group:
                print("    %s" % names[index])
        if pairs:
            print("closest pairs:")
            for pair in pairs[:10]:
                print("  %.2f  %s  <->  %s" % (pair.similarity, names[pair.left],
                                               names[pair.right]))
    if args.fail_on_dupes and groups:
        return 1
    return 0
