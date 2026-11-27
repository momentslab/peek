#!/usr/bin/env python3
"""Score caption JSONL files from ``eval_captioning.py`` (CIDEr, BLEU-4, METEOR, ROUGE-L x100).

Needs ``java`` on PATH (pycocoevalcap). Failed generations are left out. Empty
captions follow ``score_empty_captions`` in the config (as in the paper: kept
on ANC, dropped on MSR-VTT); ``n`` in the table shows how many items were scored.

Example:
    python scripts/score_captions.py --dataset msrvtt \
        --manifest data/msrvtt/test_manifest.jsonl runs/msrvtt_qwen25vl3b*.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from peek.eval.data import anc_items, msrvtt_items
from peek.eval.metrics import caption_metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=("anc", "msrvtt"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("captions", nargs="+", type=Path)
    p.add_argument("--json", type=Path, help="also write the metrics here")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "eval_captioning.yaml")
    args = p.parse_args()
    keep_empty = yaml.safe_load(args.config.read_text())["datasets"][args.dataset]["score_empty_captions"]

    loader = anc_items if args.dataset == "anc" else msrvtt_items
    references = {item.key: item.references for item in loader(args.manifest, Path("."))}

    groups: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    failed: dict[tuple[str, int], int] = defaultdict(int)
    for path in args.captions:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            group = (row["selector"], int(row["k"]))
            if row.get("status") == "success" or (keep_empty and row.get("status") == "empty"):
                groups[group][row["key"]] = row.get("caption", "")
            else:
                failed[group] += 1

    results = []
    print("| selector | k | CIDEr | BLEU-4 | METEOR | ROUGE-L | n | failed |")
    print("|---|---|---|---|---|---|---|---|")
    for selector, k in sorted(groups):
        preds = groups[(selector, k)]
        metrics = caption_metrics(preds, {key: references[key] for key in preds})
        results.append({"selector": selector, "k": k, **metrics, "failed": failed[(selector, k)]})
        print(
            f"| {selector} | {k} | {100 * metrics['CIDEr']:.2f} | {100 * metrics['BLEU_4']:.2f} | "
            f"{100 * metrics['METEOR']:.2f} | {100 * metrics['ROUGE_L']:.2f} | {metrics['n']} | "
            f"{failed[(selector, k)]} |"
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
