"""Corpus captioning metrics with pycocoevalcap (needs ``java`` on PATH).

Same recipe as the paper: PTB-tokenize references and predictions, then
BLEU-4, METEOR, ROUGE-L and CIDEr(-D) over the whole set. Scores are returned
on the 0-1 scale; the paper tables multiply by 100.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def caption_metrics(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """``predictions[key]`` is one caption, ``references[key]`` its reference list.

    Keys present in only one mapping are an error; drop failed items first.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    if set(predictions) != set(references):
        raise ValueError("predictions and references must have the same keys")
    if not predictions:
        raise ValueError("nothing to score")

    tokenizer = PTBTokenizer()
    refs = tokenizer.tokenize({k: [{"caption": c} for c in references[k]] for k in predictions})
    preds = tokenizer.tokenize({k: [{"caption": predictions[k]}] for k in predictions})

    bleu, _ = Bleu(4).compute_score(refs, preds)
    cider, _ = Cider().compute_score(refs, preds)
    rouge, _ = Rouge().compute_score(refs, preds)
    meteor_scorer = Meteor()
    try:
        meteor, _ = meteor_scorer.compute_score(refs, preds)
    finally:
        if hasattr(meteor_scorer, "close"):
            meteor_scorer.close()
    return {
        "CIDEr": float(cider),
        "BLEU_4": float(bleu[3]),
        "METEOR": float(meteor),
        "ROUGE_L": float(rouge),
        "n": len(predictions),
    }
