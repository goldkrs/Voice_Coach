# backend/filler_detector.py
"""
Contextual filler detector using a masked language model (BERT family).
For each token in the transcript we mask it and ask the model for top-k predictions.
If the actual token is unlikely under the model (not in top-k and/or low probability),
we increase its 'filler score'. Finally we aggregate contiguous filler tokens into
filler phrases and return counts.

Output format:
{
  "filler_words": {"um": 3, "you know": 1, ...},
  "total_filler_count": 4,
  "detailed": [
    {"token": "um", "start_idx": 3, "score": 0.02, "is_filler": True},
    ...
  ]
}
"""

from typing import List, Dict
import math
import re
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict

# model choice (small / fast). You can change to "bert-base-uncased" or a lighter model.
MODEL_NAME = "bert-base-uncased"
TOP_K = 10  # check top-k predictions for masked token

# load model/tokenizer once
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
_model.eval()
if torch.cuda.is_available():
    _model.to("cuda")

# punctuation splitter, keep tokens aligned to words
_WORD_SPLIT_RE = re.compile(r"\s+|(?=[,.!?;:()\"'])|(?<=[,.!?;:()\"'])")

def _split_words_keep_punct(text: str) -> List[str]:
    """
    Split text into tokens roughly aligned to words/punctuation while preserving token order.
    This is not BERT tokenization; it's a simple tokenizer so masking aligns at word-level.
    """
    parts = [p for p in re.split(r'(\s+|[,.!?;:()"\'])', text) if p and not p.isspace()]
    return parts

def _clean_token(t: str) -> str:
    return t.strip()

def _get_masked_predictions(masked_text: str, mask_token: str = "[MASK]") -> Dict[str, float]:
    """
    Return a dict[token -> probability] for the mask position (top-K).
    """
    inputs = _tokenizer(masked_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    mask_token_index = (inputs["input_ids"] == _tokenizer.mask_token_id).nonzero(as_tuple=True)
    if mask_token_index[0].numel() == 0:
        return {}
    seq_idx = mask_token_index[1].item()
    logit_row = logits[0, seq_idx, :]
    probs = torch.softmax(logit_row, dim=-1)
    topk = torch.topk(probs, k=min(TOP_K, probs.size(-1)))
    topk_ids = topk.indices.cpu().tolist()
    topk_probs = topk.values.cpu().tolist()
    tokens = _tokenizer.convert_ids_to_tokens(topk_ids)
    return {t: p for t, p in zip(tokens, topk_probs)}

def detect_fillers_with_bert(transcript: str, top_k: int = TOP_K, prob_threshold: float = 0.05) -> dict:
    """
    Main entrypoint. Returns a dict with filler_counts and totals.
    - top_k: how many top predictions to consider a good fit
    - prob_threshold: if model assigns probability below this to the actual token, it is suspicious
    """
    if not transcript or not transcript.strip():
        return {"filler_words": {}, "total_filler_count": 0, "detailed": []}

    parts = _split_words_keep_punct(transcript)
    cleaned = [_clean_token(p) for p in parts]
    # Build a sequence of masked contexts
    filler_candidates = defaultdict(int)
    detailed = []

    # We will rebuild masked strings in a safe manner: join with spaces
    for i, token in enumerate(cleaned):
        token_str = token
        if token_str == "":
            continue
        # skip pure punctuation tokens for filler detection
        if re.fullmatch(r"[,.!?;:()\"']", token_str):
            detailed.append({"token": token_str, "index": i, "score": None, "is_filler": False})
            continue

        masked = cleaned.copy()
        masked[i] = _tokenizer.mask_token  # "[MASK]"
        masked_text = " ".join(masked)

        preds = _get_masked_predictions(masked_text)
        # convert token to BERT subtoken(s) and check presence
        # We'll check if any of the top predicted tokens equal or contain the original token (simple approx)
        token_norm = token_str.lower()
        # Try direct mapping via tokenizer (may produce subtokens like '##s')
        token_pieces = _tokenizer.tokenize(token_str)
        token_in_topk = False
        token_prob = 0.0

        # check each predicted token
        for pred_token, prob in preds.items():
            # normalize pred token (remove ##)
            pred_norm = pred_token.replace("##", "")
            if pred_norm.lower() == token_norm.lower():
                token_in_topk = True
                token_prob = prob
                break

        # compute a score: lower prob => more likely filler; absence in topk is suspicious
        # We'll create a filler_score in [0,1]
        if token_in_topk:
            score = max(0.0, 1.0 - token_prob)  # higher prob -> lower score
        else:
            # absent from topk -> treat as suspicious; set score using max topk prob as baseline
            max_topk_prob = max(preds.values()) if preds else 0.0
            score = 1.0 - max_topk_prob
            token_prob = 0.0

        is_filler = (not token_in_topk) or (token_prob < prob_threshold)

        detailed.append({"token": token_str, "index": i, "score": float(score), "is_filler": bool(is_filler)})

        if is_filler:
            # We normalize token (lowercase) and collapse adjacent small tokens like "i mean" detection can be post-processed
            filler_candidates[token_str.lower()] += 1

    # Post-process to join common adjacent filler patterns (simple heuristic)
    # e.g., ["i", "mean"] -> "i mean" if both flagged consecutively
    merged = defaultdict(int)
    i = 0
    while i < len(detailed):
        d = detailed[i]
        if d["is_filler"]:
            phrase_tokens = [d["token"]]
            j = i + 1
            while j < len(detailed) and detailed[j]["is_filler"] and re.fullmatch(r"\w+", detailed[j]["token"]):
                phrase_tokens.append(detailed[j]["token"])
                j += 1
            if len(phrase_tokens) > 1:
                phrase = " ".join(t.lower() for t in phrase_tokens)
                merged[phrase] += 1
                # also decrement single-token counts to avoid double-counting
                for t in phrase_tokens:
                    if filler_candidates.get(t.lower(), 0) > 0:
                        filler_candidates[t.lower()] -= 1
                        if filler_candidates[t.lower()] == 0:
                            filler_candidates.pop(t.lower(), None)
                i = j
                continue
        i += 1

    # combine single tokens and merged phrases
    for k, v in merged.items():
        filler_candidates[k] += v

    total = sum(filler_candidates.values())
    return {"filler_words": dict(filler_candidates), "total_filler_count": int(total), "detailed": detailed}