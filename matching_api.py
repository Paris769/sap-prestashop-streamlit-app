"""A small FastAPI service that exposes a `/match` endpoint for matching
order line items against a purchase history.

This service is meant to be deployed on a platform such as Streamlit
Cloud, Render, Heroku, or any other Python hosting service. Zapier can
call the `/match` endpoint via a Webhook to perform the matching.

Usage example:

    uvicorn matching_api:app --reload --host 0.0.0.0 --port 8000

Then issue a POST request to http://localhost:8000/match with a JSON
payload of the form:

    {
      "orders": [
        {"product_code": "1234", "description": "bobina in plastica", "quantity": 5},
        {"product_code": null, "description": "guanti in nitrile", "quantity": 10}
      ],
      "history": [
        {"item_code": "1234", "item_name": "bobina in plastica 10x", "six_months_avg": 0.3},
        {"item_code": "5678", "item_name": "guanto nitrile blu", "six_months_avg": 0.5}
      ],
      "threshold": 0.35,
      "weight": 0.3
    }

and receive a response containing matched rows. Each match indicates
whether it was found via exact code matching or via description
similarity. If the similarity score for a description falls below the
specified threshold, the item will be returned with a null code and a
``match_type`` of ``manual``.

Note that this file does not attempt to parse PDF or Excel files; it
assumes that the calling application (e.g. Zapier) has already
extracted and cleaned the order and history information into JSON.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from difflib import SequenceMatcher

app = FastAPI()

# Stopwords and synonyms for Italian product descriptions. These are used
# to normalize text before computing Jaccard similarity. Feel free to
# extend these lists to suit your domain.
STOPWORDS = set(
    "in di da per a il la le i lo gli con su al del della dei degli un una uno ed e sul".split()
)
SYNONYMS = {
    "bobina": "rotolo",
    "bobine": "rotolo",
    "rotoli": "rotolo",
    "guanti": "guanto",
    "panni": "panno",
    "pellicole": "pellicola",
    "nitrile": "nitrile",
    "articolo": "item",
    "codice": "code",
    # add more domain‑specific synonyms here
}


def normalize(text: str) -> List[str]:
    """Normalize a string into a list of lowercase tokens, removing punctuation,
    stopwords and applying simple synonyms.

    This function is used to prepare strings for Jaccard similarity.
    """
    # Replace non‑alphanumeric characters with spaces and lower the text
    cleaned = [
        (ch if ch.isalnum() else " ") for ch in text.lower()
    ]
    base = "".join(cleaned)
    tokens = []
    for token in base.split():
        if token in STOPWORDS:
            continue
        # apply synonym replacement if present
        tokens.append(SYNONYMS.get(token, token))
    return tokens


def jaccard(set_a: List[str], set_b: List[str]) -> float:
    """Compute the Jaccard index of two token lists."""
    A = set(set_a)
    B = set(set_b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def similarity(desc1: str, desc2: str) -> float:
    """Compute a blended similarity score between two product descriptions.

    The score is a weighted combination of Jaccard similarity on normalized
    tokens (60 %) and the Levenshtein ratio via difflib's SequenceMatcher (40 %).
    """
    tokens_a = normalize(desc1)
    tokens_b = normalize(desc2)
    jac = jaccard(tokens_a, tokens_b)
    lev = SequenceMatcher(None, desc1.lower(), desc2.lower()).ratio()
    return 0.6 * jac + 0.4 * lev


class OrderItem(BaseModel):
    product_code: Optional[str] = None
    description: str
    quantity: float


class HistoryItem(BaseModel):
    item_code: str
    item_name: str
    six_months_avg: float = 0.0


class MatchRequest(BaseModel):
    orders: List[OrderItem]
    history: List[HistoryItem]
    threshold: float = 0.35
    weight: float = 0.3  # how much to weight the six_months_avg in the final score


class MatchResult(BaseModel):
    original_desc: str
    quantity: float
    matched_code: Optional[str] = None
    matched_desc: Optional[str] = None
    score: float
    match_type: str  # "exact_code", "description_similarity" or "manual"


@app.post("/match")
def match_items(req: MatchRequest) -> dict:
    """Match each order item against the purchase history.

    - If an order item has a product_code that exactly matches a history
      item_code, return that match immediately with a score of 1.0.
    - Otherwise, compute the similarity between the order description and
      each history description, then adjust the score by (1 + weight * freq/max_freq).
    - If the best score exceeds the threshold, return that history entry.
    - Otherwise, indicate that the match requires manual review.
    """
    # Precompute the maximum six_months_avg for weighting
    max_freq = max((h.six_months_avg for h in req.history), default=0.0)
    results: List[MatchResult] = []
    for order in req.orders:
        best_score = 0.0
        best_match: Optional[HistoryItem] = None
        match_type = "manual"

        # Try exact code match first
        code = (order.product_code or "").strip()
        if code:
            for h in req.history:
                if h.item_code == code:
                    best_match = h
                    best_score = 1.0
                    match_type = "exact_code"
                    break
        # If no exact match found, try description similarity
        if not best_match:
            for h in req.history:
                base_score = similarity(order.description, h.item_name)
                # apply frequency weight
                if max_freq > 0:
                    freq_factor = (h.six_months_avg / max_freq) * req.weight
                else:
                    freq_factor = 0.0
                score = base_score * (1.0 + freq_factor)
                if score > best_score:
                    best_score = score
                    best_match = h
                    match_type = "description_similarity"
        # Check threshold
        if best_score < req.threshold:
            best_match = None
            match_type = "manual"
        results.append(
            MatchResult(
                original_desc=order.description,
                quantity=order.quantity,
                matched_code=(best_match.item_code if best_match else None),
                matched_desc=(best_match.item_name if best_match else None),
                score=best_score,
                match_type=match_type,
            )
        )
    return {"matches": [r.dict() for r in results]}
