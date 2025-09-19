"""
Streamlit web app for matching customer order descriptions to product codes.

This app allows a user to upload a sales history Excel file and one or more customer
order files (in PDF or Excel format).  The sales history contains, per
customer, a product code, a description, quantities and a six‑month
purchase frequency.  When an order is submitted, the app attempts to
match each line of the order to a product in the history.  If the
product code in the order matches the history, that match is used
directly.  Otherwise, the app computes a similarity score between
descriptions and weights that score by the frequency of purchases to
propose the most likely product.  A threshold is used to avoid
spurious matches.

Key improvements over earlier versions include:

* File upload widgets use unique keys so that uploaded files persist
  across reruns; this avoids losing selections when a user uploads
  multiple files or when Streamlit refreshes the script.
* PDFs are parsed with pdfplumber when available; text is processed
  line by line to extract descriptions, quantities and product codes.
* Sales history parsing is resilient to different header names by
  inspecting column labels for key words (code, description, quantity,
  media 6 mesi).
* Matching logic combines Jaccard similarity on normalized tokens
  (stopwords removed and synonyms replaced) with a Levenshtein ratio
  and weights the result by purchase frequency.
"""

import io
import os
import re
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st

# Attempt to import pdfplumber for PDF parsing.  If unavailable the
# app will still run, but PDF orders cannot be parsed.
try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # type: ignore


# -----------------------------------------------------------------------------
# Environment setup
#
# Streamlit Cloud and other container environments may mount the root of the
# filesystem as read‑only.  Streamlit writes a machine ID file and some
# internal state to the user's home directory.  To avoid exceptions when
# running under these constraints, set HOME and XDG_STATE_HOME to a
# writeable directory under /tmp and ensure it exists.

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_STATE_HOME", "/tmp")
os.makedirs(os.path.join(os.environ["HOME"], ".streamlit"), exist_ok=True)


# -----------------------------------------------------------------------------
# Text normalization helpers
#
# To compare free‑form product descriptions, we tokenize each description,
# remove common Italian stopwords and apply a small set of synonyms to
# canonicalize terms.  Similarity is measured via a weighted combination of
# Jaccard similarity on the token sets and a Levenshtein ratio on the raw
# strings.

# Common Italian stop words that should not affect matching
STOPWORDS = {
    "in", "di", "da", "per", "a", "il", "la", "le", "i", "lo", "gli",
    "con", "su", "al", "del", "della", "dei", "degli", "un", "una", "uno",
    "ed", "e", "sul"
}

# Some domain‑specific synonyms to normalise variations in descriptions
SYNONYMS: Dict[str, str] = {
    "bobina": "rotolo",
    "bobine": "rotolo",
    "rotoli": "rotolo",
    "guanti": "guanto",
    "panni": "panno",
    "pellicole": "pellicola",
    "pellicola": "pellicola",
    "nitrile": "nitrile",
    # Additional variations can be added here
}


def normalize_tokens(description: str) -> List[str]:
    """Split a description into normalised tokens.

    The description is lower‑cased, punctuation is removed, and stopwords are
    discarded.  Synonyms are substituted where defined.

    Args:
        description: The original product description.

    Returns:
        A list of normalised tokens.
    """
    # Replace non‑alphanumeric characters with spaces
    cleaned = re.sub(r"[^\w\s]", " ", description.lower())
    tokens = []
    for tok in cleaned.split():
        if tok in STOPWORDS:
            continue
        tokens.append(SYNONYMS.get(tok, tok))
    return tokens


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def levenshtein_ratio(a: str, b: str) -> float:
    """Compute a normalised Levenshtein ratio between two strings.

    This implementation computes the Levenshtein distance and converts it to
    a ratio in the range [0, 1], where 1.0 indicates identical strings.
    """
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 1.0
    # Initialise distance matrix
    d = [[i if j == 0 else j if i == 0 else 0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost # substitution
            )
    distance = d[m][n]
    return 1.0 - distance / float(max(m, n))


def combined_similarity(desc1: str, desc2: str) -> float:
    """Combine Jaccard and Levenshtein similarities for two descriptions."""
    tokens1 = normalize_tokens(desc1)
    tokens2 = normalize_tokens(desc2)
    jac = jaccard_similarity(tokens1, tokens2)
    lev = levenshtein_ratio(desc1.lower(), desc2.lower())
    # Weight Jaccard slightly higher than Levenshtein
    return 0.6 * jac + 0.4 * lev


# -----------------------------------------------------------------------------
# Parsing helpers
#
def detect_headers_excel(df: pd.DataFrame) -> Dict[str, str]:
    """Infer column names for code, description, quantity and frequency.

    This function scans the DataFrame column labels for key substrings and
    returns a mapping from canonical field names to actual DataFrame column
    labels.  If no match is found the first or last column is returned as
    appropriate.

    Args:
        df: A pandas DataFrame representing the sales history.

    Returns:
        A dict with keys 'code', 'description', 'quantity', 'frequency'.
    """
    headers = [str(col).strip().lower() for col in df.columns]
    mapping: Dict[str, str] = {}

    def pick(subs: List[str], default_index: int = 0) -> str:
        for sub in subs:
            for i, h in enumerate(headers):
                if sub in h:
                    return str(df.columns[i])
        return str(df.columns[default_index])

    mapping['code'] = pick(['codice', 'code', 'art'])
    mapping['description'] = pick(['descr', 'descrizione', 'articolo', 'item'])
    mapping['quantity'] = pick(['quant', 'qta', 'qty'], default_index=len(df.columns) - 1)
    # Media 6 mesi may be in column named with 'media', 'mesi', or specific
    mapping['frequency'] = pick(['media', '6 mesi', 'six'], default_index=len(df.columns) - 1)
    return mapping


def parse_history_excel(file) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Parse an uploaded sales history Excel file.

    Args:
        file: The UploadedFile provided by Streamlit.

    Returns:
        A tuple with the DataFrame and a list of dictionaries representing
        products with their code, name, quantity ordered and frequency of
        purchase.
    """
    file_bytes = file.getvalue()
    buffer = io.BytesIO(file_bytes)
    df = pd.read_excel(buffer)
    mapping = detect_headers_excel(df)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        code = str(r.get(mapping['code'], '')).strip()
        desc = str(r.get(mapping['description'], '')).strip()
        try:
            qty = float(r.get(mapping['quantity'], 0) or 0)
        except Exception:
            qty = 0.0
        # Frequency (six‑month average)
        try:
            freq = float(r.get(mapping['frequency'], 0) or 0)
        except Exception:
            freq = 0.0
        rows.append({'item_code': code, 'item_name': desc, 'qty': qty, 'freq': freq})
    return df, rows


def parse_order_file(uploaded_file) -> List[Dict[str, Any]]:
    """Parse an uploaded order file, either PDF or Excel.

    Args:
        uploaded_file: The order file uploaded by the user.

    Returns:
        A list of dictionaries each containing 'product_code',
        'description' and 'quantity'.  If parsing fails, returns an empty list.
    """
    name = uploaded_file.name.lower()
    rows: List[Dict[str, Any]] = []

    # Helper to convert strings to floats safely
    def safe_float(val, default: float = 0.0) -> float:
        try:
            if val is None:
                return default
            return float(str(val).replace(',', '.'))
        except Exception:
            return default

    # PDF parsing
    if name.endswith('.pdf'):
        if pdfplumber is None:
            st.error("PDF support is not available because pdfplumber is not installed.")
            return []
        pdf_bytes = uploaded_file.getvalue()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    # Heuristic: first row may be header; determine columns for code, desc, qty
                    header = [str(c).lower().strip() for c in table[0]]
                    def idx_of(subs: List[str], default: int = 0) -> int:
                        for sub in subs:
                            for i, h in enumerate(header):
                                if sub in h:
                                    return i
                        return default
                    idx_code = idx_of(['code', 'codice', 'art'])
                    idx_desc = idx_of(['descr', 'articolo', 'item', 'description'], default=1 if len(header) > 1 else 0)
                    idx_qty = idx_of(['quant', 'qty', 'qta'], default=len(header) - 1)
                    for r in table[1:]:
                        if not r:
                            continue
                        code = str(r[idx_code]).strip() if idx_code < len(r) else ''
                        desc = str(r[idx_desc]).strip() if idx_desc < len(r) else ''
                        qty_val = r[idx_qty] if idx_qty < len(r) else ''
                        qty = safe_float(qty_val, default=1.0)
                        rows.append({'product_code': code, 'description': desc, 'quantity': qty})
                else:
                    # Fallback text parsing: split lines and attempt to extract quantity
                    text = page.extract_text() or ''
                    for raw in text.split('\n'):
                        line = raw.strip()
                        if not line:
                            continue
                        # Skip header or summary lines
                        if any(kw in line.lower() for kw in ['articolo fornitore', 'net total', 'grand net total', 'hsn code']):
                            continue
                        # Split by two or more spaces to separate description and numbers
                        parts = re.split(r"\s{2,}", line)
                        desc = parts[0].strip()
                        qty = 1.0
                        if len(parts) > 1:
                            # Look for the first numeric token in the remainder
                            m = re.search(r"\d+(?:[\.,]\d+)?", " ".join(parts[1:]))
                            if m:
                                qty = safe_float(m.group(0), default=1.0)
                        # Only accept lines that contain alphabetic characters
                        if desc and any(ch.isalpha() for ch in desc):
                            rows.append({'product_code': '', 'description': desc, 'quantity': qty})
        return rows

    # Excel parsing
    # Assume order file is an Excel workbook with columns: code, description, quantity
    excel_bytes = uploaded_file.getvalue()
    buffer = io.BytesIO(excel_bytes)
    df = pd.read_excel(buffer)
    # Determine columns heuristically
    cols = [str(c).lower() for c in df.columns]
    def pick_idx(subs: List[str], default: int) -> int:
        for sub in subs:
            for i, h in enumerate(cols):
                if sub in h:
                    return i
        return default
    idx_code = pick_idx(['code', 'codice', 'art'], 0)
    idx_desc = pick_idx(['descr', 'articolo', 'description', 'item'], 1 if len(cols) > 1 else 0)
    idx_qty = pick_idx(['quant', 'qty', 'qta'], len(cols) - 1)
    for _, r in df.iterrows():
        code = str(r.iloc[idx_code]).strip() if idx_code < len(r) else ''
        desc = str(r.iloc[idx_desc]).strip() if idx_desc < len(r) else ''
        qty = safe_float(r.iloc[idx_qty], default=1.0) if idx_qty < len(r) else 1.0
        rows.append({'product_code': code, 'description': desc, 'quantity': qty})
    return rows


def match_orders(
    history_rows: List[Dict[str, Any]],
    order_rows: List[Dict[str, Any]],
    threshold: float = 0.35,
    freq_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Match customer order lines to products in the sales history.

    Args:
        history_rows: A list of dicts with 'item_code', 'item_name' and 'freq'.
        order_rows: A list of dicts with 'product_code', 'description', 'quantity'.
        threshold: Minimum similarity score to accept a description match.
        freq_weight: Weight factor for purchase frequency in the similarity score.

    Returns:
        A list of dicts for each order line, containing the matched code,
        matched description, score and type ('exact_code' or 'description_similarity').
    """
    # Precompute max frequency to normalise weights
    max_freq = max((h['freq'] for h in history_rows), default=0.0)
    results: List[Dict[str, Any]] = []

    for o in order_rows:
        product_code = o.get('product_code', '').strip()
        best_match = None
        best_score = 0.0
        match_type = 'manual'

        # 1. Exact code match if code is present
        if product_code:
            for h in history_rows:
                if h['item_code'] == product_code:
                    best_match = h
                    best_score = 1.0
                    match_type = 'exact_code'
                    break

        # 2. Fuzzy description match if no exact code match
        if best_match is None:
            desc = o.get('description', '')
            for h in history_rows:
                score = combined_similarity(desc, h['item_name'])
                # Weight by frequency (scaled to [0, 1])
                if max_freq > 0:
                    score *= (1.0 + (h['freq'] / max_freq) * freq_weight)
                if score > best_score:
                    best_score = score
                    best_match = h
                    match_type = 'description_similarity'
            if best_score < threshold:
                best_match = None
                match_type = 'manual'

        results.append({
            'original_description': o.get('description', ''),
            'quantity': o.get('quantity', 0),
            'matched_code': best_match['item_code'] if best_match else '',
            'matched_description': best_match['item_name'] if best_match else '',
            'score': round(best_score, 3),
            'match_type': match_type,
        })
    return results


# -----------------------------------------------------------------------------
# Streamlit UI
#

def main() -> None:
    st.set_page_config(page_title="Order Matching", layout="wide")
    st.title("Order Matching Application")
    st.write("Upload a sales history file and a customer order to find matching product codes.")

    # Sidebar parameters
    st.sidebar.header("Matching Parameters")
    threshold = st.sidebar.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    freq_weight = st.sidebar.slider("Frequency Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    # File uploaders with unique keys so that selections persist across reruns
    uploaded_history = st.file_uploader(
        "Upload Sales History (Excel)", type=["xlsx", "xls"], key="history_file"
    )
    uploaded_order = st.file_uploader(
        "Upload Customer Order (PDF or Excel)", type=["pdf", "xlsx", "xls"], key="order_file"
    )

    history_rows: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []

    if uploaded_history:
        with st.spinner("Parsing sales history..."):
            _, history_rows = parse_history_excel(uploaded_history)
        st.success(f"Loaded {len(history_rows)} product records from history.")
    if uploaded_order:
        with st.spinner("Parsing customer order..."):
            order_rows = parse_order_file(uploaded_order)
        st.success(f"Loaded {len(order_rows)} order lines.")

    # Perform matching when both files are available
    if history_rows and order_rows:
        if st.button("Match Orders"):
            with st.spinner("Matching order lines to history..."):
                results = match_orders(history_rows, order_rows, threshold, freq_weight)
            st.subheader("Matched Results")
            st.write(
                "Rows with 'manual' match type indicate that the description did not meet the similarity "
                "threshold and require manual verification."
            )
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            # Provide a download link for the results
            towrite = io.BytesIO()
            results_df.to_excel(towrite, index=False)
            towrite.seek(0)
            st.download_button(
                label="Download Results as Excel",
                data=towrite,
                file_name="matched_orders.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
