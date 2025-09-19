"""
Streamlit web app for matching customer order descriptions to product codes.

This version includes fixes to the file upload widgets so that selected files
persist after the Streamlit script reruns. Previously, `st.file_uploader`
widgets were re-evaluated on each run without a fixed key, causing the
selected file to disappear and leaving users uncertain whether their files
had been uploaded. By assigning a unique `key` to each file uploader and
storing uploaded files in `st.session_state`, the file selections remain
visible and the data is only parsed when a new file is chosen. A small
message confirms successful uploads.

The core matching logic remains similar to the previous implementation: it
normalizes product descriptions, performs fuzzy matching using a
combination of Jaccard similarity and a Levenshtein ratio, and falls back
to an exact code match when available. The results are displayed in a
table and can be downloaded as an Excel file.
"""

import io
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
# Before importing streamlit, ensure that any state files are written to a writable
# directory. On Hugging Face Spaces the default HOME may be '/' (read-only) which
# causes os.makedirs to attempt to create '/.streamlit' and raises PermissionError.
os.environ["HOME"] = "/tmp"
os.environ["XDG_STATE_HOME"] = "/tmp"
os.makedirs(os.path.join(os.environ["HOME"], ".streamlit"), exist_ok=True)
import streamlit as st

# Ensure Streamlit can write its machine_id and state files to a writable directory.
# In some container environments the default HOME or XDG_STATE_HOME points to a read-only
# location (like '/'), which causes a `PermissionError` when Streamlit tries to write
# `~/.streamlit/machine_id`. By overriding these variables to `/tmp` and creating
# the `.streamlit` directory there, we guarantee that Streamlit has a valid place
# to store its internal state. Without this, file uploads may silently fail.

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # pdf parsing will be disabled if library is missing

# --- Normalization helpers ---
# Italian stopwords and synonyms to improve matching on product descriptions.
STOPWORDS = {
    "in", "di", "da", "per", "a", "il", "la", "le", "i", "lo", "gli",
    "con", "su", "al", "del", "della", "dei", "degli", "un", "una", "uno",
    "ed", "e", "sul"
}

SYNONYMS: Dict[str, str] = {
    "bobina": "rotolo",
    "bobine": "rotolo",
    "rotoli": "rotolo",
    "guanti": "guanto",
    "panni": "panno",
    "pellicole": "pellicola",
    "nitrile": "nitrile",
    # Other common variations can be added here
}

def normalize_tokens(s: str) -> List[str]:
    """Normalize a description by lowercasing, removing punctuation,
    splitting on whitespace, dropping stopwords and applying synonyms."""
    import re
    tokens = re.sub(r"[^\w\s]", " ", s.lower()).split()
    normalized = []
    for token in tokens:
        if token in STOPWORDS or not token:
            continue
        normalized.append(SYNONYMS.get(token, token))
    return normalized

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
    """Compute normalized Levenshtein distance ratio (1 - distance/max_len)."""
    # Simple implementation of Levenshtein distance
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 1.0
    # Initialize distance matrix
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    # Populate matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost  # substitution
            )
    distance = d[m][n]
    return 1.0 - distance / max(m, n)

def combined_similarity(desc1: str, desc2: str) -> float:
    """Combine Jaccard similarity on normalized tokens with Levenshtein ratio."""
    tokens1 = normalize_tokens(desc1)
    tokens2 = normalize_tokens(desc2)
    jac = jaccard_similarity(tokens1, tokens2)
    lev = levenshtein_ratio(desc1.lower(), desc2.lower())
    # Weight Jaccard more heavily because tokens capture key words
    return 0.6 * jac + 0.4 * lev

# --- Parsing helpers ---

def detect_headers_excel(df: pd.DataFrame) -> Dict[str, str]:
    """Detect likely column names for code, description and quantity.
    Returns a mapping from canonical names to actual column names in the dataframe.
    This is a simple heuristic based on substrings."""
    headers = [str(col).strip().lower() for col in df.columns]
    mapping: Dict[str, str] = {}
    def pick(subs: List[str]) -> Optional[str]:
        for sub in subs:
            for col in df.columns:
                if sub in str(col).lower():
                    return col
        return None
    mapping['code'] = pick(['codice', 'code', 'art', 'item_code']) or df.columns[0]
    mapping['description'] = pick(['descr', 'desc', 'articolo', 'item', 'description']) or df.columns[1] if len(df.columns) > 1 else df.columns[0]
    mapping['quantity'] = pick(['quant', 'qty', 'qta']) or df.columns[-1]
    return mapping

def parse_history_excel(file) -> Tuple[pd.DataFrame, List[Dict[str, any]]]:
    """Parse an uploaded Excel file of purchase history.
    Returns the dataframe and a list of rows with standardized keys.

    The ``file`` parameter is a :class:`~streamlit.uploaded_file_manager.UploadedFile`
    object produced by :func:`streamlit.file_uploader`.  To avoid issues
    where the underlying file-like object becomes exhausted after the
    Streamlit script reruns, we first read the file's raw bytes into
    memory.  We then wrap those bytes in an :class:`io.BytesIO` buffer
    before passing it to :func:`pandas.read_excel`.

    Parameters
    ----------
    file : UploadedFile
        The Excel file uploaded by the user.

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict[str, any]]]
        A tuple containing the DataFrame representation of the file
        and a list of dictionaries with keys ``item_code``, ``item_name``
        and ``qty``.
    """
    # Read file into an in-memory bytes buffer.  This prevents
    # Streamlit's UploadedFile from being exhausted on subsequent
    # script reruns or when accessed multiple times.
    file_bytes = file.getvalue()
    buffer = io.BytesIO(file_bytes)
    df = pd.read_excel(buffer)
    mapping = detect_headers_excel(df)
    rows: List[Dict[str, any]] = []
    for _, r in df.iterrows():
        code = str(r.get(mapping['code'], '')).strip()
        desc = str(r.get(mapping['description'], '')).strip()
        qty = r.get(mapping['quantity'], 0)
        try:
            qty = float(qty)
        except Exception:
            qty = 0.0
        rows.append({'item_code': code, 'item_name': desc, 'qty': qty})
    return df, rows

def parse_order_file(uploaded_file) -> List[Dict[str, any]]:
    """Parse an uploaded order file, which may be PDF or Excel.
    Returns a list of order lines with vendor_code, item_description and quantity."""
    name = uploaded_file.name.lower()
    rows: List[Dict[str, any]] = []
    if name.endswith('.pdf'):
        if pdfplumber is None:
            st.error("Il supporto per i file PDF non è disponibile: pdfplumber non è installato.")
            return []
        # Read PDF from bytes to avoid issues with the UploadedFile being reset
        pdf_bytes = uploaded_file.getvalue()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                # Fallback to text parsing when no table is found.
                # Extract the raw text from the page and attempt to parse
                # each product line. We look for a header marker ('Articolo Fornitore')
                # to know when the item rows start, skip HSN codes and total lines,
                # and split columns based on runs of two or more spaces. This
                # heuristic works for the Optima order PDFs which list the
                # product description followed by quantity, price and total.
                text = page.extract_text()
                if text:
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    start_parsing = False
                    for line_txt in lines:
                        # Identify the start of the product list. In the Optima
                        # PDFs the header line contains 'Articolo Fornitore'.
                        if 'Articolo Fornitore' in line_txt:
                            start_parsing = True
                            continue
                        if not start_parsing:
                            continue
                        # Skip lines that are HSN codes or totals.
                        if any(kw in line_txt for kw in ['HSN Code', 'Net Total', 'Grand Net Total']):
                            continue
                        # Split by two or more spaces to separate columns.
                        parts = re.split(r'\s{2,}', line_txt)
                        if len(parts) >= 2:
                            desc = parts[0].strip()
                            m = re.match(r'([0-9]+,[0-9]+)', parts[1])
                            if m:
                                try:
                                    qty_val = float(m.group(1).replace(',', '.'))
                                except Exception:
                                    qty_val = 0.0
                                rows.append({'vendor_code': '',
                                             'item_description': desc,
                                             'qty': qty_val})
                    # After parsing text for this page, move to next page.
                continue
            header = [c.lower().strip() for c in table[0]]
            # heuristically determine column indices
            def idx_of(subs: List[str], default: int) -> int:
                for sub in subs:
                    for i, h in enumerate(header):
                        if sub in h:
                            return i
                return default
            idx_code = idx_of(['codice', 'code', 'art'], 0)
            idx_desc = idx_of(['descr', 'desc', 'articolo', 'item'], 1 if len(header) > 1 else 0)
            idx_qty = idx_of(['quant', 'qty'], len(header) - 1)
            for r in table[1:]:
                if not r:
                    continue
                code = r[idx_code] if idx_code < len(r) else ''
                desc = r[idx_desc] if idx_desc < len(r) else ''
                qty = r[idx_qty] if idx_qty < len(r) else '0'
                try:
                    qty = float(qty)
                except Exception:
                    qty = 0.0
                rows.append({'vendor_code': str(code).strip(),
                             'item_description': str(desc).strip(),
                             'qty': qty})
    else:
        # Assume Excel format for other file extensions.  Read the
        # uploaded file's bytes into a buffer first to avoid exhausting
        # the UploadedFile object on subsequent accesses.
        excel_bytes = uploaded_file.getvalue()
        buffer = io.BytesIO(excel_bytes)
        df = pd.read_excel(buffer)
        # pick columns heuristically
        header = [str(c).lower() for c in df.columns]
        def pick_idx(subs: List[str], default: int) -> int:
            for sub in subs:
                for i, h in enumerate(header):
                    if sub in h:
                        return i
            return default
        idx_code = pick_idx(['codice', 'code', 'art'], 0)
        idx_desc = pick_idx(['descr', 'desc', 'articolo', 'item'], 1 if len(header) > 1 else 0)
        idx_qty = pick_idx(['quant', 'qty'], len(header) - 1)
        for _, r in df.iterrows():
            code = r.iloc[idx_code] if idx_code < len(r) else ''
            desc = r.iloc[idx_desc] if idx_desc < len(r) else ''
            qty = r.iloc[idx_qty] if idx_qty < len(r) else 0
            try:
                qty = float(qty)
            except Exception:
                qty = 0.0
            rows.append({'vendor_code': str(code).strip(),
                         'item_description': str(desc).strip(),
                         'qty': qty})
    return rows

# --- Matching logic ---

def match_orders(history_rows: List[Dict[str, any]], order_rows: List[Dict[str, any]], threshold: float = 0.35) -> List[Dict[str, any]]:
    """Match each order row with the most similar history item.
    Returns a list of result dicts including the match and similarity score."""
    results: List[Dict[str, any]] = []
    for o in order_rows:
        best_match = None
        best_score = 0.0
        match_type = "manual"
        # Try exact code match first (if vendor_code is present)
        v_code = (o.get('vendor_code') or '').strip()
        if v_code:
            for h in history_rows:
                if (h.get('item_code') or '').strip() == v_code:
                    best_match = h
                    best_score = 1.0
                    match_type = 'exact_code'
                    break
        # Otherwise fall back to description similarity
        if best_match is None:
            for h in history_rows:
                score = combined_similarity(o.get('item_description', ''), h.get('item_name', ''))
                if score > best_score:
                    best_match = h
                    best_score = score
                    match_type = 'description'
        # Accept match only if above threshold
        if best_match and best_score >= threshold:
            results.append({
                'order_description': o.get('item_description', ''),
                'order_qty': o.get('qty', 0),
                'match_code': best_match.get('item_code', ''),
                'match_description': best_match.get('item_name', ''),
                'score': round(best_score, 3),
                'type': match_type
            })
        else:
            results.append({
                'order_description': o.get('item_description', ''),
                'order_qty': o.get('qty', 0),
                'match_code': '',
                'match_description': '',
                'score': 0.0,
                'type': 'manual'
            })
    return results

# --- Streamlit UI ---

st.set_page_config(page_title="Order Matching App")

st.title("Order Matching App")
st.write(
    "Carica lo *storico degli acquisti* e uno o più *ordini clienti* (PDF o Excel)."
    " Dopo il caricamento, l'app mostrerà il nome del file e, quando avrai sia lo storico"
    " che un ordine, potrai eseguire il matching per trovare i codici corretti."
)

# Sidebar: similarity threshold
st.sidebar.title("Parametri")
threshold = st.sidebar.slider("Soglia di somiglianza", 0.0, 1.0, 0.35, 0.05)

# Initialize session state
for key in ['history_rows', 'order_rows', 'history_file_name', 'order_file_name']:
    if key not in st.session_state:
        st.session_state[key] = None if 'name' in key else []

# Upload historic purchases (Excel)
uploaded_history = st.file_uploader(
    "1. Carica storico acquisti (file Excel .xlsx/.xls)",
    type=["xlsx", "xls"],
    key="history_uploader",
    help="Seleziona il file Excel contenente lo storico degli acquisti."
)
if uploaded_history is not None:
    # Only parse if a new file is selected
    if st.session_state.get('history_file_name') != uploaded_history.name:
        try:
            df_hist, hist_rows = parse_history_excel(uploaded_history)
            st.session_state['history_rows'] = hist_rows
            st.session_state['history_file_name'] = uploaded_history.name
            st.success(f"Storico caricato: {uploaded_history.name}")
        except Exception as e:
            st.session_state['history_rows'] = []
            st.error(f"Errore durante il caricamento dello storico: {e}")
    else:
        st.success(f"Storico già caricato: {uploaded_history.name}")

# Upload customer order (PDF or Excel)
uploaded_order = st.file_uploader(
    "2. Carica ordine cliente (file PDF o Excel)",
    type=["pdf", "xlsx", "xls"],
    key="order_uploader",
    help="Seleziona il file dell'ordine cliente in formato PDF o Excel."
)
if uploaded_order is not None:
    if st.session_state.get('order_file_name') != uploaded_order.name:
        try:
            ord_rows = parse_order_file(uploaded_order)
            st.session_state['order_rows'] = ord_rows
            st.session_state['order_file_name'] = uploaded_order.name
            st.success(f"Ordine caricato: {uploaded_order.name}")
        except Exception as e:
            st.session_state['order_rows'] = []
            st.error(f"Errore durante il caricamento dell'ordine: {e}")
    else:
        st.success(f"Ordine già caricato: {uploaded_order.name}")

st.write("""
### 3. Matching ordini con storico
Per eseguire il matching devi prima caricare sia lo storico che almeno un ordine.
Premi il pulsante qui sotto per avviare l'elaborazione.
""")

if st.button("Avvia matching"):
    if not st.session_state.get('history_rows'):
        st.warning("Devi prima caricare lo storico degli acquisti.")
    elif not st.session_state.get('order_rows'):
        st.warning("Devi prima caricare un ordine cliente.")
    else:
        with st.spinner("Eseguo il matching…"):
            results = match_orders(st.session_state['history_rows'], st.session_state['order_rows'], threshold)
            if results:
                res_df = pd.DataFrame(results)
                st.write("## Risultati matching", res_df)
                # Provide download link for Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    res_df.to_excel(writer, index=False)
                st.download_button(
                    "Scarica risultati", data=buffer.getvalue(),
                    file_name="risultati_matching.xlsx", mime="application/vnd.ms-excel"
                )
            else:
                st.info("Nessun abbinamento trovato sopra la soglia impostata.")