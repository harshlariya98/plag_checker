"""
Internal Plagiarism Checker — /internal-plag-checker

Reference corpus is pre-loaded from a fixed local CSV.
Users only need to upload the new articles CSV to check.
"""

import html as html_module
import io
import os
import re
import sys
import threading
import time
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from plag_utils import fetch_text, check_article

# ── paths ─────────────────────────────────────────────────────────────────────
CORPUS_PATH = "/Users/harsh/Documents/final_data_plag.csv"
CACHE_DIR   = os.path.join(os.path.dirname(CORPUS_PATH), ".plag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE  = re.compile(r"\s+")

def _fast_strip(s):
    if not s or len(s) < 10:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()

def _vectorized_strip(series):
    return (series
            .str.replace(r"<[^>]+>", " ", regex=True)
            .str.replace(r"\s+",    " ", regex=True)
            .str.strip()
            .fillna(""))

def _fmt_eta(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s left"
    return f"{seconds // 60}m {seconds % 60}s left"

def _run_with_progress(fn, label, est_seconds):
    holder = [None]
    err    = [None]
    def _worker():
        try:   holder[0] = fn()
        except Exception as e: err[0] = e
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    prog  = st.progress(0.0)
    info  = st.empty()
    start = time.time()
    while t.is_alive():
        elapsed   = time.time() - start
        frac      = min(elapsed / est_seconds, 0.95)
        remaining = max(0, est_seconds - elapsed)
        prog.progress(frac)
        info.markdown(
            f'<p class="eta-label">{label} &nbsp;·&nbsp; {_fmt_eta(remaining)}</p>',
            unsafe_allow_html=True,
        )
        time.sleep(0.4)
    t.join()
    prog.progress(1.0); info.empty(); prog.empty()
    if err[0]: raise err[0]
    return holder[0]

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Checker · KollegeApply",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
/* ── reset & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }

.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1280px !important;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #FAFAFA !important;
    border-right: 1px solid #E5E7EB !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
section[data-testid="stSidebar"] .sidebar-logo {
    font-size: .7rem; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #9CA3AF; padding: 0 1rem 1rem;
}
section[data-testid="stSidebar"] hr { border-color: #E5E7EB !important; }
section[data-testid="stSidebar"] label { color: #6B7280 !important; font-size: .8rem !important; font-weight: 500 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #4F46E5 !important; }
section[data-testid="stSidebar"] p { color: #374151 !important; font-size: .85rem !important; }
section[data-testid="stSidebar"] small { color: #9CA3AF !important; }

/* ── page header ── */
.page-header { margin-bottom: 2rem; }
.page-header h1 {
    font-size: 1.5rem; font-weight: 700; color: #111827;
    margin: 0 0 .35rem; letter-spacing: -.3px;
}
.page-header p { font-size: .9rem; color: #6B7280; margin: 0; line-height: 1.6; }

/* ── corpus status bar ── */
.status-bar {
    display: flex; align-items: center; gap: .75rem;
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-radius: 10px; padding: .6rem 1rem;
    font-size: .85rem; font-weight: 500; color: #166534;
    margin-bottom: 1.75rem;
}
.status-bar .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #16A34A; flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(22,163,74,.2);
}
.status-bar.error { background: #FEF2F2; border-color: #FECACA; color: #991B1B; }
.status-bar.error .dot { background: #DC2626; box-shadow: 0 0 0 3px rgba(220,38,38,.2); }

/* ── section heading ── */
.section-head {
    display: flex; align-items: center; gap: .75rem;
    margin: 2rem 0 .75rem;
}
.section-num {
    width: 1.6rem; height: 1.6rem; border-radius: 50%;
    background: #4F46E5; color: #fff;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: .75rem; font-weight: 700; flex-shrink: 0;
}
.section-title { font-size: .95rem; font-weight: 600; color: #111827; }

/* ── callout ── */
.callout {
    background: #F8FAFF; border: 1px solid #DBEAFE;
    border-radius: 8px; padding: .75rem 1rem;
    font-size: .83rem; color: #1E40AF; line-height: 1.6;
    margin: .5rem 0 1rem;
}
.callout.warn { background: #FFFBEB; border-color: #FDE68A; color: #92400E; }
.callout.success { background: #F0FDF4; border-color: #BBF7D0; color: #166534; }

/* ── build steps (first-time index) ── */
.build-wrap {
    background: #fff; border: 1px solid #E5E7EB;
    border-radius: 12px; padding: 1.5rem 1.75rem; margin-bottom: 1rem;
}
.build-step-title {
    font-size: .8rem; font-weight: 600; letter-spacing: .05em;
    text-transform: uppercase; color: #6B7280; margin-bottom: .5rem;
}
.eta-label { font-size: .8rem; color: #9CA3AF; margin: .25rem 0 0; }

/* ── upload card ── */
.upload-hint {
    font-size: .82rem; color: #9CA3AF;
    margin: .5rem 0 0; line-height: 1.6;
}

/* ── stat tiles ── */
.stat-row { display: flex; gap: 1rem; margin: 1rem 0 1.25rem; flex-wrap: wrap; }
.stat-tile {
    flex: 1; min-width: 120px;
    background: #fff; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: .85rem 1rem;
    text-align: center;
}
.stat-tile .sv { font-size: 1.65rem; font-weight: 700; color: #111827; line-height: 1.1; }
.stat-tile .sl { font-size: .7rem; font-weight: 600; letter-spacing: .06em;
                 text-transform: uppercase; color: #9CA3AF; margin-top: .25rem; }
.stat-tile.red   .sv { color: #DC2626; }
.stat-tile.amber .sv { color: #D97706; }
.stat-tile.green .sv { color: #059669; }
.stat-tile.indigo .sv { color: #4F46E5; }

/* ── alert banner ── */
.alert-banner {
    display: flex; align-items: center; gap: .75rem;
    background: #FEF2F2; border: 1px solid #FECACA;
    border-radius: 10px; padding: .85rem 1.1rem;
    font-size: .875rem; color: #991B1B; font-weight: 500;
    margin: .75rem 0;
}
.alert-banner.ok {
    background: #F0FDF4; border-color: #BBF7D0; color: #166534;
}

/* ── result cards ── */
.rc {
    background: #fff; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: .85rem 1.1rem;
    margin: .4rem 0; display: flex; gap: 1rem; align-items: flex-start;
    transition: box-shadow .15s;
}
.rc:hover { box-shadow: 0 4px 12px rgba(0,0,0,.06); }
.rc.danger { border-left: 3px solid #DC2626; }
.rc.warn   { border-left: 3px solid #F59E0B; }
.rc.ok     { border-left: 3px solid #059669; }

.rc-pct {
    font-size: 1.4rem; font-weight: 700; min-width: 3.25rem;
    text-align: center; line-height: 1; padding-top: .1rem;
}
.rc.danger .rc-pct { color: #DC2626; }
.rc.warn   .rc-pct { color: #F59E0B; }
.rc.ok     .rc-pct { color: #059669; }

.rc-body { flex: 1; font-size: .84rem; color: #374151; line-height: 1.65; }
.rc-body a { color: #4F46E5; text-decoration: none; }
.rc-body a:hover { text-decoration: underline; }
.rc-label { font-size: .7rem; font-weight: 600; letter-spacing: .05em;
            text-transform: uppercase; color: #9CA3AF; margin-bottom: .1rem; }
.rc-badge {
    display: inline-block;
    padding: 1px 8px; border-radius: 99px; font-size: .72rem; font-weight: 600;
    margin-left: .35rem; vertical-align: middle;
}
.rc-badge.danger { background: #FEE2E2; color: #991B1B; }
.rc-badge.warn   { background: #FEF3C7; color: #92400E; }
.rc-badge.ok     { background: #D1FAE5; color: #065F46; }
.rc-meta { font-size: .75rem; color: #9CA3AF; margin-top: .3rem; }

/* ── tab switcher ── */
div[data-testid="stSegmentedControl"] { margin: .75rem 0 1rem; }

/* ── primary button ── */
.stButton > button[kind="primary"] {
    background: #4F46E5 !important; border: none !important;
    font-weight: 600 !important; letter-spacing: .01em !important;
    box-shadow: 0 1px 3px rgba(79,70,229,.3) !important;
    border-radius: 8px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4338CA !important;
    box-shadow: 0 4px 12px rgba(79,70,229,.35) !important;
}

/* ── misc ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }
div[data-testid="stExpander"] { border-radius: 10px !important; }
.stProgress > div > div > div { background: #4F46E5 !important; border-radius: 99px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def section(n, title):
    st.markdown(
        f'<div class="section-head">'
        f'<span class="section-num">{n}</span>'
        f'<span class="section-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def smart_col(df, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None

def verdict_for(score, threshold):
    if score >= threshold:
        return "danger", "Duplicate"
    if score >= 40:
        return "warn",   "Review"
    return "ok",     "Unique"

def normalise_new(df):
    url_col  = smart_col(df, ["url","link","page_url","article_url","slug"])
    desc_col = smart_col(df, ["description","content","article","body","html",
                               "article_html","text","article_body"])
    if not url_col:
        return None, f"No URL column found. Columns: {list(df.columns)}"
    rename = {url_col: "url"}
    if desc_col: rename[desc_col] = "description"
    df = df.rename(columns=rename)
    if "description" not in df.columns: df["description"] = ""
    return df[["url","description"] +
               [c for c in df.columns if c not in ("url","description")]].copy(), None


# ─────────────────────────────────────────────────────────────────────────────
# Corpus helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_corpus(path):
    df = pd.read_csv(path, usecols=lambda c: c in ["url","description"])
    df["url"] = df["url"].astype(str).str.strip()
    if "description" not in df.columns: df["description"] = ""
    df["description"] = df["description"].astype(str)
    return df

def _cache_key():
    return str(int(os.path.getmtime(CORPUS_PATH)))

def _disk_cache_exists(key):
    return all(os.path.exists(os.path.join(CACHE_DIR, f"{key}.{ext}"))
               for ext in ("vec.pkl", "mat.npz", "urls.pkl"))

def _load_from_disk(key):
    vec  = joblib.load(os.path.join(CACHE_DIR, f"{key}.vec.pkl"))
    mat  = load_npz(os.path.join(CACHE_DIR, f"{key}.mat.npz"))
    urls = joblib.load(os.path.join(CACHE_DIR, f"{key}.urls.pkl"))
    return vec, mat, urls

def _save_to_disk(key, vec, mat, urls):
    joblib.dump(vec,  os.path.join(CACHE_DIR, f"{key}.vec.pkl"))
    save_npz(         os.path.join(CACHE_DIR, f"{key}.mat.npz"), mat)
    joblib.dump(urls, os.path.join(CACHE_DIR, f"{key}.urls.pkl"))


def _ensure_corpus_index():
    if "corpus_urls" in st.session_state:
        return

    key = _cache_key()

    if _disk_cache_exists(key):
        with st.spinner("Loading index from disk…"):
            vec, mat, urls = _load_from_disk(key)
        st.session_state["corpus_urls"] = urls
        st.session_state["corpus_vec"]  = vec
        st.session_state["corpus_mat"]  = mat
        return

    # ── first-time build ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="callout">First-time setup — the index will be saved to disk '
        'and loads automatically on every session after this.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        # Step 1 — read + strip
        st.markdown('<p class="build-step-title">Step 1 of 3 &nbsp;·&nbsp; Reading & parsing corpus</p>',
                    unsafe_allow_html=True)
        prog1 = st.progress(0.0)
        info1 = st.empty()

        CHUNK = 10_000
        url_chunks, text_chunks = [], []
        with open(CORPUS_PATH, "rb") as f:
            total_rows = sum(1 for _ in f) - 1

        reader    = pd.read_csv(CORPUS_PATH,
                                usecols=lambda c: c in ["url","description"],
                                chunksize=CHUNK)
        rows_done = 0
        t0        = time.time()
        for chunk in reader:
            chunk["url"]         = chunk["url"].astype(str).str.strip()
            chunk["description"] = chunk.get("description",
                                             pd.Series([""] * len(chunk))).astype(str)
            url_chunks.append(chunk["url"])
            text_chunks.append(_vectorized_strip(chunk["description"]))
            rows_done += len(chunk)
            frac    = min(rows_done / total_rows, 1.0)
            elapsed = time.time() - t0
            eta     = (elapsed / frac * (1 - frac)) if frac > 0.01 else 0
            prog1.progress(frac)
            info1.markdown(
                f'<p class="eta-label">{rows_done:,} / {total_rows:,} rows'
                f'&nbsp;·&nbsp; {_fmt_eta(eta)}</p>',
                unsafe_allow_html=True,
            )
        prog1.empty(); info1.empty()
        urls  = pd.concat(url_chunks).tolist()
        texts = pd.concat(text_chunks).tolist()
        n     = len(urls)

        # Step 2 — TF-IDF
        st.markdown(f'<p class="build-step-title">Step 2 of 3 &nbsp;·&nbsp; Building TF-IDF index ({n:,} articles)</p>',
                    unsafe_allow_html=True)
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                              min_df=2, max_features=50000)
        mat = _run_with_progress(lambda: vec.fit_transform(texts),
                                 "Vectorising", 40)

        # Step 3 — save
        st.markdown('<p class="build-step-title">Step 3 of 3 &nbsp;·&nbsp; Saving to disk</p>',
                    unsafe_allow_html=True)
        _run_with_progress(lambda: _save_to_disk(key, vec, mat, urls),
                           "Writing files", 15)

    st.session_state["corpus_urls"] = urls
    st.session_state["corpus_vec"]  = vec
    st.session_state["corpus_mat"]  = mat
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-logo">KollegeApply</p>', unsafe_allow_html=True)
    st.markdown("**Plag Checker Settings**")
    st.divider()

    dup_threshold = st.slider("Duplicate threshold", 20, 100, 65,
                              format="%d%%",
                              help="Articles above this % similarity are flagged as duplicates")

    st.divider()
    st.markdown("**Web check** *(optional)*")
    run_web = st.checkbox("Check against open web", value=False)
    if run_web:
        n_passages   = st.slider("Passages per article", 4, 20, 8)
        web_thresh   = st.slider("Match threshold", 70, 100, 85, format="%d%%")
        own_domain   = st.text_input("Your domain (excluded)", "kollegeapply.com")
        excl_domains = st.text_area("Other excluded domains",
                                    "wikipedia.org\nyoutube.com", height=60)

    st.divider()
    top_n = st.number_input("Max results", 10, 5000, 200)

    st.divider()
    st.markdown(
        '<small>'
        '<b style="color:#059669">●</b> &lt;40% unique &nbsp; '
        '<b style="color:#D97706">●</b> 40–65% review &nbsp; '
        '<b style="color:#DC2626">●</b> &gt;65% duplicate'
        '</small>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<h1>Internal Plagiarism Checker</h1>'
    '<p>Upload a CSV of new articles to check them against the KollegeApply article database.'
    ' Results show per-article similarity scores and closest matching existing article.</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-load corpus
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CORPUS_PATH):
    st.markdown(
        f'<div class="status-bar error"><span class="dot"></span>'
        f'Corpus file not found at <code>{CORPUS_PATH}</code></div>',
        unsafe_allow_html=True,
    )
    st.stop()

_ensure_corpus_index()

corpus_urls = st.session_state["corpus_urls"]
corpus_vec  = st.session_state["corpus_vec"]
corpus_mat  = st.session_state["corpus_mat"]
n_corp      = len(corpus_urls)

st.markdown(
    f'<div class="status-bar"><span class="dot"></span>'
    f'Reference corpus ready &nbsp;·&nbsp; <b>{n_corp:,} articles</b> indexed</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
section(1, "Upload articles to check")

new_df = None
new_src = st.radio("Input method", ["Upload CSV", "Paste URLs"],
                   horizontal=True, key="new_src",
                   label_visibility="collapsed")

with st.container(border=True):
    if new_src == "Upload CSV":
        new_file = st.file_uploader(
            "Drop your CSV here",
            type=["csv"], key="new_upload",
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="upload-hint">Required column: <code>url</code> &nbsp;·&nbsp; '
            'Optional: <code>description</code> (HTML content — avoids URL fetching)</p>',
            unsafe_allow_html=True,
        )
        if new_file:
            try:
                raw_new = pd.read_csv(new_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                raw_new = None
            if raw_new is not None:
                df2, err2 = normalise_new(raw_new)
                if err2:
                    st.warning(err2)
                    mc1, mc2 = st.columns(2)
                    url_c  = mc1.selectbox("URL column", list(raw_new.columns), key="nu")
                    desc_c = mc2.selectbox("Description column",
                                           ["(none)"] + list(raw_new.columns), key="nd")
                    rename = {url_c: "url"}
                    if desc_c != "(none)": rename[desc_c] = "description"
                    df2 = raw_new.rename(columns=rename)
                    if "description" not in df2.columns: df2["description"] = ""
                new_df = df2
                if new_df is not None:
                    new_df["url"]         = new_df["url"].astype(str).str.strip()
                    new_df["description"] = new_df["description"].astype(str).fillna("")
                    has_desc = (new_df["description"].str.strip().str.len() > 10).sum()
                    no_desc  = len(new_df) - has_desc
                    parts = [f"**{len(new_df):,}** articles loaded"]
                    if has_desc: parts.append(f"{has_desc} with HTML content")
                    if no_desc:  parts.append(f"{no_desc} will be fetched")
                    st.success("  ·  ".join(parts))
    else:
        pasted = st.text_area("One URL per line", height=120,
                              placeholder="https://www.kollegeapply.com/…")
        if pasted.strip():
            urls_list = [u.strip() for u in pasted.splitlines()
                         if u.strip().startswith("http")]
            if urls_list:
                new_df = pd.DataFrame({"url": urls_list, "description": ""})
                st.success(f"**{len(urls_list)}** URLs — content will be fetched")
            else:
                st.error("No valid URLs found (must start with http).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Run
# ─────────────────────────────────────────────────────────────────────────────
if new_df is not None and len(new_df) > 0:
    section(2, "Run check")

    n_new   = len(new_df)
    n_fetch = int((new_df["description"].str.strip().str.len() <= 10).sum())
    est_sec = n_new * 0.5 + n_fetch * 4
    if run_web:
        est_sec += n_new * n_passages * 5

    # stats row
    est_label = (f"~{max(1,round(est_sec/60))} min" if est_sec > 60
                 else f"~{int(est_sec)}s")
    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-tile indigo"><div class="sv">{n_new}</div><div class="sl">New articles</div></div>'
        f'<div class="stat-tile"><div class="sv">{n_corp:,}</div><div class="sl">Corpus size</div></div>'
        f'<div class="stat-tile"><div class="sv">{n_fetch}</div><div class="sl">URLs to fetch</div></div>'
        f'<div class="stat-tile"><div class="sv">{est_label}</div><div class="sl">Est. time</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_fetch > 0:
        st.markdown(
            '<div class="callout warn">Articles without a <code>description</code> column '
            'will be fetched from their URL — add HTML content to speed this up.</div>',
            unsafe_allow_html=True,
        )

    run_btn = st.button("Run check", type="primary", use_container_width=True, key="run_btn")

    if run_btn:
        # prepare new articles
        with st.status("Preparing articles…", expanded=True) as s2:
            prog = st.progress(0.0)
            cur  = st.empty()
            new_texts, new_wc = [], []
            for i, row in enumerate(new_df.itertuples(), 1):
                cur.markdown(
                    f'<p style="font-size:.83rem;color:#6B7280;margin:0;">'
                    f'Article {i} of {n_new} &nbsp;·&nbsp; {row.url}</p>',
                    unsafe_allow_html=True,
                )
                desc = str(row.description).strip()
                text = _fast_strip(desc) if len(desc) > 10 else fetch_text(row.url)
                new_texts.append(text)
                new_wc.append(len(text.split()))
                prog.progress(i / n_new)
            cur.empty()
            s2.update(label=f"Articles ready  ·  {n_new} processed",
                      state="complete", expanded=False)

        # similarity
        with st.status("Computing similarity…", expanded=False) as s3:
            valid_new   = [(i, t) for i, t in enumerate(new_texts) if len(t.split()) >= 30]
            sim_results = [(0.0, 0)] * n_new
            if valid_new and corpus_mat.shape[0] > 0:
                vt      = [t for _, t in valid_new]
                new_mat = corpus_vec.transform(vt)
                sims    = cosine_similarity(new_mat, corpus_mat)
                for ni, (orig_i, _) in enumerate(valid_new):
                    best_j  = int(sims[ni].argmax())
                    best_sc = round(float(sims[ni, best_j]) * 100, 1)
                    sim_results[orig_i] = (best_sc, best_j)
            s3.update(label="Similarity computed", state="complete")

        # optional web check
        web_res = {}
        if run_web:
            excl = {d.strip().lower() for d in excl_domains.splitlines() if d.strip()}
            excl.add(own_domain.strip().lower().replace("www.", ""))
            with st.status(f"Web check — {n_new} articles…", expanded=True) as sw:
                wp = st.progress(0.0)
                wt = st.empty()
                for i, (row, txt) in enumerate(zip(new_df.itertuples(), new_texts), 1):
                    wt.markdown(
                        f'<p style="font-size:.83rem;color:#6B7280;margin:0;">'
                        f'{i}/{n_new} &nbsp;·&nbsp; {row.url}</p>',
                        unsafe_allow_html=True,
                    )
                    try:
                        res, _ = check_article(row.url, n_passages, web_thresh,
                                               excl, 25, preloaded_text=txt)
                    except Exception as e:
                        res = {"plagiarism_score": None, "verdict": f"Error: {e}",
                               "top_sources": "", "matches": []}
                    web_res[row.url] = res
                    wp.progress(i / n_new)
                wt.empty()
                sw.update(label="Web check complete", state="complete", expanded=False)

        # assemble
        final_rows = []
        for row, txt, wc, (sc, ci) in zip(
            new_df.itertuples(), new_texts, new_wc, sim_results
        ):
            corp_url = corpus_urls[ci] if ci < len(corpus_urls) else ""
            wr       = web_res.get(row.url, {})
            cls, vlabel = verdict_for(sc, dup_threshold)[:2]
            final_rows.append({
                "url":                    row.url,
                "word_count":             wc,
                "similarity_to_corpus_%": sc,
                "verdict":                vlabel,
                "matched_existing_url":   corp_url,
                "web_plag_score":         wr.get("plagiarism_score"),
                "web_verdict":            wr.get("verdict", ""),
                "top_web_sources":        wr.get("top_sources", ""),
                "_matches":               wr.get("matches", []),
            })

        st.session_state["results"] = final_rows
        st.rerun()

else:
    if new_df is None:
        st.markdown(
            '<div class="callout">Upload a CSV or paste URLs above to get started.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Results
# ─────────────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    rdf = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])

    # header row
    h1, h2 = st.columns([5, 1], vertical_alignment="bottom")
    with h1:
        section(3, "Results")
    with h2:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("results", None)
            st.rerun()

    # summary tiles
    n_dup  = int((rdf["similarity_to_corpus_%"] >= dup_threshold).sum())
    n_rev  = int(((rdf["similarity_to_corpus_%"] >= 40) &
                  (rdf["similarity_to_corpus_%"] < dup_threshold)).sum())
    n_ok   = int((rdf["similarity_to_corpus_%"] < 40).sum())
    avg_s  = f"{rdf['similarity_to_corpus_%'].mean():.1f}%"

    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-tile"><div class="sv">{len(rdf)}</div><div class="sl">Checked</div></div>'
        f'<div class="stat-tile"><div class="sv">{avg_s}</div><div class="sl">Avg similarity</div></div>'
        f'<div class="stat-tile red"><div class="sv">{n_dup}</div><div class="sl">Duplicates ≥{dup_threshold}%</div></div>'
        f'<div class="stat-tile amber"><div class="sv">{n_rev}</div><div class="sl">Review needed</div></div>'
        f'<div class="stat-tile green"><div class="sv">{n_ok}</div><div class="sl">Unique</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_dup:
        st.markdown(
            f'<div class="alert-banner">'
            f'<span style="font-size:1rem;">⚠</span> '
            f'<b>{n_dup} article{"s" if n_dup>1 else ""}</b> '
            f'{"are" if n_dup>1 else "is"} ≥{dup_threshold}% similar to existing content'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif results:
        st.markdown(
            '<div class="alert-banner ok">All articles are below the duplicate threshold.</div>',
            unsafe_allow_html=True,
        )

    # view tabs
    view = st.segmented_control(
        "View", ["All results", "Duplicates only", "Web plagiarism"],
        default="All results", label_visibility="collapsed",
    ) or "All results"

    # ── table + cards ─────────────────────────────────────────────────────────
    if view in ("All results", "Duplicates only"):
        show = results if view == "All results" else [
            r for r in results if r["similarity_to_corpus_%"] >= dup_threshold
        ]
        show = sorted(show, key=lambda r: r["similarity_to_corpus_%"], reverse=True)
        show = show[:int(top_n)]

        if not show:
            st.markdown(
                '<div class="callout success">No duplicates found at the current threshold.</div>',
                unsafe_allow_html=True,
            )
        else:
            tdf = pd.DataFrame([{
                "Article":          r["url"],
                "Words":            r["word_count"],
                "Similarity":       r["similarity_to_corpus_%"],
                "Verdict":          r["verdict"],
                "Closest existing": r["matched_existing_url"],
            } for r in show])

            st.dataframe(
                tdf, hide_index=True, use_container_width=True,
                column_config={
                    "Article":          st.column_config.LinkColumn(width="large"),
                    "Words":            st.column_config.NumberColumn(width="small"),
                    "Similarity":       st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.0f%%", width="small"),
                    "Verdict":          st.column_config.TextColumn(width="small"),
                    "Closest existing": st.column_config.LinkColumn(width="large"),
                },
            )

            # detail cards for flagged
            flagged = [r for r in show if r["similarity_to_corpus_%"] >= 40]
            if flagged:
                st.markdown(
                    '<p style="font-size:.8rem;font-weight:600;letter-spacing:.05em;'
                    'text-transform:uppercase;color:#9CA3AF;margin:1.25rem 0 .6rem;">Detail view</p>',
                    unsafe_allow_html=True,
                )
                for r in flagged:
                    sc  = r["similarity_to_corpus_%"]
                    cls, vlabel = verdict_for(sc, dup_threshold)[:2]
                    new_e   = html_module.escape(r["url"])
                    corp_u  = html_module.escape(r["matched_existing_url"], quote=True)
                    corp_ue = html_module.escape(r["matched_existing_url"])
                    st.markdown(
                        f'<div class="rc {cls}">'
                        f'<div class="rc-pct">{sc:.0f}%</div>'
                        f'<div class="rc-body">'
                        f'<div class="rc-label">New article</div>'
                        f'<a href="{new_e}" target="_blank">{new_e}</a>'
                        f'<span class="rc-badge {cls}">{vlabel}</span>'
                        f'<div class="rc-label" style="margin-top:.6rem;">Matches existing</div>'
                        f'<a href="{corp_u}" target="_blank">{corp_ue}</a>'
                        f'<div class="rc-meta">{r["word_count"]:,} words</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

    # ── web plagiarism ────────────────────────────────────────────────────────
    else:
        if not run_web:
            st.markdown(
                '<div class="callout">Enable <b>Check against open web</b> in the sidebar '
                'and re-run to see web results.</div>',
                unsafe_allow_html=True,
            )
        else:
            web_show = [r for r in results if r.get("web_plag_score") is not None]
            if not web_show:
                st.info("No web results available.")
            else:
                wdf = pd.DataFrame([{
                    "Article":     r["url"],
                    "Web score":   r["web_plag_score"],
                    "Verdict":     r["web_verdict"],
                    "Top sources": r["top_web_sources"],
                } for r in web_show]).sort_values("Web score", ascending=False)
                st.dataframe(
                    wdf, hide_index=True, use_container_width=True,
                    column_config={
                        "Article":     st.column_config.LinkColumn(width="large"),
                        "Web score":   st.column_config.ProgressColumn(
                            min_value=0, max_value=100, format="%.0f%%", width="small"),
                        "Verdict":     st.column_config.TextColumn(width="medium"),
                        "Top sources": st.column_config.TextColumn(width="large"),
                    },
                )
                for r in web_show:
                    if r.get("_matches"):
                        with st.expander(r["url"]):
                            for m in r["_matches"]:
                                src = html_module.escape(m["source"], quote=True)
                                st.markdown(
                                    f'<div style="border-left:3px solid #DC2626;padding:.6rem 1rem;'
                                    f'background:#FEF2F2;border-radius:0 8px 8px 0;margin:.4rem 0;'
                                    f'font-size:.84rem;color:#374151;">'
                                    f'<i>"{html_module.escape(m["passage"])}"</i><br>'
                                    f'<span style="font-size:.75rem;color:#9CA3AF;">'
                                    f'{m["score"]:.0f}% match &nbsp;·&nbsp; '
                                    f'<a href="{src}" target="_blank" style="color:#4F46E5;">{src}</a>'
                                    f'</span></div>',
                                    unsafe_allow_html=True,
                                )

    # ── export ────────────────────────────────────────────────────────────────
    st.divider()
    export = rdf.drop(columns=["_matches", "_text"], errors="ignore").copy()
    export["risk_level"] = export["similarity_to_corpus_%"].apply(
        lambda s: "Duplicate" if s >= dup_threshold else "Review" if s >= 40 else "Unique"
    )
    fname = f"plag_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    dl_col, cap_col = st.columns([2, 5], vertical_alignment="center")
    dl_col.download_button(
        "Download results CSV",
        export.to_csv(index=False).encode("utf-8"),
        fname, "text/csv", type="primary", use_container_width=True,
    )
    cap_col.markdown(
        f'<p style="font-size:.78rem;color:#9CA3AF;margin:0;">'
        f'url · word_count · similarity_to_corpus_% · verdict · matched_existing_url · risk_level'
        f'{"  ·  web_plag_score · web_verdict · top_web_sources" if run_web else ""}</p>',
        unsafe_allow_html=True,
    )
