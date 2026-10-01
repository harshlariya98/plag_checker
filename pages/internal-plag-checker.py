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
/* ════════════════════════════════════════════════
   KollegeApply × Gen-Z design system
   Navy #16324F · Coral #F47062 · Blue #408EE0
   Purple #7C3AED · Green #10B981
════════════════════════════════════════════════ */
:root {
    --navy:    #16324F;
    --coral:   #F47062;
    --blue:    #408EE0;
    --purple:  #7C3AED;
    --green:   #10B981;
    --navy-10: rgba(22,50,79,.1);
    --navy-06: rgba(22,50,79,.06);
    --g-hero:  linear-gradient(135deg, #16324F 0%, #1e4a8a 60%, #2d5da6 100%);
    --g-coral: linear-gradient(135deg, #F47062 0%, #ff8c42 100%);
    --g-purple:linear-gradient(135deg, #7C3AED 0%, #a78bfa 100%);
    --g-green: linear-gradient(135deg, #10B981 0%, #34d399 100%);
    --g-red:   linear-gradient(135deg, #EF4444 0%, #f87171 100%);
}

@import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', -apple-system, sans-serif !important;
    background: #F0F4F8 !important;
}

.block-container {
    padding: 0 2rem 4rem !important;
    max-width: 1280px !important;
}

/* ── keyframes ─────────────────────────────── */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse-dot {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,.6); }
    50%     { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes spin-in {
    from { transform: rotate(-15deg) scale(.8); opacity:0; }
    to   { transform: rotate(0deg)  scale(1);  opacity:1; }
}

/* ── hero banner ─────────────────────────── */
.hero {
    background: var(--g-hero);
    border-radius: 0 0 24px 24px;
    padding: 2rem 2.5rem 2rem;
    margin: 0 -2rem 2rem;
    animation: fadeUp .5s ease both;
    position: relative; overflow: hidden;
}
.hero::before {
    content:'';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: rgba(255,255,255,.04);
}
.hero::after {
    content:'';
    position: absolute; bottom: -80px; right: 120px;
    width: 180px; height: 180px; border-radius: 50%;
    background: rgba(244,112,98,.15);
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.2);
    color: rgba(255,255,255,.9); font-size: .7rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 99px; margin-bottom: .75rem;
}
.hero h1 {
    font-size: 1.8rem; font-weight: 900; color: #fff;
    margin: 0 0 .4rem; letter-spacing: -.4px; line-height: 1.2;
}
.hero h1 span { color: #F47062; }
.hero p { font-size: .9rem; color: rgba(255,255,255,.75); margin: 0; line-height: 1.7; max-width: 560px; }

/* ── sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid rgba(22,50,79,.1) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
.sb-brand {
    display: flex; align-items: center; gap: .5rem;
    padding: .5rem 1rem 1rem;
}
.sb-brand-dot {
    width: 28px; height: 28px; border-radius: 8px;
    background: var(--g-hero);
    display: flex; align-items: center; justify-content: center;
    font-size: .85rem;
}
.sb-brand-name { font-size: .78rem; font-weight: 900; color: var(--navy); letter-spacing: -.2px; }
.sb-brand-sub  { font-size: .65rem; color: #9CA3AF; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
section[data-testid="stSidebar"] hr { border-color: rgba(22,50,79,.1) !important; }
section[data-testid="stSidebar"] label { color: #374151 !important; font-size: .82rem !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--coral) !important; }
section[data-testid="stSidebar"] p { color: #374151 !important; font-size: .85rem !important; }
section[data-testid="stSidebar"] small { color: #9CA3AF !important; }

/* ── corpus pill ─────────────────────────── */
.corpus-pill {
    display: inline-flex; align-items: center; gap: .5rem;
    background: #fff; border: 1.5px solid rgba(22,50,79,.12);
    border-radius: 99px; padding: .45rem .9rem;
    font-size: .82rem; font-weight: 700; color: var(--navy);
    margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(22,50,79,.06);
    animation: fadeUp .4s .2s ease both;
}
.corpus-pill .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); flex-shrink: 0;
    animation: pulse-dot 2s ease infinite;
}
.corpus-pill.error { border-color: rgba(239,68,68,.3); color: #991B1B; }
.corpus-pill.error .dot { background: #EF4444; animation: none; }
.corpus-pill strong { color: var(--coral); }

/* ── section heading ─────────────────────── */
.section-head {
    display: flex; align-items: center; gap: .75rem;
    margin: 2rem 0 .9rem;
    animation: fadeUp .4s ease both;
}
.section-num {
    width: 2rem; height: 2rem; border-radius: 10px;
    background: var(--g-hero); color: #fff;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: .8rem; font-weight: 900; flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(22,50,79,.3);
}
.section-title { font-size: 1rem; font-weight: 900; color: var(--navy); letter-spacing: -.2px; }
.section-sub { font-size: .78rem; color: #9CA3AF; margin-left: auto; font-weight: 700; }

/* ── callouts ────────────────────────────── */
.callout {
    display: flex; align-items: flex-start; gap: .6rem;
    background: rgba(64,142,224,.07); border: 1px solid rgba(64,142,224,.25);
    border-radius: 10px; padding: .75rem 1rem;
    font-size: .83rem; color: #1a4a7a; line-height: 1.6;
    margin: .5rem 0 1rem;
}
.callout-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.callout.warn  { background: rgba(244,112,98,.07); border-color: rgba(244,112,98,.3); color: #7a2a1e; }
.callout.ok    { background: rgba(16,185,129,.07); border-color: rgba(16,185,129,.3); color: #065f46; }

/* ── upload hint ─────────────────────────── */
.upload-hint {
    font-size: .82rem; color: #9CA3AF;
    margin: .5rem 0 0; line-height: 1.6;
}

/* ── build steps ─────────────────────────── */
.build-step-title {
    font-size: .75rem; font-weight: 900; letter-spacing: .08em;
    text-transform: uppercase; color: var(--navy); opacity: .55;
    margin-bottom: .4rem;
}
.eta-label { font-size: .8rem; color: #9CA3AF; margin: .25rem 0 0; }

/* ── stat tiles ──────────────────────────── */
.stat-row { display: flex; gap: .875rem; margin: 1rem 0 1.25rem; flex-wrap: wrap; }
.stat-tile {
    flex: 1; min-width: 110px;
    background: #fff; border-radius: 14px;
    padding: 1rem 1rem .85rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(22,50,79,.07);
    transition: transform .2s, box-shadow .2s;
    animation: fadeUp .4s ease both;
    border: 1.5px solid transparent;
}
.stat-tile:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(22,50,79,.12); }
.stat-tile .sv {
    font-size: 1.8rem; font-weight: 900; line-height: 1.05;
    background: var(--g-hero); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; background-clip: text;
}
.stat-tile.red   .sv { background: var(--g-red);    -webkit-background-clip: text; background-clip: text; }
.stat-tile.amber .sv { background: var(--g-coral);  -webkit-background-clip: text; background-clip: text; }
.stat-tile.green .sv { background: var(--g-green);  -webkit-background-clip: text; background-clip: text; }
.stat-tile.brand .sv { background: var(--g-coral);  -webkit-background-clip: text; background-clip: text; }
.stat-tile.purple .sv{ background: var(--g-purple); -webkit-background-clip: text; background-clip: text; }
.stat-tile .sl {
    font-size: .65rem; font-weight: 900; letter-spacing: .07em;
    text-transform: uppercase; color: #9CA3AF; margin-top: .3rem;
}
.stat-tile.red   { border-color: rgba(239,68,68,.15); }
.stat-tile.green { border-color: rgba(16,185,129,.2); }
.stat-tile.amber { border-color: rgba(244,112,98,.2); }

/* ── alert banners ───────────────────────── */
.alert-banner {
    display: flex; align-items: center; gap: .75rem;
    border-radius: 12px; padding: .9rem 1.1rem;
    font-size: .875rem; font-weight: 700;
    margin: .75rem 0;
    animation: fadeUp .3s ease both;
}
.alert-banner.bad {
    background: linear-gradient(135deg,rgba(239,68,68,.1),rgba(244,112,98,.08));
    border: 1.5px solid rgba(239,68,68,.25); color: #991B1B;
}
.alert-banner.ok {
    background: linear-gradient(135deg,rgba(16,185,129,.09),rgba(52,211,153,.06));
    border: 1.5px solid rgba(16,185,129,.25); color: #065f46;
}
.alert-icon { font-size: 1.1rem; }

/* ── result cards ────────────────────────── */
.rc {
    background: #fff; border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: .5rem 0; display: flex; gap: 1rem; align-items: flex-start;
    transition: transform .2s, box-shadow .2s;
    border: 1.5px solid transparent;
    box-shadow: 0 2px 8px rgba(22,50,79,.06);
    animation: fadeUp .35s ease both;
}
.rc:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(22,50,79,.1); }
.rc.danger { border-color: rgba(239,68,68,.25); background: linear-gradient(to right,rgba(239,68,68,.03),#fff 40%); }
.rc.warn   { border-color: rgba(244,112,98,.25); background: linear-gradient(to right,rgba(244,112,98,.04),#fff 40%); }
.rc.ok     { border-color: rgba(16,185,129,.2);  background: linear-gradient(to right,rgba(16,185,129,.04),#fff 40%); }

.rc-pct-wrap {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-width: 4rem; gap: .15rem;
}
.rc-pct {
    font-size: 1.5rem; font-weight: 900; line-height: 1;
}
.rc.danger .rc-pct { background: var(--g-red);    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.rc.warn   .rc-pct { background: var(--g-coral);  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.rc.ok     .rc-pct { background: var(--g-green);  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }

.rc-bar-bg { width: 3rem; height: 4px; background: #f0f0f0; border-radius: 99px; overflow: hidden; }
.rc-bar    { height: 4px; border-radius: 99px; transition: width .6s ease; }
.rc.danger .rc-bar { background: var(--g-red); }
.rc.warn   .rc-bar { background: var(--g-coral); }
.rc.ok     .rc-bar { background: var(--g-green); }

.rc-body { flex: 1; font-size: .84rem; color: #374151; line-height: 1.65; }
.rc-body a { color: var(--blue); text-decoration: none; }
.rc-body a:hover { text-decoration: underline; }
.rc-label {
    font-size: .65rem; font-weight: 900; letter-spacing: .07em;
    text-transform: uppercase; color: #CBD5E1; margin-bottom: .15rem;
}
.rc-badge {
    display: inline-flex; align-items: center; gap: .25rem;
    padding: 2px 9px; border-radius: 99px; font-size: .72rem; font-weight: 900;
    margin-left: .35rem; vertical-align: middle; letter-spacing: .02em;
}
.rc-badge.danger { background: linear-gradient(135deg,#fee2e2,#fecaca); color: #991B1B; }
.rc-badge.warn   { background: linear-gradient(135deg,rgba(244,112,98,.15),rgba(255,140,66,.12)); color: #7a2a1e; }
.rc-badge.ok     { background: linear-gradient(135deg,#d1fae5,#a7f3d0); color: #065f46; }
.rc-meta { font-size: .75rem; color: #CBD5E1; margin-top: .3rem; }

/* ── tabs ────────────────────────────────── */
div[data-testid="stSegmentedControl"] { margin: .75rem 0 1rem; }

/* ── primary button ──────────────────────── */
.stButton > button[kind="primary"] {
    background: var(--g-hero) !important; border: none !important;
    font-weight: 900 !important; font-size: .95rem !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 16px rgba(22,50,79,.35) !important;
    border-radius: 12px !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(22,50,79,.45) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── secondary button ─────────────────────── */
.stButton > button:not([kind="primary"]) {
    border-radius: 10px !important; font-weight: 700 !important;
    border-color: rgba(22,50,79,.2) !important; color: var(--navy) !important;
}

/* ── download button ──────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: var(--g-coral) !important; border: none !important;
    color: #fff !important; font-weight: 900 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(244,112,98,.4) !important;
    letter-spacing: .02em !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(244,112,98,.5) !important;
}

/* ── links ────────────────────────────────── */
a { color: var(--blue) !important; }

/* ── progress bar ─────────────────────────── */
.stProgress > div > div > div {
    background: var(--g-coral) !important;
    border-radius: 99px !important;
}

/* ── data table / expander ────────────────── */
.stDataFrame  { border-radius: 14px !important; overflow: hidden;
                box-shadow: 0 2px 10px rgba(22,50,79,.06) !important; }
div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border-color: rgba(22,50,79,.1) !important;
}

/* ── detail section label ─────────────────── */
.detail-label {
    font-size: .7rem; font-weight: 900; letter-spacing: .08em;
    text-transform: uppercase; color: #CBD5E1;
    margin: 1.5rem 0 .6rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def section(n, title, sub=""):
    sub_html = f'<span class="section-sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="section-head">'
        f'<span class="section-num">{n}</span>'
        f'<span class="section-title">{title}</span>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

def callout(msg, kind="info", icon="💡"):
    kind_cls = {"info": "", "warn": "warn", "ok": "ok"}.get(kind, "")
    st.markdown(
        f'<div class="callout {kind_cls}">'
        f'<span class="callout-icon">{icon}</span>'
        f'<span>{msg}</span>'
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
    callout(
        "First-time setup — building the index now. "
        "This happens <b>once only</b>, then loads in ~10s every session after. ☕",
        kind="info", icon="🚀",
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
    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-brand-dot">🎓</div>'
        '<div><div class="sb-brand-name">KollegeApply</div>'
        '<div class="sb-brand-sub">Plag Checker</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**⚙️ Detection settings**")
    dup_threshold = st.slider("Duplicate threshold", 20, 100, 65,
                              format="%d%%",
                              help="Articles above this % similarity are flagged as duplicates")

    st.divider()
    st.markdown("**🌐 Web check** *(optional)*")
    run_web = st.checkbox("Check against open web", value=False)
    if run_web:
        n_passages   = st.slider("Passages per article", 4, 20, 8)
        web_thresh   = st.slider("Match threshold", 70, 100, 85, format="%d%%")
        own_domain   = st.text_input("Your domain (excluded)", "kollegeapply.com")
        excl_domains = st.text_area("Other excluded domains",
                                    "wikipedia.org\nyoutube.com", height=60)

    st.divider()
    top_n = st.number_input("Max results shown", 10, 5000, 200)

    st.divider()
    st.markdown(
        '<small>'
        '<b style="color:#10B981">●</b> &lt;40% original &nbsp;'
        '<b style="color:#F47062">●</b> 40–65% review &nbsp;'
        '<b style="color:#EF4444">●</b> ≥threshold duplicate'
        '</small>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow">🔍 Internal Tool</div>'
    '<h1>Plag <span>Checker</span> ✨</h1>'
    '<p>Drop your CSV, we\'ll catch the copycats. Checks new articles against '
    'the full KollegeApply database — similarity scores, closest match, and '
    'everything you need to make the call.</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-load corpus
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CORPUS_PATH):
    st.markdown(
        '<div class="corpus-pill error"><span class="dot"></span>'
        f'Corpus not found at <code>{CORPUS_PATH}</code></div>',
        unsafe_allow_html=True,
    )
    st.stop()

_ensure_corpus_index()

corpus_urls = st.session_state["corpus_urls"]
corpus_vec  = st.session_state["corpus_vec"]
corpus_mat  = st.session_state["corpus_mat"]
n_corp      = len(corpus_urls)

st.markdown(
    f'<div class="corpus-pill"><span class="dot"></span>'
    f'Corpus ready &nbsp;·&nbsp; <strong>{n_corp:,} articles</strong> indexed &nbsp;⚡</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
section(1, "Upload articles to check", "CSV or paste URLs")

new_df = None
new_src = st.radio("Input method", ["📂  Upload CSV", "🔗  Paste URLs"],
                   horizontal=True, key="new_src",
                   label_visibility="collapsed")

with st.container(border=True):
    if new_src == "📂  Upload CSV":
        new_file = st.file_uploader(
            "Drop your CSV here — or click to browse",
            type=["csv"], key="new_upload",
            label_visibility="visible",
        )
        st.markdown(
            '<p class="upload-hint">Need: <code>url</code> column &nbsp;·&nbsp; '
            'Bonus: <code>description</code> column (HTML saves fetch time) 🚀</p>',
            unsafe_allow_html=True,
        )
        if new_file:
            try:
                raw_new = pd.read_csv(new_file)
            except Exception as e:
                st.error(f"Couldn't read that CSV: {e}")
                raw_new = None
            if raw_new is not None:
                df2, err2 = normalise_new(raw_new)
                if err2:
                    st.warning(f"🤔 Column mapping needed — {err2}")
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
                    parts = [f"✅ **{len(new_df):,}** articles loaded"]
                    if has_desc: parts.append(f"{has_desc:,} have HTML content")
                    if no_desc:  parts.append(f"{no_desc:,} will be fetched live")
                    st.success("  ·  ".join(parts))
    else:
        pasted = st.text_area("One URL per line",
                              height=140,
                              placeholder="https://www.kollegeapply.com/article/…\nhttps://…")
        if pasted.strip():
            urls_list = [u.strip() for u in pasted.splitlines()
                         if u.strip().startswith("http")]
            if urls_list:
                new_df = pd.DataFrame({"url": urls_list, "description": ""})
                st.success(f"✅ **{len(urls_list)}** URLs queued — content will be fetched live")
            else:
                st.error("No valid URLs found. Each line should start with http.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Run
# ─────────────────────────────────────────────────────────────────────────────
if new_df is not None and len(new_df) > 0:
    section(2, "Ready to launch 🚀", "review before running")

    n_new   = len(new_df)
    n_fetch = int((new_df["description"].str.strip().str.len() <= 10).sum())
    est_sec = n_new * 0.5 + n_fetch * 4
    if run_web:
        est_sec += n_new * n_passages * 5

    est_label = (f"~{max(1,round(est_sec/60))} min" if est_sec > 60
                 else f"~{int(est_sec)}s")
    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-tile brand"><div class="sv">{n_new}</div><div class="sl">New articles</div></div>'
        f'<div class="stat-tile"><div class="sv">{n_corp:,}</div><div class="sl">Corpus</div></div>'
        f'<div class="stat-tile amber"><div class="sv">{n_fetch}</div><div class="sl">To fetch</div></div>'
        f'<div class="stat-tile purple"><div class="sv">{est_label}</div><div class="sl">Est. time</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_fetch > 0:
        callout(
            f"<b>{n_fetch} article{'s' if n_fetch>1 else ''}</b> missing HTML content — "
            "will be fetched live (slower). Add a <code>description</code> column to skip this.",
            kind="warn", icon="⚡",
        )

    run_btn = st.button("🔍  Run plagiarism check", type="primary",
                        use_container_width=True, key="run_btn")

    if run_btn:
        # prepare new articles
        with st.status("📥 Preparing articles…", expanded=True) as s2:
            prog = st.progress(0.0)
            cur  = st.empty()
            new_texts, new_wc = [], []
            for i, row in enumerate(new_df.itertuples(), 1):
                cur.markdown(
                    f'<p style="font-size:.82rem;color:#6B7280;margin:0;">'
                    f'<b>{i}/{n_new}</b> &nbsp;·&nbsp; {row.url}</p>',
                    unsafe_allow_html=True,
                )
                desc = str(row.description).strip()
                text = _fast_strip(desc) if len(desc) > 10 else fetch_text(row.url)
                new_texts.append(text)
                new_wc.append(len(text.split()))
                prog.progress(i / n_new)
            cur.empty()
            s2.update(label=f"✅ {n_new} articles ready",
                      state="complete", expanded=False)

        # similarity
        with st.status("⚡ Computing similarity…", expanded=False) as s3:
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
            s3.update(label="⚡ Similarity computed", state="complete")

        # optional web check
        web_res = {}
        if run_web:
            excl = {d.strip().lower() for d in excl_domains.splitlines() if d.strip()}
            excl.add(own_domain.strip().lower().replace("www.", ""))
            with st.status(f"🌐 Web check — {n_new} articles…", expanded=True) as sw:
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
                sw.update(label="🌐 Web check complete", state="complete", expanded=False)

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
        callout(
            "Upload a CSV or paste URLs above to get started. "
            "We handle the heavy lifting 💪",
            kind="info", icon="👆",
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
        section(3, "Results are in 🎯")
    with h2:
        if st.button("🗑 Clear", use_container_width=True):
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
        f'<div class="stat-tile"><div class="sv">{len(rdf)}</div><div class="sl">Total checked</div></div>'
        f'<div class="stat-tile"><div class="sv">{avg_s}</div><div class="sl">Avg similarity</div></div>'
        f'<div class="stat-tile red"><div class="sv">{n_dup}</div><div class="sl">🚨 Duplicates</div></div>'
        f'<div class="stat-tile amber"><div class="sv">{n_rev}</div><div class="sl">👀 Review</div></div>'
        f'<div class="stat-tile green"><div class="sv">{n_ok}</div><div class="sl">✅ Original</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_dup:
        st.markdown(
            f'<div class="alert-banner bad">'
            f'<span class="alert-icon">🚨</span>'
            f'<span><b>{n_dup} article{"s" if n_dup>1 else ""}</b> '
            f'{"are" if n_dup>1 else "is"} ≥{dup_threshold}% similar to existing content — '
            f'review before publishing</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif results:
        st.markdown(
            '<div class="alert-banner ok">'
            '<span class="alert-icon">🎉</span>'
            '<span>All clear! Every article is below the duplicate threshold.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # view tabs
    view = st.segmented_control(
        "View", ["📋  All results", "🚨  Duplicates only", "🌐  Web plagiarism"],
        default="📋  All results", label_visibility="collapsed",
    ) or "📋  All results"

    # ── table + cards ─────────────────────────────────────────────────────────
    if view in ("📋  All results", "🚨  Duplicates only"):
        show = results if view == "📋  All results" else [
            r for r in results if r["similarity_to_corpus_%"] >= dup_threshold
        ]
        show = sorted(show, key=lambda r: r["similarity_to_corpus_%"], reverse=True)
        show = show[:int(top_n)]

        if not show:
            callout(
                "No duplicates found at this threshold — you're all good! 🎉",
                kind="ok", icon="✅",
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
                    '<p class="detail-label">🔍 Detail cards — flagged articles</p>',
                    unsafe_allow_html=True,
                )
                for r in flagged:
                    sc  = r["similarity_to_corpus_%"]
                    cls, vlabel = verdict_for(sc, dup_threshold)[:2]
                    icon = "🚨" if cls == "danger" else "👀"
                    new_e   = html_module.escape(r["url"])
                    corp_u  = html_module.escape(r["matched_existing_url"], quote=True)
                    corp_ue = html_module.escape(r["matched_existing_url"])
                    bar_w   = min(int(sc), 100)
                    st.markdown(
                        f'<div class="rc {cls}">'
                        f'<div class="rc-pct-wrap">'
                        f'<div class="rc-pct">{sc:.0f}%</div>'
                        f'<div class="rc-bar-bg"><div class="rc-bar" style="width:{bar_w}%"></div></div>'
                        f'</div>'
                        f'<div class="rc-body">'
                        f'<div class="rc-label">New article</div>'
                        f'<a href="{new_e}" target="_blank">{new_e}</a>'
                        f'<span class="rc-badge {cls}">{icon} {vlabel}</span>'
                        f'<div class="rc-label" style="margin-top:.65rem;">Closest match in corpus</div>'
                        f'<a href="{corp_u}" target="_blank">{corp_ue}</a>'
                        f'<div class="rc-meta">{r["word_count"]:,} words &nbsp;·&nbsp; {sc:.1f}% similar</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

    # ── web plagiarism ────────────────────────────────────────────────────────
    else:
        if not run_web:
            callout(
                "Enable <b>Check against open web</b> in the sidebar, then re-run to see web results. 🌐",
                kind="info", icon="💡",
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
                                    f'<a href="{src}" target="_blank" style="color:#408EE0;">{src}</a>'
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
        "📥  Download CSV report",
        export.to_csv(index=False).encode("utf-8"),
        fname, "text/csv", use_container_width=True,
    )
    cap_col.markdown(
        f'<p style="font-size:.78rem;color:#CBD5E1;margin:0;">'
        f'url · words · similarity · verdict · matched_url · risk_level'
        f'{"  ·  web_score · web_verdict · top_sources" if run_web else ""}</p>',
        unsafe_allow_html=True,
    )
