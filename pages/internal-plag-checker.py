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

import base64

import joblib
import pandas as pd
import requests as _req
import streamlit as st
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from plag_utils import fetch_text, check_article, clear_page_cache, build_passages, normalize
from rapidfuzz import fuzz

@st.cache_resource(show_spinner=False)
def _load_embedder():
    """Load sentence-transformers model once. Returns (model, True) or (None, False)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, True
    except Exception:
        return None, False

_EMBEDDER, _EMBED_AVAILABLE = _load_embedder()

@st.cache_data(show_spinner=False)
def _logo_b64():
    try:
        r = _req.get("https://www.kollegeapply.com/new-logo.svg", timeout=5)
        if r.ok:
            return "data:image/svg+xml;base64," + base64.b64encode(r.content).decode()
    except Exception:
        pass
    return ""

LOGO_URI = _logo_b64()

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
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Design tokens ─────────────────────────────────────────────────────────── */
:root {
    --navy:    #16324F;
    --coral:   #F47062;
    --blue:    #408EE0;
    --purple:  #7C3AED;
    --green:   #10B981;
    --amber:   #F59E0B;
    --red:     #EF4444;

    --surface: #FFFFFF;
    --bg:      #F4F6F9;
    --border:  #E2E8F0;
    --text:    #1E293B;
    --muted:   #64748B;
    --light:   #F8FAFC;

    --green-bg: #F0FDF9; --green-border: #A7F3D0; --green-text: #065F46;
    --amber-bg: #FFFBEB; --amber-border: #FCD34D; --amber-text: #78350F;
    --red-bg:   #FEF2F2; --red-border:   #FECACA; --red-text:   #991B1B;
    --blue-bg:  #EFF6FF; --blue-border:  #BFDBFE; --blue-text:  #1E40AF;
    --purple-bg:#F5F3FF; --purple-border:#DDD6FE; --purple-text:#5B21B6;

    --shadow-xs: 0 1px 2px rgba(0,0,0,.05);
    --shadow-sm: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.04);

    --radius:    12px;
    --radius-sm:  8px;
    --radius-xs:  6px;
}

/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.block-container { padding: 4rem 2rem 4rem !important; max-width: 1440px !important; }
/* Collapse any flow-space left by the fixed header's markdown wrapper */
div[data-testid="stMarkdown"]:has(.app-header) {
    height: 0 !important; overflow: visible !important;
    margin: 0 !important; padding: 0 !important;
    line-height: 0 !important;
}

/* Streamlit top bar — hide it; we render our own fixed header */
header[data-testid="stHeader"] { display: none !important; }

/* ── Sidebar collapse button — label it "Navigation" ───────────────────────── */
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important; align-items: center !important;
    background: var(--light) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xs) !important;
    padding: 5px 10px 5px 7px !important;
    gap: 5px !important; cursor: pointer !important;
    transition: all .15s !important;
}
button[data-testid="stSidebarCollapseButton"]:hover {
    background: var(--bg) !important; border-color: var(--blue) !important;
}
button[data-testid="stSidebarCollapseButton"] svg {
    width: 14px !important; height: 14px !important;
    color: var(--navy) !important;
}
button[data-testid="stSidebarCollapseButton"]::after {
    content: "Navigation";
    font-family: 'Inter', sans-serif;
    font-size: 11.5px; font-weight: 600; color: var(--navy);
}

/* ── Keyframes ─────────────────────────────────────────────────────────────── */
@keyframes fadeUp  { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
@keyframes fadeIn  { from{opacity:0} to{opacity:1} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Hide sidebar entirely (nav moves to header) ───────────────────────────── */
section[data-testid="stSidebar"] { display: none !important; }

/* ── Fixed top accent strip ────────────────────────────────────────────────── */
.header-accent {
    position: fixed; top: 0; left: 0; right: 0; height: 4px; z-index: 1000000;
    background: linear-gradient(90deg, var(--coral) 0%, var(--blue) 50%, var(--purple) 100%);
}

/* ── Fixed app header ──────────────────────────────────────────────────────── */
.app-header {
    position: fixed; top: 4px; left: 0; right: 0; height: 54px; z-index: 999999;
    background: rgba(255,255,255,0.98);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 2px 16px rgba(22,50,79,.06);
    display: flex; align-items: center;
    padding: 0 28px; gap: 0;
}

/* ── Left: brand block ─────────────────────────────────────────────────────── */
.app-header-left { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.app-logo    { width: 26px; height: 26px; object-fit: contain; flex-shrink: 0; display: block; }
.app-logo-fb {
    width: 26px; height: 26px; border-radius: 7px;
    background: linear-gradient(135deg, var(--navy) 0%, #2a5080 100%);
    display: flex; align-items: center; justify-content: center; font-size: .85rem;
    box-shadow: 0 2px 6px rgba(22,50,79,.25);
}
.app-brand-block { display: flex; flex-direction: column; gap: 0; }
.app-brand {
    font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    color: var(--muted); line-height: 1;
}
.app-page  {
    font-size: 14px; font-weight: 800; color: var(--navy); line-height: 1.2;
    letter-spacing: -.3px;
}
.app-sep   { width: 1px; height: 22px; background: var(--border); flex-shrink: 0; margin: 0 4px; }
.app-badge {
    font-size: 9.5px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase;
    color: var(--coral); background: rgba(244,112,98,.08);
    border: 1px solid rgba(244,112,98,.22);
    padding: 3px 8px; border-radius: 5px; align-self: center;
}

/* ── Center: nav tabs ──────────────────────────────────────────────────────── */
.app-nav {
    display: flex; align-items: stretch; justify-content: center;
    gap: 0; flex: 1; padding: 0 32px; height: 100%;
}
.nav-tab {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0 18px; height: 100%;
    font-size: 12.5px; font-weight: 600;
    color: var(--muted);
    border-bottom: 2.5px solid transparent;
    transition: color .15s, border-color .15s;
    cursor: default; white-space: nowrap; user-select: none;
    position: relative;
}
.nav-tab:hover { color: var(--navy); }
.nav-tab.active { color: var(--navy); border-bottom-color: var(--blue); }
.nav-tab.active-green  { color: var(--green);  border-bottom-color: var(--green);  }
.nav-tab.active-purple { color: var(--purple); border-bottom-color: var(--purple); }
.nav-tab-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
    animation: pulse 1.8s ease infinite;
}

/* ── Right: corpus status ──────────────────────────────────────────────────── */
.app-header-right { flex-shrink: 0; }
.corpus-chip {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 12px; font-weight: 500; color: var(--muted);
    background: var(--light); border: 1px solid var(--border);
    border-radius: 8px; padding: 5px 12px 5px 10px;
    box-shadow: var(--shadow-xs);
}
.corpus-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--green);
    flex-shrink: 0; animation: pulse 1.8s ease infinite;
}
.corpus-chip strong { color: var(--navy); font-weight: 700; }
.corpus-count { font-size: 12.5px; font-weight: 700; color: var(--blue); }
.corpus-chip.err .corpus-dot { background: var(--red); animation: none; }
.corpus-chip.err { color: var(--red-text); }

/* Backward-compat aliases used elsewhere */
.nav-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; animation: pulse 1.8s ease infinite; }

/* ── Page intro ────────────────────────────────────────────────────────────── */
.page-intro { margin: 0 0 12px !important; animation: fadeUp .3s ease both; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.page-intro-text { flex: 1; min-width: 220px; }
.page-intro h2 {
    font-size: 18px; font-weight: 800; color: var(--navy);
    margin: 0 0 2px; letter-spacing: -.4px;
}
.page-intro p { font-size: 12px; color: var(--muted); margin: 0; line-height: 1.5; }
.page-intro-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--blue-bg); border: 1px solid var(--blue-border);
    border-radius: var(--radius-sm); padding: 5px 11px;
    font-size: 11px; font-weight: 700; color: var(--blue-text);
    white-space: nowrap; flex-shrink: 0;
}
.page-intro-badge strong { font-size: 13px; color: var(--blue); }
.pib-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse 1.8s ease infinite; flex-shrink: 0; }

/* ── Panel card (Streamlit bordered container) ─────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 12px !important;
    animation: fadeUp .35s ease both;
    transition: box-shadow .2s, border-color .2s !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--shadow-md) !important;
}

/* ── Primary run button ────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--coral) 0%, #E85D4E 100%) !important;
    border: none !important; color: #fff !important;
    font-size: 14px !important; font-weight: 700 !important;
    padding: 10px 28px !important; border-radius: var(--radius-sm) !important;
    box-shadow: 0 2px 8px rgba(244,112,98,.35) !important;
    transition: all .2s !important; letter-spacing: .01em !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(244,112,98,.5) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 6px rgba(244,112,98,.3) !important;
}

.panel-hd {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 14px; margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
}
.panel-num {
    width: 26px; height: 26px; border-radius: 7px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; color: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,.15);
}
.panel-num.blue   { background: linear-gradient(135deg, var(--blue)   0%, #5BA3EC 100%); }
.panel-num.coral  { background: linear-gradient(135deg, var(--coral)  0%, #E85D4E 100%); }
.panel-num.green  { background: linear-gradient(135deg, var(--green)  0%, #34D399 100%); }
.panel-num.purple { background: linear-gradient(135deg, var(--purple) 0%, #9F67F5 100%); }
.panel-title { font-size: 14.5px; font-weight: 700; color: var(--navy); }
.panel-sub   { font-size: 11.5px; color: var(--muted); margin-left: auto; font-weight: 500; }

/* ── Top filter accordion ────────────────────────────────────────────────────── */
.filter-label {
    font-size: 10.5px; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: var(--navy); margin-bottom: 8px;
    display: flex; align-items: center; gap: 5px;
}
.filter-sub { font-size: 11px; color: var(--muted); margin-top: 3px; line-height: 1.5; }

/* Style the Streamlit expander to look like a filter bar toggle */
div[data-testid="stExpander"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 12px !important;
    animation: fadeUp .3s ease both;
}
div[data-testid="stExpander"] summary {
    padding: 10px 16px !important;
    font-size: 12px !important; font-weight: 700 !important;
    color: var(--navy) !important; letter-spacing: .01em !important;
}
div[data-testid="stExpander"] summary:hover { background: var(--bg) !important; border-radius: var(--radius) !important; }
div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    padding: 4px 16px 16px !important;
    border-top: 1px solid var(--border) !important;
}

/* Upload success state */
.upload-ok {
    background: var(--green-bg); border: 1px solid var(--green-border);
    border-radius: var(--radius-sm); padding: 8px 12px; margin-top: 8px;
    animation: fadeUp .3s ease both;
}
.upload-ok-row { display: flex; gap: 20px; margin-bottom: 6px; align-items: center; }
.upload-stat { display: flex; flex-direction: column; gap: 0; }
.upload-big { font-size: 17px; font-weight: 800; color: var(--navy); line-height: 1.15; letter-spacing: -.3px; }
.upload-big.green { color: var(--green); }
.upload-big.amber { color: #D97706; }
.upload-lbl { font-size: 9.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.upload-cov-bar-wrap { height: 4px; background: var(--green-border); border-radius: 99px; overflow: hidden; margin-bottom: 3px; }
.upload-cov-bar { height: 4px; background: var(--green); border-radius: 99px; transition: width .4s ease; }
.upload-cov-lbl { font-size: 10px; color: var(--green-text); font-weight: 600; }
.upload-warn { font-size: 10px; color: #D97706; font-weight: 600; margin-top: 4px; }

/* ── Tabs (replacing radio) ────────────────────────────────────────────────── */
div[data-testid="stTabs"] [data-testid="stTabsContainer"] {
    gap: 0 !important; border-bottom: 1px solid var(--border) !important;
    background: transparent !important; margin-bottom: 16px !important;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 13px !important; font-weight: 600 !important;
    color: var(--muted) !important; padding: 8px 14px !important;
    border: none !important; border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: color .15s, border-color .15s !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--navy) !important;
    border-bottom-color: var(--blue) !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover { color: var(--navy) !important; }

/* ── File uploader ─────────────────────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--light) !important;
    transition: border-color .2s, background .2s !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--blue) !important;
    background: var(--blue-bg) !important;
}

/* ── Schema badges ─────────────────────────────────────────────────────────── */
.schema-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 10px 0 3px; }
.schema-group { display: flex; align-items: center; gap: 5px; }
.schema-tag {
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
}
.schema-key {
    font-family: 'SF Mono', 'Consolas', 'Menlo', monospace;
    font-size: 11.5px; font-weight: 600;
    padding: 2px 8px; border-radius: 4px;
    background: rgba(22,50,79,.06); color: var(--navy);
    border: 1px solid rgba(22,50,79,.12);
}
.schema-key.req {
    background: var(--blue-bg); color: var(--blue); border-color: var(--blue-border);
}
.schema-hint { font-size: 11.5px; color: var(--muted); margin: 4px 0 0; line-height: 1.55; }

/* ── Pre-run stat row ──────────────────────────────────────────────────────── */
.info-row { display: flex; gap: 8px; margin: 10px 0 8px; flex-wrap: wrap; }
.info-tile {
    flex: 1; min-width: 72px; background: var(--surface);
    border-radius: var(--radius-sm); padding: 10px 8px 8px;
    border: 1px solid var(--border);
    border-top: 3px solid var(--border);
    text-align: center;
    box-shadow: var(--shadow-xs);
    transition: transform .18s, box-shadow .18s;
    animation: fadeUp .4s ease both;
}
.info-tile:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }
.info-tile.blue   { border-top-color: var(--blue);   }
.info-tile.amber  { border-top-color: var(--amber);  }
.info-tile.purple { border-top-color: var(--purple); }
.info-tile.green  { border-top-color: var(--green);  }
.info-val { font-size: 18px; font-weight: 800; color: var(--navy); line-height: 1; letter-spacing: -.4px; }
.info-tile.blue   .info-val { color: var(--blue);   }
.info-tile.amber  .info-val { color: #D97706;        }
.info-tile.purple .info-val { color: var(--purple);  }
.info-tile.green  .info-val { color: var(--green);   }
.info-lbl { font-size: 9.5px; color: var(--muted); font-weight: 600; margin-top: 3px; text-transform: uppercase; letter-spacing: .04em; }

/* ── Summary stat tiles ────────────────────────────────────────────────────── */
.stat-row { display: flex; gap: 8px; margin: 0 0 16px; flex-wrap: wrap; }
.stat-tile {
    flex: 1; min-width: 90px; background: var(--surface);
    border-radius: var(--radius-sm); padding: 14px 12px 12px;
    text-align: center; border: 1px solid var(--border);
    border-top: 2.5px solid var(--border);
    box-shadow: var(--shadow-xs);
    transition: transform .18s, box-shadow .18s;
    animation: fadeUp .4s ease both;
}
.stat-tile:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }
.stat-val { font-size: 26px; font-weight: 800; color: var(--navy); line-height: 1; letter-spacing: -.6px; }
.stat-lbl { font-size: 10.5px; font-weight: 600; color: var(--muted); margin-top: 5px; text-transform: uppercase; letter-spacing: .04em; }
.stat-tile.red    { border-top-color: var(--red);    } .stat-tile.red    .stat-val { color: var(--red);    }
.stat-tile.amber  { border-top-color: var(--amber);  } .stat-tile.amber  .stat-val { color: #D97706;       }
.stat-tile.green  { border-top-color: var(--green);  } .stat-tile.green  .stat-val { color: var(--green);  }
.stat-tile.purple { border-top-color: var(--purple); } .stat-tile.purple .stat-val { color: var(--purple); }

/* ── Alert / callout ───────────────────────────────────────────────────────── */
.alert-strip {
    display: flex; align-items: flex-start; gap: 10px;
    border-radius: var(--radius-sm); padding: 12px 14px;
    font-size: 13px; margin: 12px 0;
    animation: fadeUp .3s ease both;
}
.alert-strip.danger  { background: var(--red-bg);   border: 1px solid var(--red-border);   color: var(--red-text);    }
.alert-strip.success { background: var(--green-bg); border: 1px solid var(--green-border); color: var(--green-text);  }
.alert-strip.warn    { background: var(--amber-bg); border: 1px solid var(--amber-border); color: var(--amber-text);  }
.alert-strip.info    { background: var(--blue-bg);  border: 1px solid var(--blue-border);  color: var(--blue-text);   }
.alert-icon { font-size: 15px; flex-shrink: 0; margin-top: 1px; }
.alert-strip b { font-weight: 700; }

.callout {
    display: flex; align-items: flex-start; gap: 8px;
    border-radius: var(--radius-sm); padding: 10px 12px;
    font-size: 13px; line-height: 1.6; margin: 8px 0;
    background: var(--blue-bg); border: 1px solid var(--blue-border); color: var(--blue-text);
}
.callout.warn { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber-text); }
.callout.ok   { background: var(--green-bg); border-color: var(--green-border); color: var(--green-text); }
.callout-icon { font-size: 14px; flex-shrink: 0; line-height: 1.65; }

/* ── Result cards ──────────────────────────────────────────────────────────── */
.rc {
    background: var(--surface); border-radius: var(--radius-sm);
    padding: 13px 15px; margin: 5px 0;
    display: flex; gap: 14px; align-items: flex-start;
    border: 1px solid var(--border); border-left: 3px solid var(--border);
    box-shadow: var(--shadow-xs);
    transition: box-shadow .18s, transform .18s;
    animation: fadeUp .35s ease both;
}
.rc:hover { box-shadow: var(--shadow-sm); transform: translateY(-1px); }
.rc.danger { border-left-color: var(--red);   }
.rc.warn   { border-left-color: var(--amber); }
.rc.info   { border-left-color: var(--blue);  }
.rc.ok     { border-left-color: var(--green); }
.rc-score { display: flex; flex-direction: column; align-items: center; gap: 4px; min-width: 56px; }
.rc-pct { font-size: 19px; font-weight: 800; line-height: 1; }
.rc.danger .rc-pct { color: var(--red);   }
.rc.warn   .rc-pct { color: #D97706;      }
.rc.info   .rc-pct { color: var(--blue);  }
.rc.ok     .rc-pct { color: var(--green); }
.rc-bar-bg { width: 40px; height: 3px; background: var(--border); border-radius: 99px; overflow: hidden; }
.rc-bar    { height: 3px; border-radius: 99px; }
.rc.danger .rc-bar { background: var(--red);   }
.rc.warn   .rc-bar { background: var(--amber); }
.rc.info   .rc-bar { background: var(--blue);  }
.rc.ok     .rc-bar { background: var(--green); }
.rc-body { flex: 1; font-size: 13px; color: #374151; line-height: 1.65; min-width: 0; }
.rc-body a { color: var(--blue); text-decoration: none; overflow-wrap: break-word; word-break: break-all; font-size: 12.5px; }
.rc-body a:hover { text-decoration: underline; }
.rc-label { font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #94A3B8; margin-bottom: 2px; }
.rc-badge { display: inline-flex; align-items: center; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-left: 6px; vertical-align: middle; }
.rc-badge.danger { background: var(--red-bg);   color: var(--red-text);   }
.rc-badge.warn   { background: var(--amber-bg); color: var(--amber-text); }
.rc-badge.info   { background: var(--blue-bg);  color: var(--blue);       }
.rc-badge.ok     { background: var(--green-bg); color: var(--green-text); }
.rc-meta { font-size: 11px; color: #94A3B8; margin-top: 4px; }

/* ── Empty state ───────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center; padding: 42px 24px;
    background: var(--surface); border-radius: var(--radius);
    border: 1.5px dashed var(--border);
    animation: fadeUp .4s ease both;
}
.es-icon  { font-size: 28px; margin-bottom: 10px; opacity: .55; }
.es-title { font-size: 15px; font-weight: 700; color: var(--navy); margin-bottom: 6px; }
.es-sub   { font-size: 13px; color: var(--muted); line-height: 1.65; }

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: var(--navy) !important; border: none !important;
    font-weight: 700 !important; font-size: 14px !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 2px 8px rgba(22,50,79,.25) !important;
    transition: transform .15s, box-shadow .15s, background .15s !important;
    padding: .55rem 1.5rem !important; letter-spacing: .01em !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(22,50,79,.35) !important;
    background: #1d3f63 !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }
.stButton > button:not([kind="primary"]) {
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
    border-color: var(--border) !important; color: var(--text) !important;
    font-size: 13px !important; background: var(--surface) !important;
}
div[data-testid="stDownloadButton"] > button {
    background: var(--blue) !important; border: none !important;
    color: #fff !important; font-weight: 700 !important;
    border-radius: var(--radius-sm) !important; font-size: 13px !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(64,142,224,.35) !important;
}

/* ── Misc ──────────────────────────────────────────────────────────────────── */
a { color: var(--blue) !important; }
.stProgress > div > div > div { background: var(--blue) !important; border-radius: 99px !important; }
.stDataFrame { border-radius: var(--radius) !important; overflow: hidden !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow-xs) !important; }
div[data-testid="stExpander"] { border-radius: var(--radius-sm) !important; border-color: var(--border) !important; }
div[data-testid="stSegmentedControl"] { margin: .5rem 0 1rem !important; }
section[data-testid="stSidebar"] label { color: var(--text) !important; font-size: 12.5px !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: var(--text) !important; font-size: 12.5px !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--blue) !important; }
.upload-hint  { font-size: 12px; color: var(--muted); margin: 8px 0 0; }
.build-step-title { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
.eta-label { font-size: 12px; color: var(--muted); margin: 2px 0 0; }
.detail-label { font-size: 11px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: #94A3B8; margin: 20px 0 8px; }

/* ── Verdict guide ─────────────────────────────────────────────────────────── */
.vguide { display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 20px; }
.vg-item { flex: 1; min-width: 140px; border-radius: var(--radius-sm); padding: 12px 14px; border: 1px solid var(--border); }
.vg-item.ok     { background: var(--green-bg);  border-color: var(--green-border); }
.vg-item.info   { background: var(--blue-bg);   border-color: var(--blue-border);  }
.vg-item.warn   { background: var(--amber-bg);  border-color: #FDE68A;             }
.vg-item.danger { background: var(--red-bg);    border-color: #FECACA;             }
.vg-head { display: flex; align-items: center; gap: 7px; margin-bottom: 5px; }
.vg-dot  { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.vg-dot.ok { background: var(--green); } .vg-dot.info { background: var(--blue); }
.vg-dot.warn { background: #F59E0B; }   .vg-dot.danger { background: var(--red); }
.vg-title { font-size: 12px; font-weight: 700; }
.vg-item.ok .vg-title     { color: var(--green); }
.vg-item.info .vg-title   { color: var(--blue);  }
.vg-item.warn .vg-title   { color: #D97706;       }
.vg-item.danger .vg-title { color: var(--red);    }
.vg-action { font-size: 12.5px; font-weight: 700; color: var(--navy); margin-bottom: 3px; }
.vg-desc  { font-size: 11px; color: var(--muted); line-height: 1.45; }

/* ── Article result cards ──────────────────────────────────────────────────── */
.art-card { border: 1px solid var(--border); border-radius: var(--radius-sm); margin: 6px 0;
            background: #fff; overflow: hidden; transition: box-shadow .15s; }
.art-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.08); }
.art-card.danger { border-left: 4px solid var(--red);  }
.art-card.warn   { border-left: 4px solid #F59E0B;     }
.art-card.info   { border-left: 4px solid var(--blue); }
.art-card.ok     { border-left: 4px solid var(--green);}
.art-row { display: flex; align-items: center; gap: 16px; padding: 12px 16px; flex-wrap: wrap; }
.art-url-block { flex: 2; min-width: 160px; }
.art-url  { font-size: 12.5px; font-weight: 600; color: var(--blue); text-decoration: none;
            overflow-wrap: break-word; word-break: break-all; line-height: 1.4; }
.art-url:hover { text-decoration: underline; }
.art-wc   { font-size: 10.5px; color: var(--muted); margin-top: 2px; }
.art-scores { display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
.art-metric { display: flex; flex-direction: column; align-items: center; gap: 2px; min-width: 54px; }
.art-metric-val { font-size: 15px; font-weight: 800; line-height: 1; }
.art-metric-lbl { font-size: 9.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.art-meter-bg { width: 48px; height: 4px; background: var(--border); border-radius: 99px; overflow: hidden; margin-top: 2px; }
.art-meter-fill { height: 4px; border-radius: 99px; }
.art-verdict-col { flex-shrink: 0; min-width: 120px; }
.art-vbadge { display: inline-flex; align-items: center; gap: 5px; padding: 5px 12px;
              border-radius: 99px; font-size: 11.5px; font-weight: 700; white-space: nowrap; }
.art-vbadge.danger { background: var(--red-bg);   color: var(--red);   border: 1px solid #FECACA; }
.art-vbadge.warn   { background: var(--amber-bg); color: #B45309;      border: 1px solid #FDE68A; }
.art-vbadge.info   { background: var(--blue-bg);  color: var(--blue);  border: 1px solid var(--blue-border); }
.art-vbadge.ok     { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
.art-action-col { flex: 1; min-width: 180px; }
.art-action { font-size: 12px; font-weight: 600; line-height: 1.4; }
.art-action.danger { color: #DC2626; } .art-action.warn { color: #B45309; }
.art-action.info   { color: var(--blue); } .art-action.ok { color: var(--green); }
.art-matched { font-size: 10.5px; color: var(--muted); margin-top: 3px; }
.art-matched a { color: var(--muted); text-decoration: none; }
.art-matched a:hover { color: var(--blue); text-decoration: underline; }
.art-detail-body { padding: 14px 20px 16px; border-top: 1px solid var(--border);
                   background: #FAFAFA; font-size: 12.5px; color: #374151; }
.art-detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 24px; margin-top: 10px; }
.art-detail-item { background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; }
.art-detail-item .adi-lbl { font-size: 10px; font-weight: 700; text-transform: uppercase;
                             letter-spacing: .05em; color: #94A3B8; margin-bottom: 4px; }
.art-detail-item .adi-val { font-size: 13px; font-weight: 700; color: var(--navy); }
.art-detail-item .adi-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _panel_hd(n, title, sub="", color="blue"):
    sub_html = f'<span class="panel-sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="panel-hd">'
        f'<span class="panel-num {color}">{n}</span>'
        f'<span class="panel-title">{title}</span>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )

def callout(msg, kind="info", icon="💡"):
    kind_cls = {"info": "", "warn": "warn", "ok": "ok"}.get(kind, "")
    st.markdown(
        f'<div class="callout {kind_cls}">'
        f'<span class="callout-icon">{icon}</span>'
        f'<span>{msg}</span></div>',
        unsafe_allow_html=True,
    )

def alert_strip(msg, kind="info", icon=None):
    icons = {"danger": "🚨", "success": "✅", "warn": "⚠️", "info": "ℹ️"}
    i = icon or icons.get(kind, "ℹ️")
    st.markdown(
        f'<div class="alert-strip {kind}">'
        f'<span class="alert-icon">{i}</span>'
        f'<span>{msg}</span></div>',
        unsafe_allow_html=True,
    )

def smart_col(df, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None

def _detect_url_col(df):
    """Return the column most likely to contain URLs by sampling cell values."""
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        if len(sample) == 0:
            continue
        http_frac = sample.str.strip().str.startswith("http").mean()
        if http_frac >= 0.5:
            return col
    return None

def _detect_desc_col(df, url_col):
    """Return the column most likely to contain article body text/HTML."""
    best_col, best_len = None, 0
    for col in df.columns:
        if col == url_col:
            continue
        sample = df[col].dropna().astype(str).head(20)
        if len(sample) == 0:
            continue
        avg_len = sample.str.len().mean()
        if avg_len > best_len and avg_len >= 200:
            best_len, best_col = avg_len, col
    return best_col

def normalise_new(df):
    # 1. Name-based match first
    url_col  = smart_col(df, ["url","link","page_url","article_url","slug"])
    desc_col = smart_col(df, ["description","content","article","body","html",
                               "article_html","text","article_body"])
    # 2. Content-based fallback
    if not url_col:
        url_col = _detect_url_col(df)
    if not desc_col:
        desc_col = _detect_desc_col(df, url_col)
    rename = {}
    if url_col:  rename[url_col]  = "url"
    if desc_col: rename[desc_col] = "description"
    df = df.rename(columns=rename)
    if "url"         not in df.columns: df["url"]         = ""
    if "description" not in df.columns: df["description"] = ""
    return df[["url","description"] +
               [c for c in df.columns if c not in ("url","description")]].copy(), bool(url_col)

def verdict_for(copy_score, topic_score, threshold, similar_topic_threshold=None):
    """
    copy_score              = 5-word shingle Jaccard % (primary)
    topic_score             = TF-IDF cosine %          (secondary)
    threshold               = user-set copy threshold
    similar_topic_threshold = if set, fire "Similar topic" when topic_score >= this value;
                              None means the "Similar topic" verdict is disabled (default)
    """
    if copy_score >= threshold:
        return "danger", "Duplicate"
    if copy_score >= max(threshold * 0.4, 12):
        return "warn", "High overlap"
    if similar_topic_threshold is not None and topic_score >= similar_topic_threshold:
        return "info", "Similar topic"
    return "ok", "Unique"


# ─────────────────────────────────────────────────────────────────────────────
# Corpus helpers (UNCHANGED)
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
    texts_path = os.path.join(CACHE_DIR, f"{key}.texts.pkl")
    texts = joblib.load(texts_path) if os.path.exists(texts_path) else None
    return vec, mat, urls, texts

def _save_to_disk(key, vec, mat, urls, texts=None):
    joblib.dump(vec,  os.path.join(CACHE_DIR, f"{key}.vec.pkl"))
    save_npz(         os.path.join(CACHE_DIR, f"{key}.mat.npz"), mat)
    joblib.dump(urls, os.path.join(CACHE_DIR, f"{key}.urls.pkl"))
    if texts is not None:
        joblib.dump(texts, os.path.join(CACHE_DIR, f"{key}.texts.pkl"))


def _ensure_corpus_index():
    if "corpus_urls" in st.session_state:
        return

    key = _cache_key()

    if _disk_cache_exists(key):
        with st.spinner("Loading index from disk…"):
            vec, mat, urls, texts = _load_from_disk(key)
        st.session_state["corpus_urls"]   = urls
        st.session_state["corpus_vec"]    = vec
        st.session_state["corpus_mat"]    = mat
        st.session_state["corpus_texts"]  = texts
        # one-time migration: build texts.pkl if it didn't exist yet
        if texts is None:
            texts_path = os.path.join(CACHE_DIR, f"{key}.texts.pkl")
            with st.spinner("Building phrase index (one-time)…"):
                _df = load_corpus(CORPUS_PATH)
                short = _vectorized_strip(_df["description"]).tolist()
                joblib.dump(short, texts_path)
            st.session_state["corpus_texts"] = short
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
        _run_with_progress(lambda: _save_to_disk(key, vec, mat, urls, texts=texts),
                           "Writing files", 15)

    st.session_state["corpus_urls"]  = urls
    st.session_state["corpus_vec"]   = vec
    st.session_state["corpus_mat"]   = mat
    st.session_state["corpus_texts"] = texts
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Application header
# ─────────────────────────────────────────────────────────────────────────────
_n_pre = len(st.session_state.get("corpus_urls", []))
_corpus_badge = (
    f'<div class="corpus-chip">'
    f'<span class="corpus-dot"></span>'
    f'<span>Corpus &nbsp;<span class="corpus-count">{_n_pre:,}</span>&nbsp; articles</span>'
    f'</div>'
    if _n_pre > 0 else
    '<div class="corpus-chip err"><span class="corpus-dot"></span>Loading corpus…</div>'
)
_hlogo = (f'<img src="{LOGO_URI}" class="app-logo" alt="KollegeApply">'
          if LOGO_URI else '<div class="app-logo-fb">🎓</div>')
_has_results = bool(st.session_state.get("results"))
_web_active  = st.session_state.get("run_web_chk", False)
_tab_web_cls    = "active-green"  if _web_active  else ""
_tab_res_cls    = "active-purple" if _has_results else ""
_dot_res     = f'<span class="nav-tab-dot" style="background:var(--purple)"></span>' if _has_results else ""
st.markdown(
    f'<div class="header-accent"></div>'
    f'<div class="app-header">'
    f'  <div class="app-header-left">'
    f'    {_hlogo}'
    f'    <div class="app-brand-block">'
    f'      <span class="app-brand">KollegeApply</span>'
    f'      <span class="app-page">Plag Checker</span>'
    f'    </div>'
    f'    <span class="app-sep"></span>'
    f'    <span class="app-badge">Internal</span>'
    f'  </div>'
    f'  <nav class="app-nav">'
    f'    <span class="nav-tab active">'
    f'      <span class="nav-tab-dot" style="background:var(--blue)"></span>Detection</span>'
    f'    <span class="nav-tab {_tab_web_cls}">Web Check</span>'
    f'    <span class="nav-tab {_tab_res_cls}">{_dot_res}Results</span>'
    f'  </nav>'
    f'  <div class="app-header-right">{_corpus_badge}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus check + load
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CORPUS_PATH):
    alert_strip(
        f'Corpus CSV not found at <code>{CORPUS_PATH}</code>. '
        'Check the path and restart the app.',
        kind="danger", icon="❌",
    )
    st.stop()

_ensure_corpus_index()

corpus_urls  = st.session_state["corpus_urls"]
corpus_vec   = st.session_state["corpus_vec"]
corpus_mat   = st.session_state["corpus_mat"]
corpus_texts = st.session_state.get("corpus_texts")   # may be None for old caches
n_corp       = len(corpus_urls)



# ─────────────────────────────────────────────────────────────────────────────
# Top filter bar — collapsed accordion by default
# threshold is fixed; no user-adjustable slider
# ─────────────────────────────────────────────────────────────────────────────
dup_threshold = 35  # fixed Copyscape-style threshold — not user-configurable

# Defaults (used when accordion is collapsed and widgets haven't rendered)
flag_similar = st.session_state.get("flag_similar_chk", False)
run_web      = st.session_state.get("run_web_chk", False)
top_n        = 200
n_passages, web_thresh, own_domain, excl_domains = 8, 85, "kollegeapply.com", "wikipedia.org\nyoutube.com"

_engine_chip = (
    '<span style="display:inline-flex;align-items:center;gap:5px;padding:2px 9px;'
    'background:var(--green-bg);border:1px solid var(--green-border);'
    'border-radius:99px;font-size:10.5px;font-weight:600;color:var(--green-text);">'
    '<span style="width:5px;height:5px;border-radius:50%;background:var(--green);'
    'animation:pulse 1.8s ease infinite;"></span>Semantic on</span>'
    if _EMBED_AVAILABLE else
    '<span style="display:inline-flex;align-items:center;gap:5px;padding:2px 9px;'
    'background:var(--amber-bg);border:1px solid var(--amber-border);'
    'border-radius:99px;font-size:10.5px;font-weight:600;color:var(--amber-text);">'
    '<span style="width:5px;height:5px;border-radius:50%;background:var(--amber);"></span>Phrase only</span>'
)

with st.expander(
    "🔽  Detection filters — Similar topic · Web verification · Max results",
    expanded=False,
):
    fc1, fc2, fc3, fc4 = st.columns([4, 4, 3, 3], gap="large")

    with fc1:
        st.markdown(
            '<div class="filter-label">'
            '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
            '&nbsp;Similar topic</div>',
            unsafe_allow_html=True,
        )
        flag_similar = st.checkbox("Flag similar topics", value=False, key="flag_similar_chk")
        st.markdown(
            '<div class="filter-sub">Off by default — education articles naturally share vocabulary.</div>',
            unsafe_allow_html=True,
        )

    with fc2:
        st.markdown(
            '<div class="filter-label">'
            '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
            '&nbsp;Web verification</div>',
            unsafe_allow_html=True,
        )
        run_web = st.checkbox("Check against open web", value=False, key="run_web_chk")
        st.markdown(
            '<div class="filter-sub">Compare against publicly accessible web pages.</div>',
            unsafe_allow_html=True,
        )
        if run_web:
            st.markdown('<div class="filter-sub" style="margin-top:10px;font-weight:600;color:var(--navy);">Options</div>', unsafe_allow_html=True)
            n_passages   = st.slider("Passages per article", 4, 20, 8)
            web_thresh   = st.slider("Match threshold", 70, 100, 85, format="%d%%")
            own_domain   = st.text_input("Your domain (excluded)", "kollegeapply.com")
            excl_domains = st.text_area("Other excluded domains", "wikipedia.org\nyoutube.com", height=60)

    with fc3:
        st.markdown(
            '<div class="filter-label">'
            '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>'
            '&nbsp;Max results</div>',
            unsafe_allow_html=True,
        )
        top_n = st.number_input("Max results shown", 10, 5000, 200, label_visibility="collapsed")
        st.markdown(
            '<div class="filter-sub">Matches displayed per run.</div>',
            unsafe_allow_html=True,
        )

    with fc4:
        st.markdown('<div class="filter-label">Engine status</div>', unsafe_allow_html=True)
        st.markdown(_engine_chip, unsafe_allow_html=True)

similar_topic_threshold = 80 if flag_similar else None

# ── Upload + run ──────────────────────────────────────────────────────────────
if True:

    # ── Section 1: Upload ─────────────────────────────────────────────────────
    with st.container(border=True):
        _panel_hd("1", "Add content to check", color="blue")

        new_df = None
        tab_csv, tab_url = st.tabs(["📂  Upload CSV", "🔗  Paste URLs"])

        with tab_csv:
            new_file = st.file_uploader(
                "upload", type=["csv"], key="new_upload",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="schema-row">'
                '<div class="schema-group"><span class="schema-tag">Required</span>'
                '<span class="schema-key req">url</span></div>'
                '<div class="schema-group"><span class="schema-tag">Optional</span>'
                '<span class="schema-key">description</span></div>'
                '</div>'
                '<p class="schema-hint">Providing a <code>description</code> column (HTML content) avoids an additional page fetch per article.</p>',
                unsafe_allow_html=True,
            )
            if new_file:
                try:
                    raw_new = pd.read_csv(new_file)
                except Exception as e:
                    st.error(f"Couldn't read that CSV: {e}")
                    raw_new = None
                if raw_new is not None:
                    try:
                        df2, url_detected = normalise_new(raw_new)
                    except Exception as e:
                        st.error(f"Column mapping failed: {e}")
                        df2, url_detected = None, False
                    new_df = df2
                    if new_df is not None:
                        new_df["url"]         = new_df["url"].astype(str).str.strip()
                        new_df["description"] = new_df["description"].astype(str).fillna("")
                        # Drop rows with null/empty/nan URLs only when a URL column was found
                        if url_detected:
                            before = len(new_df)
                            new_df = new_df[
                                new_df["url"].notna() &
                                (new_df["url"].str.strip() != "") &
                                (new_df["url"].str.lower() != "nan")
                            ].reset_index(drop=True)
                            dropped = before - len(new_df)
                        else:
                            dropped = 0
                            st.info(
                                "⚠️ URL column not auto-detected. "
                                f"Columns in your CSV: {', '.join(f'`{c}`' for c in raw_new.columns[:8])}. "
                                "Rename the URL column to **url** for best results."
                            )
                        has_desc = int((new_df["description"].str.strip().str.len() > 10).sum())
                        no_desc  = len(new_df) - has_desc
                        pct_ready = int(has_desc / max(len(new_df), 1) * 100)
                        drop_html = (
                            f'<div class="upload-warn">⚠️ {dropped:,} rows skipped — no URL</div>'
                            if dropped else ""
                        )
                        st.markdown(
                            f'<div class="upload-ok">'
                            f'  <div class="upload-ok-row">'
                            f'    <div class="upload-stat">'
                            f'      <span class="upload-big">{len(new_df):,}</span>'
                            f'      <span class="upload-lbl">articles loaded</span>'
                            f'    </div>'
                            f'    <div class="upload-stat">'
                            f'      <span class="upload-big green">{has_desc:,}</span>'
                            f'      <span class="upload-lbl">have HTML (fast)</span>'
                            f'    </div>'
                            f'    <div class="upload-stat">'
                            f'      <span class="upload-big amber">{no_desc:,}</span>'
                            f'      <span class="upload-lbl">need live fetch</span>'
                            f'    </div>'
                            f'  </div>'
                            f'  <div class="upload-cov-bar-wrap">'
                            f'    <div class="upload-cov-bar" style="width:{pct_ready}%"></div>'
                            f'  </div>'
                            f'  <div class="upload-cov-lbl">{pct_ready}% content ready without network fetch</div>'
                            f'  {drop_html}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        with tab_url:
            pasted = st.text_area(
                "One URL per line", height=120,
                placeholder="https://www.kollegeapply.com/article/…\nhttps://…",
            )
            if pasted.strip():
                urls_list = [u.strip() for u in pasted.splitlines()
                             if u.strip().startswith("http")]
                if urls_list:
                    new_df = pd.DataFrame({"url": urls_list, "description": ""})
                    st.success(f"✅ **{len(urls_list)}** URLs ready — content will be fetched live")
                else:
                    st.error("No valid URLs found. Each line should start with http.")

        # ── Run controls — inline inside same card ────────────────────────────
        if new_df is not None and len(new_df) > 0:
            st.divider()
            n_new   = len(new_df)
            n_fetch = int((new_df["description"].str.strip().str.len() <= 10).sum())
            est_sec = n_new * 0.5 + n_fetch * 4
            if run_web:
                est_sec += n_new * n_passages * 5
            est_label = (f"~{max(1,round(est_sec/60))} min" if est_sec > 60
                         else f"~{int(est_sec)}s")

            run_left, run_right = st.columns([3, 1], vertical_alignment="bottom")
            with run_left:
                st.markdown(
                    f'<div class="info-row">'
                    f'<div class="info-tile blue"><div class="info-val">{n_new}</div><div class="info-lbl">Articles</div></div>'
                    f'<div class="info-tile green"><div class="info-val">{n_corp:,}</div><div class="info-lbl">Corpus</div></div>'
                    f'<div class="info-tile amber"><div class="info-val">{n_fetch}</div><div class="info-lbl">To fetch</div></div>'
                    f'<div class="info-tile purple"><div class="info-val">{est_label}</div><div class="info-lbl">Est. time</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with run_right:
                run_btn = st.button("→  Run check", type="primary",
                                    use_container_width=True, key="run_btn")

            if n_fetch > 0:
                callout(
                    f"<b>{n_fetch} article{'s' if n_fetch>1 else ''}</b> will be fetched live — add a <code>description</code> column to speed this up.",
                    kind="warn", icon="⚡",
                )

            if run_btn:
                # Clear cached page text for these URLs so re-runs pick up live changes
                clear_page_cache(new_df["url"].tolist())

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

                # ── Multi-signal similarity (Copyscape / Turnitin style) ─────────────
                _sem_label = " + semantic" if _EMBED_AVAILABLE else ""
                with st.status(f"⚡ Computing similarity{_sem_label}…", expanded=False) as s3:
                    valid_new   = [(i, t) for i, t in enumerate(new_texts) if len(t.split()) >= 30]
                    # tuple: (tfidf_sc, corpus_idx, oov_ratio, copy_score, matched_sents)
                    sim_results = [(0.0, 0, 0.0, 0.0, 0)] * n_new

                    corpus_vocab = set(corpus_vec.vocabulary_.keys())
                    _stop = corpus_vec.get_stop_words() or set()

                    def _norm_url(u):
                        return str(u).strip().lower().rstrip("/")

                    def _oov_ratio(text):
                        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
                        tokens = [t for t in tokens if t not in _stop]
                        if not tokens: return 0.0
                        oov = sum(1 for t in tokens if t not in corpus_vocab)
                        return round(oov / len(tokens) * 100, 1)

                    def _words(text):
                        return re.findall(r'\b[a-z]{2,}\b', text.lower())

                    def _jaccard_words(wa, wb, k):
                        if len(wa) < k or len(wb) < k:
                            return 0.0
                        sa = set(zip(*[wa[i:] for i in range(k)]))
                        sb = set(zip(*[wb[i:] for i in range(k)]))
                        inter = len(sa & sb)
                        union = len(sa | sb)
                        return inter / union if union else 0.0

                    def _shingle_jaccard(text_a, text_b):
                        """Dual k=4 + k=5 shingle Jaccard. Returns % of best signal."""
                        wa, wb = _words(text_a), _words(text_b)
                        j5 = _jaccard_words(wa, wb, 5)
                        j4 = _jaccard_words(wa, wb, 4)
                        return round(max(j4 * 0.85, j5) * 100, 1)

                    def _paragraph_jaccard(new_text, corp_text):
                        """Split both texts into paragraphs; return the highest paragraph-pair
                        Jaccard. Catches partial copies where only one section is lifted."""
                        def _paras(t):
                            chunks = re.split(r'\n{2,}|(?<=[.!?])\s{2,}', t)
                            return [c.strip() for c in chunks if len(c.split()) >= 25]
                        new_paras  = _paras(new_text)
                        corp_paras = _paras(corp_text)
                        if not new_paras or not corp_paras:
                            return 0.0
                        best = 0.0
                        for np_ in new_paras:
                            wn = _words(np_)
                            for cp in corp_paras:
                                wc_ = _words(cp)
                                j = _jaccard_words(wn, wc_, 5)
                                if j > best:
                                    best = j
                        return round(best * 100, 1)

                    def _verbatim_sentences(new_text, corp_text, min_words=8):
                        corp_norm = re.sub(r'\s+', ' ', corp_text.lower())
                        count = 0
                        for sent in re.split(r'(?<=[.!?])\s+|\n', new_text):
                            s = re.sub(r'\s+', ' ', sent.lower().strip())
                            if len(s.split()) >= min_words and s in corp_norm:
                                count += 1
                        return count

                    import numpy as np_mod
                    TOP_K = 15  # check against top-15 TF-IDF candidates

                    if valid_new and corpus_mat.shape[0] > 0:
                        vt      = [t for _, t in valid_new]
                        new_mat = corpus_vec.transform(vt)
                        sims    = cosine_similarity(new_mat, corpus_mat)
                        corpus_url_idx = {_norm_url(u): i for i, u in enumerate(corpus_urls)}

                        for ni, (orig_i, t) in enumerate(valid_new):
                            row_url = _norm_url(new_df.iloc[orig_i]["url"])
                            scores  = sims[ni].copy()
                            if row_url in corpus_url_idx:
                                scores[corpus_url_idx[row_url]] = -1.0

                            best_j  = int(scores.argmax())
                            best_sc = round(float(scores[best_j]) * 100, 1)

                            copy_sc = 0.0
                            matched_sents = 0
                            if corpus_texts is not None:
                                k_size   = min(TOP_K, len(scores))
                                top_idxs = np_mod.argpartition(scores, -k_size)[-k_size:]
                                for cj in top_idxs:
                                    ct = corpus_texts[int(cj)]
                                    if not ct:
                                        continue
                                    # Full-text dual-gram Jaccard
                                    js = _shingle_jaccard(t, ct)
                                    # Paragraph-level max (catches partial section copies)
                                    pj = _paragraph_jaccard(t, ct)
                                    best_pair = max(js, pj)
                                    if best_pair > copy_sc:
                                        copy_sc = best_pair
                                        matched_sents = _verbatim_sentences(t, ct)

                            # Semantic similarity boost (sentence-transformers, optional)
                            # Runs only when embeddings are available and phrase signals are
                            # ambiguous (copy_sc in 8–30%: could be paraphrase or false positive).
                            if _EMBED_AVAILABLE and _EMBEDDER is not None and corpus_texts is not None:
                                if 8.0 <= copy_sc <= 30.0:
                                    try:
                                        import numpy as _np2
                                        best_j_txt = corpus_texts[best_j] or ""
                                        if best_j_txt:
                                            t_trunc  = t[:3000]
                                            ct_trunc = best_j_txt[:3000]
                                            embs = _EMBEDDER.encode(
                                                [t_trunc, ct_trunc],
                                                normalize_embeddings=True,
                                                show_progress_bar=False,
                                            )
                                            sem_sc = float(_np2.dot(embs[0], embs[1])) * 100
                                            # Semantic score blends in at 30% weight when
                                            # phrase signal is in the ambiguous zone.
                                            copy_sc = round(copy_sc * 0.70 + sem_sc * 0.30, 1)
                                    except Exception:
                                        pass

                            sim_results[orig_i] = (best_sc, best_j, _oov_ratio(t),
                                                   copy_sc, matched_sents)
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
                for row, txt, wc, (tfidf_sc, ci, oov, copy_sc, msents) in zip(
                    new_df.itertuples(), new_texts, new_wc, sim_results
                ):
                    corp_url = corpus_urls[ci] if ci < len(corpus_urls) else ""
                    wr       = web_res.get(row.url, {})
                    cls, vlabel = verdict_for(copy_sc, tfidf_sc, dup_threshold, similar_topic_threshold)[:2]
                    final_rows.append({
                        "url":                    row.url,
                        "word_count":             wc,
                        "copy_score_%":           copy_sc,
                        "topic_overlap_%":        tfidf_sc,
                        "matched_sentences":      msents,
                        "new_vocab_%":            oov,
                        "verdict":                vlabel,
                        "matched_existing_url":   corp_url,
                        "web_plag_score":         wr.get("plagiarism_score"),
                        "web_verdict":            wr.get("verdict", ""),
                        "top_web_sources":        wr.get("top_sources", ""),
                        "_matches":               wr.get("matches", []),
                    })

                st.session_state["results"] = final_rows
                st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Results (full width)
# ─────────────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    rdf = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])

    n_dup = int((rdf["verdict"] == "Duplicate").sum())
    n_rev = int(rdf["verdict"].isin(["High overlap", "Similar topic"]).sum())
    n_ok  = int((rdf["verdict"] == "Unique").sum())
    avg_s = f"{rdf['copy_score_%'].mean():.1f}%"
    hi_s  = f"{rdf['copy_score_%'].max():.1f}%"

    # Results header
    hdr_l, hdr_r = st.columns([6, 1], vertical_alignment="bottom")
    with hdr_l:
        st.markdown(
            '<div class="panel-hd" style="margin-top:8px;border:none;padding-bottom:6px;">'
            '<span class="panel-num green">3</span>'
            '<span class="panel-title" style="font-size:16px;">Results</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.pop("results", None)
            st.rerun()

    # Summary stat tiles
    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-tile"><div class="stat-val">{len(rdf)}</div><div class="stat-lbl">Checked</div></div>'
        f'<div class="stat-tile green"><div class="stat-val">{n_ok}</div><div class="stat-lbl">Original</div></div>'
        f'<div class="stat-tile amber"><div class="stat-val">{n_rev}</div><div class="stat-lbl">High overlap</div></div>'
        f'<div class="stat-tile red"><div class="stat-val">{n_dup}</div><div class="stat-lbl">Duplicate</div></div>'
        f'<div class="stat-tile purple"><div class="stat-val">{avg_s}</div><div class="stat-lbl">Avg copy score</div></div>'
        f'<div class="stat-tile"><div class="stat-val">{hi_s}</div><div class="stat-lbl">Highest copy</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_dup:
        alert_strip(
            f'<b>{n_dup} article{"s" if n_dup>1 else ""}</b> '
            f'{"have" if n_dup>1 else "has"} copy score ≥{dup_threshold}% — '
            'text is substantially copied from existing content.',
            kind="danger",
        )
    else:
        alert_strip(
            'All clear — every article is below the duplicate threshold.',
            kind="success", icon="🎉",
        )

    # ── Verdict guide ─────────────────────────────────────────────────────────
    _sim_tile = (
        '<div class="vg-item info">'
        '  <div class="vg-head"><div class="vg-dot info"></div><span class="vg-title">Similar topic</span></div>'
        '  <div class="vg-action">📝 Review differentiation</div>'
        '  <div class="vg-desc">Same subject as a published article, but independently written. OK to publish — consider adding a unique angle.</div>'
        '</div>'
    ) if flag_similar else ""
    st.markdown(
        '<div class="vguide">'
        '<div class="vg-item ok">'
        '  <div class="vg-head"><div class="vg-dot ok"></div><span class="vg-title">Unique</span></div>'
        '  <div class="vg-action">✅ Publish</div>'
        '  <div class="vg-desc">No phrase overlap with existing content. Original writing — safe to go live.</div>'
        '</div>'
        + _sim_tile +
        '<div class="vg-item warn">'
        '  <div class="vg-head"><div class="vg-dot warn"></div><span class="vg-title">High overlap</span></div>'
        '  <div class="vg-action">✏️ Rewrite before publishing</div>'
        '  <div class="vg-desc">Significant shared phrases found. Rewrite the overlapping passages, then re-check.</div>'
        '</div>'
        '<div class="vg-item danger">'
        '  <div class="vg-head"><div class="vg-dot danger"></div><span class="vg-title">Duplicate</span></div>'
        '  <div class="vg-action">🚫 Do not publish</div>'
        '  <div class="vg-desc">Text is substantially copied from an existing article. Must be fully rewritten.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── View selector ──────────────────────────────────────────────────────────
    view = st.segmented_control(
        "View", ["📋  All results", "🚨  Issues only", "🌐  Web plagiarism"],
        default="📋  All results", label_visibility="collapsed",
    ) or "📋  All results"

    # ── Article cards ──────────────────────────────────────────────────────────
    if view in ("📋  All results", "🚨  Issues only"):
        show = results if view == "📋  All results" else [
            r for r in results if r["verdict"] in ("Duplicate", "High overlap", "Similar topic")
        ]
        show = sorted(show, key=lambda r: r["copy_score_%"], reverse=True)
        show = show[:int(top_n)]

        if not show:
            callout(
                "No issues found at this threshold — you're all good! 🎉",
                kind="ok", icon="✅",
            )
        else:
            # ── Per-article cards
            _ACTION = {
                "danger": ("🚫 Do not publish — text is substantially copied from an existing article.", "danger"),
                "warn":   ("✏️ Rewrite before publishing — significant phrase overlap found.", "warn"),
                "info":   ("📝 OK to publish — independently written. Consider adding a unique angle.", "info"),
                "ok":     ("✅ Safe to publish — no significant phrase overlap detected.", "ok"),
            }
            _BADGE  = {
                "danger": "🚨 Duplicate",
                "warn":   "⚠️ High overlap",
                "info":   "📝 Similar topic",
                "ok":     "✅ Unique",
            }

            def _slug(url):
                """Return last 2 path segments of a URL as a readable label."""
                parts = url.rstrip("/").split("/")
                return "/".join(parts[-2:]) if len(parts) >= 2 else url

            for r in show:
                copy_sc  = r.get("copy_score_%", 0.0)
                topic_sc = r.get("topic_overlap_%", 0.0)
                msents   = r.get("matched_sentences", 0)
                oov      = r.get("new_vocab_%", 0.0)
                cls, vlabel = verdict_for(copy_sc, topic_sc, dup_threshold, similar_topic_threshold)[:2]
                action_txt, action_cls = _ACTION[cls]
                badge_lbl  = _BADGE[cls]
                url_e   = html_module.escape(r["url"])
                corp_u  = html_module.escape(r["matched_existing_url"], quote=True)
                corp_ue = html_module.escape(r["matched_existing_url"])
                corp_slug = html_module.escape(_slug(r["matched_existing_url"]))

                # Copy score bar: scale 0→dup_threshold to 0→100%
                copy_bar = min(int(copy_sc / max(dup_threshold, 1) * 100), 100)
                # Topic bar: 0→100%
                topic_bar = min(int(topic_sc), 100)

                copy_color  = {"danger":"#DC2626","warn":"#F59E0B","info":"#408EE0","ok":"#10B981"}[cls]
                topic_color = "#7C3AED"

                sents_badge = (
                    f'<span style="background:#FEF2F2;color:#DC2626;border:1px solid #FECACA;'
                    f'border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;margin-left:6px;">'
                    f'⚠ {msents} verbatim line{"s" if msents!=1 else ""}</span>'
                ) if msents > 0 else ""

                st.markdown(
                    f'<div class="art-card {cls}">'
                    f'<div class="art-row">'
                    # URL + word count
                    f'<div class="art-url-block">'
                    f'  <a class="art-url" href="{url_e}" target="_blank">{url_e}</a>'
                    f'  <div class="art-wc">{r["word_count"]:,} words</div>'
                    f'</div>'
                    # Scores block
                    f'<div class="art-scores">'
                    f'  <div class="art-metric">'
                    f'    <div class="art-metric-val" style="color:{copy_color};">{copy_sc:.1f}%</div>'
                    f'    <div class="art-meter-bg"><div class="art-meter-fill" style="width:{copy_bar}%;background:{copy_color};"></div></div>'
                    f'    <div class="art-metric-lbl">Copy score</div>'
                    f'  </div>'
                    f'  <div class="art-metric">'
                    f'    <div class="art-metric-val" style="color:{topic_color};">{topic_sc:.0f}%</div>'
                    f'    <div class="art-meter-bg"><div class="art-meter-fill" style="width:{topic_bar}%;background:{topic_color};"></div></div>'
                    f'    <div class="art-metric-lbl">Topic match</div>'
                    f'  </div>'
                    f'  <div class="art-metric">'
                    f'    <div class="art-metric-val" style="color:#64748B;">{oov:.0f}%</div>'
                    f'    <div class="art-meter-bg"><div class="art-meter-fill" style="width:{min(int(oov),100)}%;background:#64748B;"></div></div>'
                    f'    <div class="art-metric-lbl">New vocab</div>'
                    f'  </div>'
                    f'</div>'
                    # Verdict badge
                    f'<div class="art-verdict-col">'
                    f'  <span class="art-vbadge {cls}">{badge_lbl}</span>{sents_badge}'
                    f'</div>'
                    # Action + matched
                    f'<div class="art-action-col">'
                    f'  <div class="art-action {action_cls}">{action_txt}</div>'
                    f'  <div class="art-matched">Closest: <a href="{corp_u}" target="_blank">{corp_slug}</a></div>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Expandable detail for flagged articles
                if cls in ("danger", "warn", "info"):
                    with st.expander(f"🔍 Detail — {_slug(r['url'])}"):
                        st.markdown(
                            f'<div class="art-detail-body">'
                            f'<p style="margin:0 0 10px;font-weight:600;">Why is this flagged?</p>'
                            f'<div class="art-detail-grid">'
                            f'  <div class="art-detail-item">'
                            f'    <div class="adi-lbl">Copy score (phrase Jaccard)</div>'
                            f'    <div class="adi-val" style="color:{copy_color};">{copy_sc:.1f}%</div>'
                            f'    <div class="adi-sub">% of 5-word phrases shared with closest article.<br>'
                            f'    Threshold: {dup_threshold}% = Duplicate, {max(int(dup_threshold*0.4),12)}% = High overlap.</div>'
                            f'  </div>'
                            f'  <div class="art-detail-item">'
                            f'    <div class="adi-lbl">Topic overlap (TF-IDF)</div>'
                            f'    <div class="adi-val" style="color:#7C3AED;">{topic_sc:.0f}%</div>'
                            f'    <div class="adi-sub">Vocabulary similarity — high for same-subject articles<br>even when written differently. Not a copy signal alone.</div>'
                            f'  </div>'
                            f'  <div class="art-detail-item">'
                            f'    <div class="adi-lbl">Verbatim sentences matched</div>'
                            f'    <div class="adi-val" style="color:{"#DC2626" if msents else "#10B981"};">{msents}</div>'
                            f'    <div class="adi-sub">Sentences ≥8 words found word-for-word in the corpus article.<br>Any match is strong evidence of direct copying.</div>'
                            f'  </div>'
                            f'  <div class="art-detail-item">'
                            f'    <div class="adi-lbl">New vocabulary</div>'
                            f'    <div class="adi-val">{oov:.1f}%</div>'
                            f'    <div class="adi-sub">Meaningful words in this article not in any corpus article.<br>High = genuinely new content; low = same vocabulary pool.</div>'
                            f'  </div>'
                            f'</div>'
                            f'<p style="margin:14px 0 4px;font-weight:600;font-size:12px;">Closest existing article</p>'
                            f'<a href="{corp_u}" target="_blank" style="color:#408EE0;font-size:12px;">{corp_ue}</a>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── Web plagiarism view ────────────────────────────────────────────────────
    else:
        if not run_web:
            callout(
                "Enable <b>Web verification</b> in the settings panel, then re-run to see web results.",
                kind="info", icon="🌐",
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

    # ── Export ─────────────────────────────────────────────────────────────────
    st.divider()
    export = rdf.drop(columns=["_matches", "_text"], errors="ignore").copy()
    export["action"] = export["verdict"].map({
        "Duplicate":    "Do not publish — rewrite fully",
        "High overlap": "Rewrite before publishing",
        "Similar topic":"OK to publish — review differentiation",
        "Unique":       "Safe to publish",
    })
    fname = f"plag_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    dl_col, cap_col = st.columns([2, 5], vertical_alignment="center")
    dl_col.download_button(
        "📥  Download CSV report",
        export.to_csv(index=False).encode("utf-8"),
        fname, "text/csv", use_container_width=True,
    )
    cap_col.markdown(
        f'<p style="font-size:.78rem;color:#94A3B8;margin:0;">'
        f'url · words · copy_score · topic_overlap · matched_sentences · new_vocab · verdict · action · matched_url'
        f'{"  ·  web_score · web_verdict · top_sources" if run_web else ""}</p>',
        unsafe_allow_html=True,
    )
