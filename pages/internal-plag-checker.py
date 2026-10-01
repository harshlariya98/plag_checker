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
from plag_utils import fetch_text, check_article

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
.block-container { padding: 3.75rem 2rem 4rem !important; max-width: 1440px !important; }

/* Streamlit top bar — make it transparent so the custom app-header is the real header */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
div[data-testid="stToolbar"] > div:last-child { display: none !important; }

/* ── Keyframes ─────────────────────────────────────────────────────────────── */
@keyframes fadeUp  { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
@keyframes fadeIn  { from{opacity:0} to{opacity:1} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Sidebar nav rail ──────────────────────────────────────────────────────── */
/* Hide Streamlit's auto-generated page navigation list */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] { display: none !important; }
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
.nav-brand {
    padding: 16px 16px 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 6px;
    display: flex; align-items: center; gap: 9px;
}
.nav-logo { width: 28px; height: 28px; object-fit: contain; flex-shrink: 0; display: block; }
.nav-logo-fb {
    width: 28px; height: 28px; border-radius: 7px; background: var(--navy);
    display: flex; align-items: center; justify-content: center; font-size: .85rem; flex-shrink: 0;
}
.nav-title { font-size: 13px; font-weight: 700; color: var(--navy); line-height: 1.25; }
.nav-sub   { font-size: 10px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-top: 1px; }
.nav-section {
    padding: 14px 16px 4px;
    font-size: 10px; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: var(--muted);
}
.nav-link {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 16px; margin: 1px 8px;
    border-radius: var(--radius-xs);
    font-size: 13px; font-weight: 500; color: var(--text);
    transition: background .15s;
}
.nav-link:hover { background: var(--bg); }
.nav-link.active { background: var(--blue-bg); color: var(--blue); font-weight: 600; }
.nav-icon { font-size: 14px; width: 18px; text-align: center; flex-shrink: 0; }

/* ── Application header strip ──────────────────────────────────────────────── */
.app-header {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
    animation: fadeIn .3s ease;
}
.app-logo    { width: 24px; height: 24px; object-fit: contain; flex-shrink: 0; display: block; }
.app-logo-fb { width: 24px; height: 24px; border-radius: 6px; background: var(--navy);
               display: flex; align-items: center; justify-content: center; font-size: .8rem; }
.app-brand   { font-size: 13.5px; font-weight: 600; color: var(--navy); }
.app-sep     { width: 1px; height: 14px; background: var(--border); flex-shrink: 0; }
.app-page    { font-size: 14px; font-weight: 700; color: var(--navy); }
.app-badge   {
    font-size: 10px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
    color: var(--blue); background: var(--blue-bg);
    border: 1px solid var(--blue-border);
    padding: 2px 8px; border-radius: 4px;
}
.app-right { margin-left: auto; display: flex; align-items: center; gap: 14px; }
.corpus-chip {
    display: flex; align-items: center; gap: 6px;
    font-size: 12.5px; color: var(--muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 99px; padding: 4px 12px 4px 8px;
    box-shadow: var(--shadow-xs);
}
.corpus-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; animation: pulse 1.8s ease infinite; }
.corpus-chip strong { color: var(--navy); font-weight: 700; }
.corpus-chip.err .corpus-dot { background: var(--red); animation: none; }
.corpus-chip.err { color: var(--red-text); }

/* ── Page intro ────────────────────────────────────────────────────────────── */
.page-intro { margin-bottom: 20px; animation: fadeUp .3s ease both; }
.page-intro h2 {
    font-size: 22px; font-weight: 800; color: var(--navy);
    margin: 0 0 5px; letter-spacing: -.4px;
}
.page-intro p { font-size: 13.5px; color: var(--muted); margin: 0; line-height: 1.6; }

/* ── Panel card (Streamlit bordered container) ─────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 12px !important;
    animation: fadeUp .35s ease both;
}

.panel-hd {
    display: flex; align-items: center; gap: 9px;
    padding-bottom: 14px; margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
}
.panel-num {
    width: 22px; height: 22px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800; color: #fff;
}
.panel-num.blue   { background: var(--blue); }
.panel-num.coral  { background: var(--coral); }
.panel-num.green  { background: var(--green); }
.panel-num.purple { background: var(--purple); }
.panel-title { font-size: 14px; font-weight: 700; color: var(--navy); }
.panel-sub   { font-size: 12px; color: var(--muted); margin-left: auto; }

/* ── Settings panel ────────────────────────────────────────────────────────── */
.settings-hd {
    display: flex; align-items: center; gap: 7px;
    font-size: 13px; font-weight: 700; color: var(--navy);
    margin-bottom: 8px;
}
.settings-block {
    padding-bottom: 14px; margin-bottom: 14px;
    border-bottom: 1px solid var(--border);
}
.settings-block:last-child { border-bottom: none; padding-bottom: 0; margin-bottom: 0; }
.settings-label {
    font-size: 11px; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px; display: block;
}
.settings-sub { font-size: 11.5px; color: var(--muted); margin-top: 5px; line-height: 1.5; }

/* Threshold colour bar */
.thr-bar { display: flex; height: 5px; border-radius: 99px; overflow: hidden; margin: 10px 0 6px; }
.thr-ok   { background: var(--green); }
.thr-warn { background: var(--amber); }
.thr-dup  { background: var(--red);   }
.thr-labels { display: flex; font-size: 10.5px; color: var(--muted); font-weight: 500; }
.thr-labels span { display: flex; align-items: center; gap: 3px; }
.thr-labels span:last-child { margin-left: auto; }
.tld { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

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
.info-row { display: flex; gap: 8px; margin: 14px 0; flex-wrap: wrap; }
.info-tile {
    flex: 1; min-width: 80px; background: var(--light);
    border-radius: var(--radius-sm); padding: 12px 10px;
    border: 1px solid var(--border); text-align: center;
}
.info-val { font-size: 20px; font-weight: 800; color: var(--navy); line-height: 1; letter-spacing: -.5px; }
.info-val.amber  { color: #D97706; }
.info-val.purple { color: var(--purple); }
.info-lbl { font-size: 11px; color: var(--muted); font-weight: 500; margin-top: 3px; }

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
.rc.ok     { border-left-color: var(--green); }
.rc-score { display: flex; flex-direction: column; align-items: center; gap: 4px; min-width: 50px; }
.rc-pct { font-size: 19px; font-weight: 800; line-height: 1; }
.rc.danger .rc-pct { color: var(--red);   }
.rc.warn   .rc-pct { color: #D97706;      }
.rc.ok     .rc-pct { color: var(--green); }
.rc-bar-bg { width: 40px; height: 3px; background: var(--border); border-radius: 99px; overflow: hidden; }
.rc-bar    { height: 3px; border-radius: 99px; }
.rc.danger .rc-bar { background: var(--red);   }
.rc.warn   .rc-bar { background: var(--amber); }
.rc.ok     .rc-bar { background: var(--green); }
.rc-body { flex: 1; font-size: 13px; color: #374151; line-height: 1.65; min-width: 0; }
.rc-body a { color: var(--blue); text-decoration: none; overflow-wrap: break-word; word-break: break-all; font-size: 12.5px; }
.rc-body a:hover { text-decoration: underline; }
.rc-label { font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #94A3B8; margin-bottom: 2px; }
.rc-badge { display: inline-flex; align-items: center; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-left: 6px; vertical-align: middle; }
.rc-badge.danger { background: var(--red-bg);   color: var(--red-text);   }
.rc-badge.warn   { background: var(--amber-bg); color: var(--amber-text); }
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
# Sidebar — compact navigation rail
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    _sb_logo = (f'<img src="{LOGO_URI}" class="nav-logo" alt="KollegeApply">'
                if LOGO_URI else '<div class="nav-logo-fb">🎓</div>')
    st.markdown(
        f'<div class="nav-brand">{_sb_logo}'
        f'<div><div class="nav-title">KollegeApply</div>'
        f'<div class="nav-sub">Plag Checker · Internal</div></div>'
        f'</div>'
        f'<div class="nav-section">Workspace</div>'
        f'<div class="nav-link active"><span class="nav-icon">🔍</span> Detection</div>'
        f'<div class="nav-link"><span class="nav-icon">🌐</span> Web Check</div>'
        f'<div class="nav-link"><span class="nav-icon">📊</span> Results</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Application header
# ─────────────────────────────────────────────────────────────────────────────
_n_pre = len(st.session_state.get("corpus_urls", []))
_corpus_badge = (
    f'<div class="corpus-chip">'
    f'<span class="corpus-dot"></span>'
    f'Corpus ready &nbsp;·&nbsp; <strong>{_n_pre:,} articles indexed</strong>'
    f'</div>'
    if _n_pre > 0 else
    '<div class="corpus-chip err"><span class="corpus-dot"></span>Loading corpus…</div>'
)
_hlogo = (f'<img src="{LOGO_URI}" class="app-logo" alt="KollegeApply">'
          if LOGO_URI else '<div class="app-logo-fb">🎓</div>')
st.markdown(
    f'<div class="app-header">'
    f'{_hlogo}'
    f'<span class="app-brand">KollegeApply</span>'
    f'<span class="app-sep"></span>'
    f'<span class="app-page">Plag Checker</span>'
    f'<span class="app-badge">INTERNAL TOOL</span>'
    f'<div class="app-right">{_corpus_badge}</div>'
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

corpus_urls = st.session_state["corpus_urls"]
corpus_vec  = st.session_state["corpus_vec"]
corpus_mat  = st.session_state["corpus_mat"]
n_corp      = len(corpus_urls)


# ─────────────────────────────────────────────────────────────────────────────
# Page intro
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="page-intro">'
    f'<h2>Plagiarism Checker</h2>'
    f'<p>Check your content against <strong>{n_corp:,}+</strong> indexed articles before publishing.</p>'
    f'</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Two-column workspace
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([7, 3], gap="large")

# ── Right column: settings panel ─────────────────────────────────────────────
with right_col:
    with st.container(border=True):
        st.markdown('<div class="settings-hd">⚙️ Detection settings</div>', unsafe_allow_html=True)

        # Similarity threshold
        st.markdown('<span class="settings-label">Similarity threshold</span>', unsafe_allow_html=True)
        dup_threshold = st.slider(
            "Duplicate threshold", 20, 100, 65, format="%d%%",
            label_visibility="collapsed",
            help="Articles ≥ this similarity score are flagged as duplicates",
        )
        warn_w = max(1, dup_threshold - 40)
        dup_w  = max(1, 100 - dup_threshold)
        st.markdown(
            f'<div class="thr-bar">'
            f'<div class="thr-ok" style="flex:40"></div>'
            f'<div class="thr-warn" style="flex:{warn_w}"></div>'
            f'<div class="thr-dup" style="flex:{dup_w}"></div>'
            f'</div>'
            f'<div class="thr-labels">'
            f'<span><span class="tld" style="background:var(--green)"></span>&lt;40% Original</span>'
            f'<span><span class="tld" style="background:var(--red)"></span>≥{dup_threshold}% Dup</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Web verification
        st.markdown('<span class="settings-label">🌐 Web verification</span>', unsafe_allow_html=True)
        run_web = st.checkbox("Check against open web", value=False)
        st.markdown(
            '<span class="settings-sub">Compare content against publicly accessible web pages.</span>',
            unsafe_allow_html=True,
        )
        if run_web:
            n_passages   = st.slider("Passages per article", 4, 20, 8)
            web_thresh   = st.slider("Match threshold", 70, 100, 85, format="%d%%")
            own_domain   = st.text_input("Your domain (excluded)", "kollegeapply.com")
            excl_domains = st.text_area("Other excluded domains",
                                        "wikipedia.org\nyoutube.com", height=60)

        st.divider()

        # Results
        st.markdown('<span class="settings-label">📊 Results</span>', unsafe_allow_html=True)
        top_n = st.number_input("Max results shown", 10, 5000, 200)
        st.markdown(
            '<span class="settings-sub">Maximum matches displayed per run.</span>',
            unsafe_allow_html=True,
        )


# ── Left column: upload + run ─────────────────────────────────────────────────
with left_col:

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

        with tab_url:
            pasted = st.text_area(
                "One URL per line", height=140,
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

    # ── Section 2: Run ────────────────────────────────────────────────────────
    if new_df is not None and len(new_df) > 0:
        with st.container(border=True):
            _panel_hd("2", "Ready to run", "review before launching", color="coral")

            n_new   = len(new_df)
            n_fetch = int((new_df["description"].str.strip().str.len() <= 10).sum())
            est_sec = n_new * 0.5 + n_fetch * 4
            if run_web:
                est_sec += n_new * n_passages * 5

            est_label = (f"~{max(1,round(est_sec/60))} min" if est_sec > 60
                         else f"~{int(est_sec)}s")
            st.markdown(
                f'<div class="info-row">'
                f'<div class="info-tile"><div class="info-val">{n_new}</div><div class="info-lbl">Articles</div></div>'
                f'<div class="info-tile"><div class="info-val">{n_corp:,}</div><div class="info-lbl">Corpus size</div></div>'
                f'<div class="info-tile"><div class="info-val amber">{n_fetch}</div><div class="info-lbl">To fetch</div></div>'
                f'<div class="info-tile"><div class="info-val purple">{est_label}</div><div class="info-lbl">Est. time</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if n_fetch > 0:
                callout(
                    f"<b>{n_fetch} article{'s' if n_fetch>1 else ''}</b> missing HTML content — "
                    "will be fetched live (slower). Add a <code>description</code> column to skip this.",
                    kind="warn", icon="⚡",
                )

            run_btn = st.button("→  Run plagiarism check", type="primary",
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
            st.markdown(
                '<div class="empty-state">'
                '<div class="es-icon">📂</div>'
                '<div class="es-title">Ready to check your content</div>'
                '<div class="es-sub">Upload a CSV with a <code>url</code> column, '
                'or paste article URLs above to get started.<br>'
                'Supports up to 200 MB CSV files.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Results (full width)
# ─────────────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    rdf = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])

    n_dup = int((rdf["similarity_to_corpus_%"] >= dup_threshold).sum())
    n_rev = int(((rdf["similarity_to_corpus_%"] >= 40) &
                 (rdf["similarity_to_corpus_%"] < dup_threshold)).sum())
    n_ok  = int((rdf["similarity_to_corpus_%"] < 40).sum())
    avg_s = f"{rdf['similarity_to_corpus_%'].mean():.1f}%"
    hi_s  = f"{rdf['similarity_to_corpus_%'].max():.1f}%"

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
        f'<div class="stat-tile amber"><div class="stat-val">{n_rev}</div><div class="stat-lbl">Needs review</div></div>'
        f'<div class="stat-tile red"><div class="stat-val">{n_dup}</div><div class="stat-lbl">Duplicate</div></div>'
        f'<div class="stat-tile purple"><div class="stat-val">{avg_s}</div><div class="stat-lbl">Avg similarity</div></div>'
        f'<div class="stat-tile"><div class="stat-val">{hi_s}</div><div class="stat-lbl">Highest</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if n_dup:
        alert_strip(
            f'<b>{n_dup} article{"s" if n_dup>1 else ""}</b> '
            f'{"are" if n_dup>1 else "is"} ≥{dup_threshold}% similar to existing content — '
            'review before publishing.',
            kind="danger",
        )
    else:
        alert_strip(
            'All clear — every article is below the duplicate threshold.',
            kind="success", icon="🎉",
        )

    # View tabs
    view = st.segmented_control(
        "View", ["📋  All results", "🚨  Duplicates only", "🌐  Web plagiarism"],
        default="📋  All results", label_visibility="collapsed",
    ) or "📋  All results"

    # ── Table + detail cards ───────────────────────────────────────────────────
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

            # Detail cards for flagged articles
            flagged = [r for r in show if r["similarity_to_corpus_%"] >= 40]
            if flagged:
                st.markdown(
                    '<p class="detail-label">Flagged articles — detail view</p>',
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
                        f'<div class="rc-score">'
                        f'<div class="rc-pct">{sc:.0f}%</div>'
                        f'<div class="rc-bar-bg"><div class="rc-bar" style="width:{bar_w}%"></div></div>'
                        f'</div>'
                        f'<div class="rc-body">'
                        f'<div class="rc-label">New article</div>'
                        f'<a href="{new_e}" target="_blank">{new_e}</a>'
                        f'<span class="rc-badge {cls}">{icon} {vlabel}</span>'
                        f'<div class="rc-label" style="margin-top:.6rem;">Closest match in corpus</div>'
                        f'<a href="{corp_u}" target="_blank">{corp_ue}</a>'
                        f'<div class="rc-meta">{r["word_count"]:,} words &nbsp;·&nbsp; {sc:.1f}% similar</div>'
                        f'</div></div>',
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
        f'<p style="font-size:.78rem;color:#94A3B8;margin:0;">'
        f'url · words · similarity · verdict · matched_url · risk_level'
        f'{"  ·  web_score · web_verdict · top_sources" if run_web else ""}</p>',
        unsafe_allow_html=True,
    )
