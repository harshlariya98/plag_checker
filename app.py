"""
KollegeApply Bulk Plagiarism Checker (100% free, no API keys)

Input : CSV of URLs, paste URLs, or a sitemap URL
Output: per-article plagiarism score, verdict, top matching sources,
        internal-duplicate check against the other articles in the batch,
        and optional cross-check against a separate reference URL set.

How it works
1. Fetch each article and extract clean body text (trafilatura).
2. Split the text into ~25-word passages and sample N of them evenly.
3. Search each passage on the web (DuckDuckGo, exact-phrase first).
4. Open the top results and fuzzy-match the passage against the page text.
5. Score = % of sampled passages found on other websites.
6. Also compares every article against the others in the batch (TF-IDF).
7. Optionally cross-checks new articles against a reference set of existing URLs.
"""

import html
import re
import time
import random
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 20


# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_urls_from_sitemap(sitemap_url, url_filter="", max_urls=500, _depth=0):
    """Parse a sitemap (handles sitemap index files recursively)."""
    urls = []
    try:
        r = requests.get(sitemap_url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        st.warning(f"Could not read sitemap {sitemap_url}: {e}")
        return urls

    soup = BeautifulSoup(r.content, "xml")
    if soup.find("sitemapindex") and _depth < 3:
        for loc in soup.find_all("loc"):
            urls += get_urls_from_sitemap(loc.text.strip(), url_filter, max_urls, _depth + 1)
            if len(urls) >= max_urls:
                break
    else:
        for loc in soup.find_all("loc"):
            u = loc.text.strip()
            if u.endswith((".xml", ".jpg", ".png", ".pdf")):
                continue
            if url_filter and url_filter not in u:
                continue
            urls.append(u)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:max_urls]


def get_urls_from_csv(file):
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    for key in ("url", "urls", "link", "links", "page", "address"):
        if key in cols:
            col = cols[key]
            break
    else:
        col = df.columns[0]
    urls = df[col].dropna().astype(str).str.strip()
    return [u for u in urls if u.startswith("http")]


# ─────────────────────────────────────────────────────────────────────────────
# Fetching + text extraction
# ─────────────────────────────────────────────────────────────────────────────
_page_cache = {}


def fetch_text(url):
    if url in _page_cache:
        return _page_cache[url]
    text = ""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and "html" in r.headers.get("content-type", ""):
            text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
            if not text:
                soup = BeautifulSoup(r.text, "html.parser")
                text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    except Exception:
        text = ""
    _page_cache[url] = text
    return text


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def domain_of(url):
    return urlparse(url).netloc.lower().replace("www.", "")


# ─────────────────────────────────────────────────────────────────────────────
# Passage building
# ─────────────────────────────────────────────────────────────────────────────
def build_passages(text, words_per_passage=25, n_samples=10):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    passages, buf = [], []
    for s in sentences:
        s = s.strip()
        if len(s.split()) < 4:
            continue
        buf.append(s)
        if len(" ".join(buf).split()) >= words_per_passage:
            passages.append(" ".join(buf))
            buf = []
    if buf and len(" ".join(buf).split()) >= 12:
        passages.append(" ".join(buf))
    if len(passages) <= n_samples:
        return passages
    step = len(passages) / n_samples
    return [passages[int(i * step)] for i in range(n_samples)]


# ─────────────────────────────────────────────────────────────────────────────
# Web search + matching
# ─────────────────────────────────────────────────────────────────────────────
def search(query, max_results=6, retries=3):
    for attempt in range(retries):
        try:
            with DDGS() as d:
                return list(d.text(query, max_results=max_results)) or []
        except Exception:
            time.sleep(2 + attempt * 2 + random.random())
    return []


def check_passage(passage, own_domain, excluded_domains, match_threshold, verify_pages=3):
    """Return (is_match, best_score, best_url)."""
    words = passage.split()
    short = " ".join(words[:22])
    results = search(f'"{short}"') or search(short)
    results = [
        r for r in results
        if domain_of(r.get("href", "")) not in excluded_domains
        and own_domain not in domain_of(r.get("href", ""))
    ]
    best_score, best_url = 0, ""
    norm_passage = normalize(passage)

    for r in results[:verify_pages]:
        url = r.get("href", "")
        snippet = normalize((r.get("title", "") + " " + r.get("body", "")))
        s_score = fuzz.partial_ratio(normalize(short), snippet) if snippet else 0
        page = normalize(fetch_text(url))
        p_score = fuzz.partial_ratio(norm_passage, page) if page else 0
        score = max(p_score, s_score if not page else 0)
        if score > best_score:
            best_score, best_url = score, url
        if best_score >= 97:
            break
    return best_score >= match_threshold, best_score, best_url


def check_article(url, n_samples, match_threshold, excluded_domains, words_per_passage):
    text = fetch_text(url)
    wc = len(text.split())
    if wc < 80:
        return {
            "url": url, "word_count": wc, "passages_checked": 0,
            "passages_matched": 0, "plagiarism_score": None,
            "verdict": "Could not read / too short", "top_sources": "",
            "matches": [],
        }, text

    own = domain_of(url)
    passages = build_passages(text, words_per_passage, n_samples)
    matched, source_hits, matches = 0, Counter(), []
    for p in passages:
        ok, p_score, src = check_passage(p, own, excluded_domains, match_threshold)
        if ok:
            matched += 1
            source_hits[src] += 1
            matches.append({"passage": p, "score": p_score, "source": src})
        time.sleep(random.uniform(0.8, 1.6))

    checked = len(passages)
    score = round(matched / checked * 100, 1) if checked else 0.0
    verdict = "Low" if score < 15 else "Medium - review" if score < 40 else "High - likely copied"
    top = " | ".join(f"{u} ({c})" for u, c in source_hits.most_common(3))
    return {
        "url": url, "word_count": wc, "passages_checked": checked,
        "passages_matched": matched, "plagiarism_score": score,
        "verdict": verdict, "top_sources": top, "matches": matches,
    }, text


# ─────────────────────────────────────────────────────────────────────────────
# Internal duplicate check (within-batch + optional reference set)
# ─────────────────────────────────────────────────────────────────────────────
def internal_similarity(urls, texts, ref_urls=None, ref_texts=None):
    """
    Compare each article to every other article in the batch.
    If ref_urls/ref_texts are provided, also compare to that reference set
    and flag when a batch article is more similar to a reference article
    than to anything in the batch.
    """
    valid = [(u, t) for u, t in zip(urls, texts) if len(t.split()) >= 80]
    out = {u: {"closest_url": "", "similarity": 0.0, "is_ref_match": False} for u in urls}
    if len(valid) < 1:
        return out

    # combine ref into corpus if provided
    ref_valid = []
    if ref_urls and ref_texts:
        ref_valid = [(u, t) for u, t in zip(ref_urls, ref_texts) if len(t.split()) >= 80]

    all_texts = [t for _, t in valid] + [t for _, t in ref_valid]
    if len(all_texts) < 2:
        return out

    vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1)
    m = vec.fit_transform(all_texts)
    sim = cosine_similarity(m)
    n_batch = len(valid)

    for i, (u, _) in enumerate(valid):
        row = sim[i].copy()
        row[i] = 0  # exclude self
        j = row.argmax()
        best_score = round(float(row[j]) * 100, 1)
        is_ref = j >= n_batch
        best_url = (ref_valid[j - n_batch][0] if is_ref else valid[j][0]) if best_score > 0 else ""
        out[u] = {"closest_url": best_url, "similarity": best_score, "is_ref_match": is_ref}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UI ─ Page config & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Checker · KollegeApply",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
/* Layout */
.block-container {padding-top: 1.5rem !important; max-width: 1300px; padding-bottom: 3rem;}
section[data-testid="stSidebar"] {background: #0f172a;}
section[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stNumberInput input {
    background: #1e293b !important; border-color: #334155 !important; color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] hr {border-color: #334155 !important;}
section[data-testid="stSidebar"] label {color: #94a3b8 !important; font-size: .8rem !important;}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
    border-radius: 20px; padding: 2rem 2.25rem 1.75rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(139,92,246,.3);
    box-shadow: 0 20px 60px rgba(79,70,229,.25);
}
.hero h1 {color:#fff; font-size:2rem; margin:0 0 .4rem; font-weight:800; letter-spacing:-.5px;}
.hero p {color:rgba(255,255,255,.8); margin:0; font-size:1rem; line-height:1.5;}
.hero-chips {margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap;}
.chip {
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2);
    color:#e0e7ff !important; padding:4px 13px; border-radius:99px; font-size:.78rem; font-weight:500;
}

/* Step headers */
.step-label {
    display:flex; align-items:center; gap:.6rem; font-size:1rem; font-weight:700;
    color:#1e293b; margin: 1.5rem 0 .6rem;
}
.step-badge {
    width:1.75rem; height:1.75rem; border-radius:50%;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:#fff; display:inline-flex; align-items:center; justify-content:center;
    font-size:.85rem; font-weight:700; flex-shrink:0;
    box-shadow:0 2px 8px rgba(99,102,241,.4);
}

/* Cards */
.card {
    background:#fff; border-radius:16px; padding:1.25rem 1.5rem;
    border:1px solid #e2e8f0; box-shadow:0 2px 12px rgba(0,0,0,.06); margin-bottom:.75rem;
}
.card-dark {background:#0f172a; border-color:#1e293b;}

/* Metric tiles */
.metric-tile {
    background:#fff; border-radius:14px; padding:1rem 1.25rem;
    border:1px solid #e2e8f0; box-shadow:0 1px 6px rgba(0,0,0,.05); text-align:center;
}
.metric-tile .mt-val {font-size:2rem; font-weight:800; color:#1e293b; line-height:1;}
.metric-tile .mt-lab {font-size:.78rem; color:#64748b; margin-top:.3rem; font-weight:500;}
.metric-tile.red   .mt-val {color:#dc2626;}
.metric-tile.amber .mt-val {color:#d97706;}
.metric-tile.green .mt-val {color:#16a34a;}
.metric-tile.blue  .mt-val {color:#2563eb;}

/* Verdict badges */
.badge {display:inline-flex; align-items:center; gap:.3rem; padding:3px 10px;
        border-radius:99px; font-size:.78rem; font-weight:600;}
.badge-red   {background:#fef2f2; color:#dc2626; border:1px solid #fecaca;}
.badge-amber {background:#fffbeb; color:#d97706; border:1px solid #fde68a;}
.badge-green {background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0;}
.badge-gray  {background:#f8fafc; color:#64748b; border:1px solid #e2e8f0;}

/* Matched passage cards */
.match-card {
    border-left:4px solid #ef4444; padding:.7rem 1rem; margin:.5rem 0;
    border-radius:0 10px 10px 0; background:#fef2f2;
}
.match-card .match-text {font-size:.92rem; color:#1e293b; font-style:italic; line-height:1.5;}
.match-card .match-meta {margin-top:.4rem; font-size:.8rem; color:#64748b;}
.match-card .match-meta a {color:#6366f1; text-decoration:none;}
.match-card .match-meta a:hover {text-decoration:underline;}

/* Internal dup warning */
.dup-warning {
    border-left:4px solid #f59e0b; padding:.7rem 1rem; margin:.5rem 0;
    border-radius:0 10px 10px 0; background:#fffbeb;
    font-size:.9rem; color:#92400e;
}
.dup-warning .dup-ref {color:#b45309; font-weight:600;}

/* Info callout */
.callout {
    background:#f0f9ff; border:1px solid #bae6fd; border-radius:12px;
    padding:.8rem 1rem; font-size:.88rem; color:#0c4a6e; line-height:1.5;
}
.callout b {color:#0369a1;}

/* Progress indicator */
.prog-url {font-size:.85rem; color:#64748b; font-family:monospace; overflow:hidden;
           text-overflow:ellipsis; white-space:nowrap; max-width:100%;}

/* Score ring colors */
div[data-testid="stMetric"] {border:1px solid #e2e8f0; border-radius:12px; padding:.75rem 1rem;}

/* Streamlit tweaks */
div[data-testid="stRadio"] > label:first-child {display:none;}
div[data-testid="stExpander"] {border-radius:12px !important;}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border:none !important; font-weight:600 !important;
    box-shadow:0 4px 14px rgba(99,102,241,.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform:translateY(-1px); box-shadow:0 6px 20px rgba(99,102,241,.45) !important;
}
.stDownloadButton > button {border-radius:10px !important;}
div[data-testid="stSegmentedControl"] {justify-content:flex-start;}
</style>
""", unsafe_allow_html=True)


# ─── helpers ─────────────────────────────────────────────────────────────────
VERDICT_META = {
    "Low":                 ("badge-green", "🟢"),
    "Medium - review":     ("badge-amber", "🟡"),
    "High - likely copied":("badge-red",   "🔴"),
}

def badge_html(v):
    cls, icon = VERDICT_META.get(v, ("badge-gray", "⚪"))
    label = v.replace("High - likely copied", "High risk").replace("Medium - review", "Review")
    return f'<span class="badge {cls}">{icon} {html.escape(label)}</span>'

def pretty_verdict(v):
    _, icon = VERDICT_META.get(v, ("", "⚪"))
    return f"{icon} {v}"

def step(n, label):
    st.markdown(
        f'<div class="step-label"><span class="step-badge">{n}</span>{label}</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    st.markdown("**🌐 Your site**")
    own_domain_in = st.text_input(
        "Your domain", "kollegeapply.com",
        help="URLs on this domain are never counted as an external source.",
    )

    st.divider()
    st.markdown("**🎯 Accuracy**")
    n_samples = st.slider("Passages per article", 4, 25, 10,
                          help="More = more accurate but slower. 10 is a good balance.")
    words_per_passage = st.slider("Words per passage", 15, 40, 25,
                                  help="Length of each text chunk that gets searched.")
    match_threshold = st.slider("Match strictness (%)", 70, 100, 85,
                                help="Minimum fuzzy-match score to count as plagiarism.")

    st.divider()
    st.markdown("**🚫 Ignored domains**")
    extra_excl = st.text_area(
        "One per line", "wikipedia.org\nyoutube.com",
        help="Matches on these sites won't count as plagiarism.",
        height=90,
    )

    st.divider()
    st.markdown("**📊 Run limits**")
    max_urls = st.number_input("Max URLs to check", 1, 500, 30)

    st.divider()
    st.caption(
        "**Score guide:**  \n"
        "🟢 **<15%** — original  \n"
        "🟡 **15–40%** — review  \n"
        "🔴 **>40%** — high risk"
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 Bulk Plagiarism Checker</h1>
  <p>Check many articles at once against the open web · spot near-duplicates inside your site · cross-check new content against existing articles</p>
  <div class="hero-chips">
    <span class="chip">✅ 100% free</span>
    <span class="chip">🔑 No API keys</span>
    <span class="chip">📂 CSV · paste · sitemap</span>
    <span class="chip">🔁 Internal duplicate check</span>
    <span class="chip">📚 Reference set comparison</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Step 1: Input ────────────────────────────────────────────────────────────
step(1, "Add articles to check")

input_mode = st.radio(
    "How to add URLs",
    ["📋 Paste URLs", "📄 Upload CSV", "🗺️ Sitemap"],
    horizontal=True,
)
urls: list[str] = []

with st.container(border=True):
    if input_mode == "📋 Paste URLs":
        raw = st.text_area(
            "Paste one URL per line",
            placeholder="https://example.com/article-1\nhttps://example.com/article-2",
            height=150,
        )
        if raw.strip():
            urls = [u.strip() for u in raw.strip().splitlines()
                    if u.strip().startswith("http")]
            if not urls:
                st.error("No valid URLs found. Make sure each line starts with http.")
            else:
                st.success(f"✅ {len(urls)} URL{'s' if len(urls)!=1 else ''} ready")

    elif input_mode == "📄 Upload CSV":
        f = st.file_uploader(
            "CSV file — needs a `url` column (or URLs in the first column)",
            type=["csv"],
        )
        if f:
            urls = get_urls_from_csv(f)
            if not urls:
                st.error("No URLs starting with http found in that file.")
            else:
                st.success(f"✅ {len(urls)} URL{'s' if len(urls)!=1 else ''} loaded from CSV")

    else:  # Sitemap
        c1, c2, c3 = st.columns([4, 2, 1], vertical_alignment="bottom")
        sm_url = c1.text_input("Sitemap URL", "https://www.kollegeapply.com/index-updates.xml")
        sm_filter = c2.text_input("Only URLs containing", "", placeholder="/blog/")
        if c3.button("Load", use_container_width=True):
            with st.spinner("Reading sitemap…"):
                loaded = get_urls_from_sitemap(sm_url, sm_filter, int(max_urls) * 10)
            if loaded:
                st.session_state["sm_urls"] = loaded
                st.success(f"✅ {len(loaded)} URLs found in sitemap")
            else:
                st.error("No page URLs found in that sitemap.")
        urls = st.session_state.get("sm_urls", [])


# ─── Step 1b: Reference URL set (for cross-checking) ─────────────────────────
with st.expander("📚 Add reference URLs (optional — check against your existing articles)"):
    st.markdown("""
<div class="callout">
<b>What is this?</b> Add your already-published articles here.
The tool will compare new articles against them and flag when a new article
looks too similar to one you already have — useful for catching near-duplicate
or spun content before publishing.
</div>
""", unsafe_allow_html=True)
    st.write("")
    ref_mode = st.radio("Reference source", ["📋 Paste", "📄 CSV", "🗺️ Sitemap"],
                        horizontal=True, key="ref_mode")
    ref_urls_raw: list[str] = []

    if ref_mode == "📋 Paste":
        ref_raw = st.text_area("Reference URLs (one per line)", height=100, key="ref_paste")
        if ref_raw.strip():
            ref_urls_raw = [u.strip() for u in ref_raw.strip().splitlines()
                            if u.strip().startswith("http")]
    elif ref_mode == "📄 CSV":
        rf = st.file_uploader("Reference CSV", type=["csv"], key="ref_csv")
        if rf:
            ref_urls_raw = get_urls_from_csv(rf)
    else:
        rc1, rc2, rc3 = st.columns([4, 2, 1], vertical_alignment="bottom")
        ref_sm = rc1.text_input("Reference sitemap URL", key="ref_sm")
        ref_flt = rc2.text_input("Filter", "", placeholder="/blog/", key="ref_flt")
        if rc3.button("Load", key="ref_sm_btn", use_container_width=True):
            with st.spinner("Reading reference sitemap…"):
                loaded_ref = get_urls_from_sitemap(ref_sm, ref_flt, 2000)
            st.session_state["ref_sm_urls"] = loaded_ref
        ref_urls_raw = st.session_state.get("ref_sm_urls", [])

    if ref_urls_raw:
        st.info(f"📚 {len(ref_urls_raw)} reference URLs will be used for cross-checking.", icon="✅")
    st.session_state["ref_urls"] = ref_urls_raw


# ─── Step 2: Run ──────────────────────────────────────────────────────────────
if not urls and "results" not in st.session_state:
    st.info("👆 Add some URLs above to get started.", icon="ℹ️")

if urls:
    total_found = len(urls)
    urls = urls[: int(max_urls)]
    step(2, "Run the check")

    with st.container(border=True):
        est_sec = len(urls) * n_samples * 5
        est_str = f"~{max(1, round(est_sec/60))} min" if est_sec >= 60 else f"~{est_sec}s"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("URLs to check", len(urls),
                  delta=f"{total_found - len(urls)} skipped" if total_found > len(urls) else None,
                  delta_color="off")
        c2.metric("Passages per article", n_samples)
        c3.metric("Est. time", est_str)
        c4.metric("Reference URLs", len(st.session_state.get("ref_urls", [])))

        with st.expander(f"Preview {len(urls)} URLs"):
            st.dataframe(
                pd.DataFrame({"#": range(1, len(urls)+1), "URL": urls}),
                hide_index=True, use_container_width=True,
                column_config={"URL": st.column_config.LinkColumn()},
            )

        run = st.button("🚀 Run plagiarism check", type="primary", use_container_width=True)

    if run:
        excluded = {d.strip().lower() for d in extra_excl.splitlines() if d.strip()}
        if own_domain_in.strip():
            excluded.add(own_domain_in.strip().lower().replace("www.", ""))

        rows, texts = [], []
        start_time = time.time()

        with st.status("⏳ Checking articles…", expanded=True) as status:
            prog_bar = st.progress(0.0)
            prog_text = st.empty()
            live_table = st.empty()

            for i, u in enumerate(urls, 1):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = int((len(urls) - i) / rate) if rate > 0 else 0
                eta = f" · ETA {remaining//60}m {remaining%60}s" if i > 1 else ""

                prog_text.markdown(
                    f"**{i} / {len(urls)}**{eta}  \n"
                    f'<span class="prog-url">{html.escape(u)}</span>',
                    unsafe_allow_html=True,
                )
                try:
                    row, text = check_article(u, n_samples, match_threshold, excluded, words_per_passage)
                except Exception as e:
                    row, text = {
                        "url": u, "word_count": 0, "passages_checked": 0,
                        "passages_matched": 0, "plagiarism_score": None,
                        "verdict": f"Error: {e}", "top_sources": "", "matches": [],
                    }, ""
                rows.append(row)
                texts.append(text)

                # live preview table
                preview = pd.DataFrame([
                    {"Article": r["url"],
                     "Score": f"{r['plagiarism_score']:.0f}%" if r["plagiarism_score"] is not None else "–",
                     "Verdict": pretty_verdict(r["verdict"])}
                    for r in rows
                ])
                live_table.dataframe(preview, hide_index=True, use_container_width=True,
                                     column_config={"Article": st.column_config.LinkColumn()})
                prog_bar.progress(i / len(urls))

            # Internal + reference similarity
            prog_text.markdown("🔁 Comparing articles against each other…")
            ref_urls_list = st.session_state.get("ref_urls", [])
            ref_texts = []
            if ref_urls_list:
                prog_text.markdown("📚 Fetching reference articles for comparison…")
                for ru in ref_urls_list[:500]:
                    ref_texts.append(fetch_text(ru))

            sim_results = internal_similarity(urls, texts, ref_urls_list, ref_texts)
            for r in rows:
                sim = sim_results[r["url"]]
                r["closest_url"] = sim["closest_url"]
                r["similarity_%"] = sim["similarity"]
                r["is_ref_match"] = sim["is_ref_match"]

            live_table.empty()
            prog_text.empty()
            total_elapsed = time.time() - start_time
            status.update(
                label=f"✅ Checked {len(rows)} articles in {total_elapsed/60:.1f} min",
                state="complete", expanded=False,
            )

        st.session_state["results"] = rows
        st.session_state["run_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p")


# ─── Step 3: Results ──────────────────────────────────────────────────────────
rows = st.session_state.get("results")
if rows:
    df = pd.DataFrame(rows)
    scored = df[df["plagiarism_score"].notna()]
    run_time = st.session_state.get("run_time", "")

    # ── header ──
    hc1, hc2 = st.columns([6, 1], vertical_alignment="bottom")
    with hc1:
        step(3, f"Results  {'· ' + run_time if run_time else ''}")
    with hc2:
        if st.button("🗑️ Clear", use_container_width=True):
            for k in ("results", "run_time", "sm_urls", "ref_sm_urls"):
                st.session_state.pop(k, None)
            st.rerun()

    # ── summary metrics ──
    n_high    = int((df["verdict"] == "High - likely copied").sum())
    n_med     = int((df["verdict"] == "Medium - review").sum())
    n_ok      = int((df["verdict"] == "Low").sum())
    n_err     = int(df["plagiarism_score"].isna().sum())
    avg_score = f"{scored['plagiarism_score'].mean():.1f}%" if len(scored) else "–"
    n_ref_dup = int(df.get("is_ref_match", pd.Series([False]*len(df))).sum()) if "is_ref_match" in df.columns else 0

    col_metrics = st.columns(6)
    tiles = [
        ("Articles", len(df), "blue"),
        ("Avg score", avg_score, "blue"),
        ("🔴 High risk", n_high, "red"),
        ("🟡 Review", n_med, "amber"),
        ("🟢 Original", n_ok, "green"),
        ("📚 Ref matches", n_ref_dup, "amber"),
    ]
    for col, (label, val, color) in zip(col_metrics, tiles):
        col.markdown(
            f'<div class="metric-tile {color}"><div class="mt-val">{val}</div>'
            f'<div class="mt-lab">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # ── view toggle ──
    view = st.segmented_control(
        "View",
        ["🌐 Web plagiarism", "🔁 Internal duplicates", "📚 Ref cross-check"],
        default="🌐 Web plagiarism",
        label_visibility="collapsed",
    ) or "🌐 Web plagiarism"

    # ════════════════════════════════════════════
    # VIEW 1: Web plagiarism
    # ════════════════════════════════════════════
    if view == "🌐 Web plagiarism":
        vdf = df.copy()
        vdf["verdict_pretty"] = vdf["verdict"].map(pretty_verdict)

        # filter
        fc1, fc2 = st.columns([3, 1])
        verdict_options = sorted(vdf["verdict_pretty"].unique())
        chosen = fc1.multiselect("Filter by verdict", verdict_options, default=verdict_options)
        sort_col = fc2.selectbox("Sort by", ["Score ↓", "Score ↑", "Words ↓"], index=0)

        vdf = vdf[vdf["verdict_pretty"].isin(chosen)]
        if sort_col == "Score ↓":
            vdf = vdf.sort_values("plagiarism_score", ascending=False, na_position="last")
        elif sort_col == "Score ↑":
            vdf = vdf.sort_values("plagiarism_score", ascending=True, na_position="last")
        else:
            vdf = vdf.sort_values("word_count", ascending=False, na_position="last")

        st.dataframe(
            vdf[["url", "plagiarism_score", "verdict_pretty", "passages_matched",
                 "passages_checked", "word_count", "top_sources"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("Article", width="large"),
                "plagiarism_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.0f%%", width="small"),
                "verdict_pretty": st.column_config.TextColumn("Verdict", width="medium"),
                "passages_matched": st.column_config.NumberColumn("Matched ✓", width="small"),
                "passages_checked": st.column_config.NumberColumn("Checked", width="small"),
                "word_count": st.column_config.NumberColumn("Words", width="small"),
                "top_sources": st.column_config.TextColumn("Top sources", width="large"),
            },
        )

        # matched passages drill-down
        flagged = [r for r in rows if r.get("matches")]
        flagged.sort(key=lambda r: r.get("plagiarism_score") or 0, reverse=True)
        if flagged:
            st.markdown("#### 🔎 Matched passages")
            st.caption("Expand any article to see the exact passages found on other websites.")
            for r in flagged:
                score_str = f"{r['plagiarism_score']:.0f}%"
                _, icon = VERDICT_META.get(r["verdict"], ("", "⚪"))
                label = f"{icon} {score_str} — {r['url']}"
                with st.expander(label, expanded=r["verdict"] == "High - likely copied"):
                    for match in r["matches"]:
                        src_esc = html.escape(match["source"], quote=True)
                        st.markdown(
                            f'<div class="match-card">'
                            f'<div class="match-text">"{html.escape(match["passage"])}"</div>'
                            f'<div class="match-meta">'
                            f'{match["score"]:.0f}% match · '
                            f'<a href="{src_esc}" target="_blank">{src_esc}</a>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
        elif len(scored):
            st.success("✅ No passages were found on other websites for the checked articles.", icon="✅")

    # ════════════════════════════════════════════
    # VIEW 2: Internal duplicates
    # ════════════════════════════════════════════
    elif view == "🔁 Internal duplicates":
        st.markdown("""
<div class="callout">
<b>What you're seeing:</b> Each article is compared to every other article
in <b>this batch</b> using TF-IDF cosine similarity.
A score above <b>60%</b> usually means heavy content overlap —
the two articles likely cover the same topic with similar wording.
</div>
""", unsafe_allow_html=True)
        st.write("")

        dup_threshold = st.slider("Flag pairs above (%)", 30, 100, 60, key="dup_thresh")

        if "closest_url" in df.columns:
            idf = df[["url", "closest_url", "similarity_%"]].copy()
            idf = idf[idf.get("is_ref_match", False) == False] if "is_ref_match" in df.columns else idf
            idf = idf.sort_values("similarity_%", ascending=False)
            idf["flagged"] = idf["similarity_%"] >= dup_threshold

            flagged_dups = idf[idf["flagged"]]
            if not flagged_dups.empty:
                st.warning(f"⚠️ {len(flagged_dups)} article(s) exceed the {dup_threshold}% similarity threshold with another article in the batch.", icon="⚠️")

            st.dataframe(
                idf[["url", "closest_url", "similarity_%"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "url": st.column_config.LinkColumn("Article", width="large"),
                    "closest_url": st.column_config.LinkColumn("Most similar article", width="large"),
                    "similarity_%": st.column_config.ProgressColumn(
                        "Similarity", min_value=0, max_value=100, format="%.0f%%", width="small"),
                },
            )
        else:
            st.info("Run the check first to see internal similarity scores.")

    # ════════════════════════════════════════════
    # VIEW 3: Reference cross-check
    # ════════════════════════════════════════════
    else:
        if "is_ref_match" not in df.columns or not df["is_ref_match"].any():
            st.markdown("""
<div class="callout">
<b>No reference URLs were used in this run.</b><br>
Add your existing published articles in the <b>Reference URLs</b> section above
(Step 1), then re-run the check. The tool will compare each new article against
your reference set and flag anything that looks too similar —
useful for catching duplicate or spun content before it goes live.
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="callout">
<b>Reference cross-check:</b> These articles from your batch look similar to
an article in your reference set. They may be duplicates or rewrites of existing content.
</div>
""", unsafe_allow_html=True)
            st.write("")
            ref_thresh = st.slider("Show pairs above (%)", 30, 100, 50, key="ref_thresh")
            rdf = df[df["is_ref_match"] == True][["url", "closest_url", "similarity_%"]].copy()
            rdf = rdf[rdf["similarity_%"] >= ref_thresh].sort_values("similarity_%", ascending=False)
            if rdf.empty:
                st.success(f"No new articles exceed {ref_thresh}% similarity to any reference article.")
            else:
                st.error(f"⚠️ {len(rdf)} article(s) are suspiciously similar to existing reference articles.")
                st.dataframe(
                    rdf,
                    hide_index=True, use_container_width=True,
                    column_config={
                        "url": st.column_config.LinkColumn("New Article", width="large"),
                        "closest_url": st.column_config.LinkColumn("Matches reference", width="large"),
                        "similarity_%": st.column_config.ProgressColumn(
                            "Similarity", min_value=0, max_value=100, format="%.0f%%", width="small"),
                    },
                )
                for _, rrow in rdf.iterrows():
                    st.markdown(
                        f'<div class="dup-warning">🚨 <b>{html.escape(rrow["url"])}</b>'
                        f' is <span class="dup-ref">{rrow["similarity_%"]:.0f}% similar</span>'
                        f' to existing article: <a href="{html.escape(rrow["closest_url"], quote=True)}"'
                        f' target="_blank">{html.escape(rrow["closest_url"])}</a></div>',
                        unsafe_allow_html=True,
                    )

    # ── download ──
    st.write("")
    export_df = df.drop(columns=["matches"], errors="ignore")
    st.download_button(
        "⬇️ Download full results CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        f"plagiarism_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        type="primary",
    )
