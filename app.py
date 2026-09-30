"""
KollegeApply Bulk Plagiarism Checker (100% free, no API keys)

Input : CSV of URLs  OR  a sitemap URL
Output: per-article plagiarism score, verdict, top matching sources,
        plus internal-duplicate check against the other articles in the batch.

How it works
1. Fetch each article and extract clean body text (trafilatura).
2. Split the text into ~25-word passages and sample N of them evenly.
3. Search each passage on the web (DuckDuckGo, exact-phrase first).
4. Open the top results and fuzzy-match the passage against the page text.
5. Score = % of sampled passages found on other websites.
6. Also compares every article against the others in the batch (TF-IDF).
"""

import html
import re
import time
import random
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ddgs import DDGS  # newer package name
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 20


# ----------------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------------
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
            urls += get_urls_from_sitemap(
                loc.text.strip(), url_filter, max_urls, _depth + 1
            )
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
    # de-duplicate, keep order
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


# ----------------------------------------------------------------------------
# Fetching + text extraction
# ----------------------------------------------------------------------------
_page_cache = {}


def fetch_text(url):
    if url in _page_cache:
        return _page_cache[url]
    text = ""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and "html" in r.headers.get("content-type", ""):
            text = trafilatura.extract(
                r.text, include_comments=False, include_tables=False
            ) or ""
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


# ----------------------------------------------------------------------------
# Passage building
# ----------------------------------------------------------------------------
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
    # sample evenly across the article
    step = len(passages) / n_samples
    return [passages[int(i * step)] for i in range(n_samples)]


# ----------------------------------------------------------------------------
# Web search + matching
# ----------------------------------------------------------------------------
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
        # quick check on the search snippet
        snippet = normalize((r.get("title", "") + " " + r.get("body", "")))
        s_score = fuzz.partial_ratio(normalize(short), snippet) if snippet else 0
        # deep check on the actual page
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
        time.sleep(random.uniform(0.8, 1.6))  # be polite, avoid rate limit

    checked = len(passages)
    score = round(matched / checked * 100, 1) if checked else 0.0
    verdict = "Low" if score < 15 else "Medium - review" if score < 40 else "High - likely copied"
    top = " | ".join(f"{u} ({c})" for u, c in source_hits.most_common(3))
    return {
        "url": url, "word_count": wc, "passages_checked": checked,
        "passages_matched": matched, "plagiarism_score": score,
        "verdict": verdict, "top_sources": top, "matches": matches,
    }, text


# ----------------------------------------------------------------------------
# Internal duplicate check (between articles in the same batch)
# ----------------------------------------------------------------------------
def internal_similarity(urls, texts):
    valid = [(u, t) for u, t in zip(urls, texts) if len(t.split()) >= 80]
    out = {u: ("", 0.0) for u in urls}
    if len(valid) < 2:
        return out
    vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1)
    m = vec.fit_transform([t for _, t in valid])
    sim = cosine_similarity(m)
    for i, (u, _) in enumerate(valid):
        sim[i, i] = 0
        j = sim[i].argmax()
        out[u] = (valid[j][0], round(float(sim[i, j]) * 100, 1))
    return out




# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plagiarism Checker · KollegeApply",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 3.5rem; max-width: 1250px;}
    .hero {
        padding: 1.5rem 1.75rem; border-radius: 16px; margin-bottom: 1.25rem;
        background: linear-gradient(135deg, #4338ca 0%, #7c3aed 100%); color: #fff;
    }
    .hero h1 {color: #fff; font-size: 1.85rem; margin: 0 0 .35rem 0; padding: 0;}
    .hero p {margin: 0; opacity: .92; font-size: 1rem;}
    .hero .chips {margin-top: .9rem; display: flex; gap: .5rem; flex-wrap: wrap;}
    .hero .chip {
        background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
        padding: 3px 11px; border-radius: 999px; font-size: .8rem;
    }
    .step {font-weight: 600; font-size: 1.05rem; margin: .25rem 0 .5rem 0;}
    .step span {
        display: inline-flex; align-items: center; justify-content: center;
        width: 1.6rem; height: 1.6rem; border-radius: 50%; margin-right: .5rem;
        background: #7c3aed; color: #fff; font-size: .85rem;
    }
    .match {
        border-left: 4px solid #ef4444; padding: .55rem .8rem; margin: .45rem 0;
        border-radius: 6px; background: rgba(239,68,68,.07); font-size: .92rem;
    }
    .match small {display: block; margin-top: .3rem; opacity: .75;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22); border-radius: 12px;
        padding: .75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔍 Bulk Plagiarism Checker</h1>
      <p>Check many articles at once against the open web, and spot near-duplicates inside your own site.</p>
      <div class="chips">
        <span class="chip">100% free</span>
        <span class="chip">No API keys</span>
        <span class="chip">CSV or sitemap input</span>
        <span class="chip">Internal duplicate check</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

VERDICT_ICON = {
    "Low": "🟢 Low",
    "Medium - review": "🟡 Medium – review",
    "High - likely copied": "🔴 High – likely copied",
}


def pretty_verdict(v):
    if v in VERDICT_ICON:
        return VERDICT_ICON[v]
    return f"⚪ {v}"


# ---- Sidebar: settings -----------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("**Your site**")
    own_domain_in = st.text_input(
        "Your domain", "kollegeapply.com",
        help="Pages on this domain are never counted as a source.",
    )

    st.markdown("**Accuracy**")
    n_samples = st.slider("Passages checked per article", 4, 25, 10,
                          help="More passages = more accurate, but slower.")
    words_per_passage = st.slider("Words per passage", 15, 40, 25)
    match_threshold = st.slider("Match strictness (%)", 70, 100, 85,
                                help="Higher = only near-exact copies count.")

    with st.expander("Ignored domains"):
        extra_excl = st.text_area(
            "One per line",
            "wikipedia.org\nyoutube.com",
            help="Matches on these sites won't count as plagiarism.",
        )

    st.markdown("**Run limits**")
    max_urls = st.number_input("Max URLs per run", 1, 500, 30)

    st.divider()
    st.caption(
        "Score = % of sampled passages found on other websites. "
        "🟢 under 15% · 🟡 15–40% · 🔴 40%+"
    )

# ---- Step 1: input ---------------------------------------------------------
st.markdown('<div class="step"><span>1</span>Add the articles to check</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    mode = st.radio("Input type", ["📄 Upload CSV", "🗺️ Sitemap URL"],
                    horizontal=True, label_visibility="collapsed")
    urls = []

    if mode.endswith("Upload CSV"):
        f = st.file_uploader(
            "CSV file with a `url` column (or URLs in the first column)",
            type=["csv"],
        )
        if f:
            urls = get_urls_from_csv(f)
            if not urls:
                st.error("No URLs starting with http were found in that file.")
    else:
        c1, c2, c3 = st.columns([4, 2, 1], vertical_alignment="bottom")
        sm = c1.text_input("Sitemap URL", "https://www.kollegeapply.com/index-updates.xml")
        flt = c2.text_input("Only URLs containing", "", placeholder="e.g. /blog/",
                            help="Skip non-article pages such as /tag/ or /author/.")
        if c3.button("Load", use_container_width=True):
            with st.spinner("Reading sitemap…"):
                st.session_state["sm_urls"] = get_urls_from_sitemap(
                    sm, flt, int(max_urls) * 10
                )
            if not st.session_state["sm_urls"]:
                st.error("No page URLs were found in that sitemap.")
        urls = st.session_state.get("sm_urls", [])

if not urls and "results" not in st.session_state:
    st.info("Upload a CSV or load a sitemap to get started.", icon="👆")

# ---- Step 2: run -----------------------------------------------------------
if urls:
    total_found = len(urls)
    urls = urls[: int(max_urls)]
    st.markdown('<div class="step"><span>2</span>Run the check</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        # ~5s per passage (search + page fetch + polite delay)
        est_min = max(1, round(len(urls) * n_samples * 5 / 60))
        c1, c2, c3 = st.columns(3)
        c1.metric("URLs to check", len(urls),
                  help=f"{total_found} found; limited by 'Max URLs per run'."
                  if total_found > len(urls) else None)
        c2.metric("Passages per article", n_samples)
        c3.metric("Estimated time", f"~{est_min} min")

        with st.expander(f"Preview the {len(urls)} URLs"):
            st.dataframe(pd.DataFrame({"url": urls}), hide_index=True,
                         use_container_width=True,
                         column_config={"url": st.column_config.LinkColumn("URL")})

        run = st.button("🚀 Run plagiarism check", type="primary",
                        use_container_width=True)

    if run:
        excluded = {d.strip().lower() for d in extra_excl.splitlines() if d.strip()}
        if own_domain_in.strip():
            excluded.add(own_domain_in.strip().lower().replace("www.", ""))
        rows, texts = [], []

        with st.status("Checking articles…", expanded=True) as status:
            bar = st.progress(0.0)
            current = st.empty()
            live = st.empty()
            for i, u in enumerate(urls, 1):
                current.markdown(f"**{i}/{len(urls)}** · `{u}`")
                try:
                    row, text = check_article(
                        u, n_samples, match_threshold, excluded, words_per_passage
                    )
                except Exception as e:
                    row, text = {
                        "url": u, "word_count": 0, "passages_checked": 0,
                        "passages_matched": 0, "plagiarism_score": None,
                        "verdict": f"Error: {e}", "top_sources": "", "matches": [],
                    }, ""
                rows.append(row)
                texts.append(text)
                live.dataframe(
                    pd.DataFrame(rows)[["url", "plagiarism_score", "verdict"]],
                    hide_index=True, use_container_width=True,
                )
                bar.progress(i / len(urls))

            current.markdown("Comparing articles against each other…")
            internal = internal_similarity(urls, texts)
            for r in rows:
                r["most_similar_internal_url"], r["internal_similarity_%"] = internal[r["url"]]
            live.empty()
            current.empty()
            status.update(label=f"Checked {len(rows)} articles", state="complete",
                          expanded=False)

        st.session_state["results"] = rows

# ---- Step 3: results -------------------------------------------------------
rows = st.session_state.get("results")
if rows:
    df = pd.DataFrame(rows)
    scored = df[df["plagiarism_score"].notna()]

    head_l, head_r = st.columns([5, 1], vertical_alignment="bottom")
    head_l.markdown('<div class="step"><span>3</span>Results</div>',
                    unsafe_allow_html=True)
    if head_r.button("Clear results", use_container_width=True):
        del st.session_state["results"]
        st.rerun()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Articles", len(df))
    m2.metric("Average score",
              f"{scored['plagiarism_score'].mean():.1f}%" if len(scored) else "–")
    m3.metric("🔴 High", int((df["verdict"] == "High - likely copied").sum()))
    m4.metric("🟡 Review", int((df["verdict"] == "Medium - review").sum()))
    m5.metric("⚪ Unreadable", int(df["plagiarism_score"].isna().sum()))

    view_choice = st.segmented_control(
        "View", ["🌐 Web matches", "🔁 Internal duplicates"],
        default="🌐 Web matches", label_visibility="collapsed",
    ) or "🌐 Web matches"

    if view_choice == "🌐 Web matches":
        view = df.copy()
        view["verdict"] = view["verdict"].map(pretty_verdict)
        options = sorted(view["verdict"].unique())
        chosen = st.multiselect("Show verdicts", options, default=options)
        view = view[view["verdict"].isin(chosen)].sort_values(
            "plagiarism_score", ascending=False, na_position="last"
        )
        st.dataframe(
            view[["url", "plagiarism_score", "verdict", "passages_matched",
                  "passages_checked", "word_count", "top_sources"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("Article", width="large"),
                "plagiarism_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.0f%%"),
                "verdict": st.column_config.TextColumn("Verdict"),
                "passages_matched": st.column_config.NumberColumn("Matched"),
                "passages_checked": st.column_config.NumberColumn("Checked"),
                "word_count": st.column_config.NumberColumn("Words"),
                "top_sources": st.column_config.TextColumn("Top sources", width="large"),
            },
        )

        flagged = [r for r in rows if r.get("matches")]
        flagged.sort(key=lambda r: r["plagiarism_score"] or 0, reverse=True)
        if flagged:
            st.markdown("##### Matched passages")
            for r in flagged:
                label = (f"{pretty_verdict(r['verdict'])} · {r['plagiarism_score']:.0f}% · "
                         f"{r['url']}")
                with st.expander(label):
                    for m in r["matches"]:
                        src = html.escape(m["source"], quote=True)
                        st.markdown(
                            f'<div class="match">“{html.escape(m["passage"])}”'
                            f'<small>{m["score"]:.0f}% match · '
                            f'<a href="{src}" target="_blank">{src}</a>'
                            f'</small></div>',
                            unsafe_allow_html=True,
                        )
        elif len(scored):
            st.success("No passages were found on other websites.", icon="✅")

    else:
        st.caption("How similar each article is to the closest other article in this batch "
                   "(TF-IDF cosine similarity). 60%+ usually means heavy overlap.")
        idf = df[["url", "most_similar_internal_url", "internal_similarity_%"]] \
            .sort_values("internal_similarity_%", ascending=False)
        st.dataframe(
            idf, hide_index=True, use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("Article", width="large"),
                "most_similar_internal_url": st.column_config.LinkColumn(
                    "Most similar article", width="large"),
                "internal_similarity_%": st.column_config.ProgressColumn(
                    "Similarity", min_value=0, max_value=100, format="%.0f%%"),
            },
        )

    st.download_button(
        "⬇️ Download results CSV",
        df.drop(columns=["matches"]).to_csv(index=False).encode("utf-8"),
        "plagiarism_results.csv",
        "text/csv",
        type="primary",
    )
