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
        }, text

    own = domain_of(url)
    passages = build_passages(text, words_per_passage, n_samples)
    matched, source_hits = 0, Counter()
    for p in passages:
        ok, _score, src = check_passage(p, own, excluded_domains, match_threshold)
        if ok:
            matched += 1
            source_hits[src] += 1
        time.sleep(random.uniform(0.8, 1.6))  # be polite, avoid rate limit

    checked = len(passages)
    score = round(matched / checked * 100, 1) if checked else 0.0
    verdict = "Low" if score < 15 else "Medium - review" if score < 40 else "High - likely copied"
    top = " | ".join(f"{u} ({c})" for u, c in source_hits.most_common(3))
    return {
        "url": url, "word_count": wc, "passages_checked": checked,
        "passages_matched": matched, "plagiarism_score": score,
        "verdict": verdict, "top_sources": top,
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
st.set_page_config(page_title="KollegeApply Plagiarism Checker", layout="wide")
st.title("KollegeApply - Bulk Plagiarism Checker")
st.caption("Free tool. Upload a CSV of article URLs or paste a sitemap link.")

with st.sidebar:
    st.header("Settings")
    own_domain_in = st.text_input("Your domain (ignored as a source)", "kollegeapply.com")
    n_samples = st.slider("Passages to check per article", 4, 25, 10,
                          help="More = more accurate but slower.")
    words_per_passage = st.slider("Words per passage", 15, 40, 25)
    match_threshold = st.slider("Match strictness (%)", 70, 100, 85,
                                help="Higher = only near-exact copies count.")
    extra_excl = st.text_area(
        "Domains to ignore (one per line)",
        "wikipedia.org\nyoutube.com",
        help="Quotes from these won't count as plagiarism.",
    )
    max_urls = st.number_input("Max URLs per run", 1, 500, 30)

mode = st.radio("Input type", ["Upload CSV", "Sitemap URL"], horizontal=True)
urls = []

if mode == "Upload CSV":
    f = st.file_uploader("CSV with a 'url' column", type=["csv"])
    if f:
        urls = get_urls_from_csv(f)
else:
    sm = st.text_input("Sitemap URL", "https://www.kollegeapply.com/sitemap.xml")
    flt = st.text_input("Only URLs containing (optional)", "",
                        help="e.g. /blog/ or /articles/ to skip non-article pages")
    if st.button("Load sitemap"):
        st.session_state["sm_urls"] = get_urls_from_sitemap(sm, flt, int(max_urls) * 10)
    urls = st.session_state.get("sm_urls", [])

if urls:
    urls = urls[: int(max_urls)]
    st.success(f"{len(urls)} URLs ready")
    with st.expander("Preview URLs"):
        st.write(urls)

    if st.button("Run plagiarism check", type="primary"):
        excluded = {d.strip().lower() for d in extra_excl.splitlines() if d.strip()}
        rows, texts = [], []
        bar = st.progress(0.0)
        status = st.empty()
        table = st.empty()

        for i, u in enumerate(urls, 1):
            status.info(f"Checking {i}/{len(urls)}: {u}")
            try:
                row, text = check_article(
                    u, n_samples, match_threshold, excluded, words_per_passage
                )
            except Exception as e:
                row, text = {
                    "url": u, "word_count": 0, "passages_checked": 0,
                    "passages_matched": 0, "plagiarism_score": None,
                    "verdict": f"Error: {e}", "top_sources": "",
                }, ""
            rows.append(row)
            texts.append(text)
            table.dataframe(pd.DataFrame(rows), use_container_width=True)
            bar.progress(i / len(urls))

        status.info("Comparing articles against each other...")
        internal = internal_similarity(urls, texts)
        df = pd.DataFrame(rows)
        df["most_similar_internal_url"] = df["url"].map(lambda u: internal[u][0])
        df["internal_similarity_%"] = df["url"].map(lambda u: internal[u][1])
        status.success("Done")
        table.dataframe(df, use_container_width=True)

        st.download_button(
            "Download results CSV",
            df.to_csv(index=False).encode("utf-8"),
            "plagiarism_results.csv",
            "text/csv",
        )
