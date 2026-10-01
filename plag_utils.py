"""
Shared backend utilities for the KollegeApply Plagiarism Checker.
Compatible with Python 3.9+.
"""

import re
import time
import random
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import requests
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
_page_cache = {}


def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def domain_of(url):
    return urlparse(url).netloc.lower().replace("www.", "")


def fetch_text(url):
    """Fetch a URL and extract article text."""
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


def extract_text_from_html(html_content):
    """Extract clean article text from a stored HTML string (no network call)."""
    if not html_content or not isinstance(html_content, str):
        return ""
    try:
        text = trafilatura.extract(html_content, include_comments=False, include_tables=False) or ""
        if not text:
            soup = BeautifulSoup(html_content, "html.parser")
            tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "li", "td"])
            text = " ".join(t.get_text(" ", strip=True) for t in tags)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


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
    best_score, best_url = 0.0, ""
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


def check_article(url, n_samples, match_threshold, excluded_domains, words_per_passage,
                  preloaded_text=""):
    """Check one article for web plagiarism. preloaded_text skips URL fetch."""
    text = preloaded_text if preloaded_text else fetch_text(url)
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


def compute_similarity_matrix(texts):
    """Full N×N cosine similarity matrix."""
    if len(texts) < 2:
        return np.zeros((len(texts), len(texts)))
    vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english",
                          min_df=1, max_features=50000)
    m = vec.fit_transform(texts)
    return cosine_similarity(m)


def build_duplicate_clusters(urls, texts, threshold=0.6):
    """Returns (sim_matrix, {cluster_id: [idx, ...]})."""
    n = len(urls)
    valid_pairs = [(i, u, t) for i, (u, t) in enumerate(zip(urls, texts))
                   if len(t.split()) >= 80]
    sim = np.zeros((n, n))
    if len(valid_pairs) >= 2:
        vidx = [i for i, _, _ in valid_pairs]
        vsim = compute_similarity_matrix([t for _, _, t in valid_pairs])
        for ri, i in enumerate(vidx):
            for rj, j in enumerate(vidx):
                sim[i, j] = vsim[ri, rj]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[min(px, py)] = max(px, py)

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return sim, {k: v for k, v in clusters.items() if len(v) > 1}


def get_duplicate_pairs(urls, sim_matrix, threshold=0.6):
    """Return all article pairs sorted by similarity descending."""
    pairs = []
    n = len(urls)
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j] * 100
            if score >= threshold * 100:
                pairs.append({"url_a": urls[i], "url_b": urls[j],
                               "similarity_%": round(score, 1)})
    pairs.sort(key=lambda x: x["similarity_%"], reverse=True)
    return pairs
