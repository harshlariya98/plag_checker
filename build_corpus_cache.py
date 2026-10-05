"""
Build KollegeApply corpus cache from:
  1. An existing CSV  (fast — use this if you already have the master sheet)
  2. KA sitemap crawl (crawls all 3 KA sitemaps, fetches article text)

Usage:
  # From existing CSV:
  python3 build_corpus_cache.py --csv /path/to/final_data_plag.csv

  # From sitemap (crawls KA sitemaps, fetches article text):
  python3 build_corpus_cache.py --sitemap

  # From sitemap, also save fetched CSV:
  python3 build_corpus_cache.py --sitemap --save-csv corpus/final_data_plag.csv

Outputs:
  corpus/final_data_plag.csv    (url, description columns)
  .plag_cache/                  (TF-IDF index — upload to EC2)

Upload to EC2:
  scp -i ~/Downloads/kapp_plag_checker.pem \\
    corpus/final_data_plag.csv \\
    ubuntu@3.6.65.226:/home/ubuntu/plag_checker/corpus/final_data_plag.csv
  scp -i ~/Downloads/kapp_plag_checker.pem \\
    -r .plag_cache \\
    ubuntu@3.6.65.226:/home/ubuntu/plag_checker/

Then on EC2:
  sudo systemctl restart plag_checker
"""

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import joblib
import pandas as pd
import requests
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

_HERE       = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_HERE, "corpus", "final_data_plag.csv")
CACHE_DIR   = os.path.join(_HERE, ".plag_cache")

KA_SITEMAPS = [
    "https://www.kollegeapply.com/index-updates.xml",
    "https://www.kollegeapply.com/latest-updates.xml",
    "https://www.kollegeapply.com/weekly-updates.xml",
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE  = re.compile(r"\s+")
_SM_NS  = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(os.path.join(_HERE, "corpus"), exist_ok=True)


def _strip(s):
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", str(s))).strip().lower()


def _cache_key(path):
    return str(os.path.getsize(path))


# ── Sitemap crawler ────────────────────────────────────────────────────────────

def _parse_sitemap(url):
    """Fetch one sitemap XML and return all <loc> URLs (handles sitemap index)."""
    collected = []
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        tag  = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if tag == "sitemapindex":
            for sm in root.findall(".//sm:loc", _SM_NS):
                collected.extend(_parse_sitemap(sm.text.strip()))
        else:
            for loc in root.findall(".//sm:loc", _SM_NS):
                collected.append(loc.text.strip())
    except Exception as exc:
        print(f"  Warning: could not fetch {url}: {exc}")
    return collected


def _fetch_sitemap_urls(sitemap_urls=KA_SITEMAPS):
    """Return deduplicated article URLs from all KA sitemaps."""
    all_urls = []
    for sm_url in sitemap_urls:
        print(f"  Fetching {sm_url} …")
        locs = _parse_sitemap(sm_url)
        print(f"    → {len(locs):,} URLs")
        all_urls.extend(locs)
    unique = list(dict.fromkeys(all_urls))
    print(f"  Total unique URLs: {len(unique):,}")
    return unique


def _fetch_article_text(url):
    """Fetch article text via trafilatura (falls back to raw HTML strip)."""
    try:
        import trafilatura
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        text = trafilatura.extract(r.text) or ""
        return text[:5000]
    except Exception:
        pass
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        return _strip(r.text)[:5000]
    except Exception:
        return ""


def build_from_sitemap(save_csv=None):
    print("Fetching KA sitemap URLs …")
    urls = _fetch_sitemap_urls()
    if not urls:
        print("ERROR: no URLs found in sitemaps.")
        sys.exit(1)

    print(f"\nFetching article text for {len(urls):,} URLs …")
    rows = []
    t0   = time.time()
    for i, url in enumerate(urls, 1):
        text = _fetch_article_text(url)
        rows.append({"url": url, "description": text})
        if i % 100 == 0 or i == len(urls):
            elapsed = time.time() - t0
            eta     = elapsed / i * (len(urls) - i)
            print(f"  {i:,}/{len(urls):,}  ({elapsed:.0f}s elapsed, ~{int(eta)}s left)", end="\r")
    print()

    df  = pd.DataFrame(rows)
    out = save_csv or CORPUS_PATH
    df.to_csv(out, index=False)
    print(f"  Saved {len(df):,} rows → {out}")
    return out


# ── TF-IDF index builder ───────────────────────────────────────────────────────

def build_index(csv_path):
    print(f"\n[1/3] Reading {csv_path} …")
    CHUNK = 20_000
    url_chunks, text_chunks = [], []
    rows_done = 0
    t0 = time.time()

    for chunk in pd.read_csv(csv_path,
                              usecols=lambda c: c in ["url", "description"],
                              chunksize=CHUNK):
        chunk["url"] = chunk["url"].astype(str).str.strip()
        if "description" not in chunk.columns:
            chunk["description"] = ""
        chunk["description"] = chunk["description"].astype(str)
        url_chunks.append(chunk["url"])
        text_chunks.append(chunk["description"].apply(_strip))
        rows_done += len(chunk)
        print(f"  {rows_done:,} rows  ({time.time()-t0:.0f}s)", end="\r")

    urls  = pd.concat(url_chunks).tolist()
    texts = pd.concat(text_chunks).tolist()
    print(f"\n  Done — {len(urls):,} articles in {time.time()-t0:.1f}s")

    print("\n[2/3] Building TF-IDF index …")
    t1  = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                          min_df=2, max_features=50_000)
    mat = vec.fit_transform(texts)
    print(f"  Done — matrix {mat.shape} in {time.time()-t1:.1f}s")

    print("\n[3/3] Saving cache …")
    t2  = time.time()
    key = _cache_key(csv_path)
    joblib.dump(vec,   os.path.join(CACHE_DIR, f"{key}.vec.pkl"))
    save_npz(          os.path.join(CACHE_DIR, f"{key}.mat.npz"), mat)
    joblib.dump(urls,  os.path.join(CACHE_DIR, f"{key}.urls.pkl"))
    joblib.dump(texts, os.path.join(CACHE_DIR, f"{key}.texts.pkl"))
    print(f"  Done in {time.time()-t2:.1f}s")

    print("\n✅ Cache built. Files in .plag_cache/:")
    for f in os.listdir(CACHE_DIR):
        size = os.path.getsize(os.path.join(CACHE_DIR, f)) / 1e6
        print(f"   {f}  ({size:.1f} MB)")

    print(f"""
Upload to EC2:
  scp -i ~/Downloads/kapp_plag_checker.pem \\
    {csv_path} \\
    ubuntu@3.6.65.226:/home/ubuntu/plag_checker/corpus/final_data_plag.csv
  scp -i ~/Downloads/kapp_plag_checker.pem \\
    -r {CACHE_DIR} \\
    ubuntu@3.6.65.226:/home/ubuntu/plag_checker/

Then on EC2:
  sudo systemctl restart plag_checker
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KollegeApply plag corpus cache")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",     metavar="PATH", help="Path to existing KA articles CSV (url, description columns)")
    group.add_argument("--sitemap", action="store_true", help="Crawl KA sitemaps to build corpus")
    parser.add_argument("--save-csv", metavar="PATH", default=None,
                        help="(with --sitemap) also save fetched CSV here")
    args = parser.parse_args()

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"ERROR: CSV not found: {args.csv}")
            sys.exit(1)
        build_index(args.csv)
    else:
        save_path = args.save_csv or CORPUS_PATH
        fetched   = build_from_sitemap(save_csv=save_path)
        build_index(fetched)
