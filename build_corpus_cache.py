"""
Run this ONCE on your Mac to pre-build the corpus index.
Usage:  python3 build_corpus_cache.py
Output: .plag_cache/ folder with 4 files — upload these to EC2.
"""

import os, time, joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import re

CORPUS_PATH = "/Users/harsh/Documents/final_data_plag.csv"
CACHE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".plag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_TAG_RE = re.compile(r"<[^>]+>")
_WSP_RE = re.compile(r"\s+")

def _strip(s):
    return _WSP_RE.sub(" ", _TAG_RE.sub(" ", s)).strip().lower()

def _cache_key():
    return str(os.path.getsize(CORPUS_PATH))

key = _cache_key()
print(f"Cache key (mtime): {key}")
print(f"Output dir: {CACHE_DIR}")

# ── Step 1: read CSV in chunks ────────────────────────────────────────────────
print("\n[1/3] Reading corpus CSV…")
CHUNK = 20_000
total_bytes = os.path.getsize(CORPUS_PATH)
url_chunks, text_chunks = [], []
rows_done = 0
t0 = time.time()

for chunk in pd.read_csv(CORPUS_PATH,
                          usecols=lambda c: c in ["url", "description"],
                          chunksize=CHUNK):
    chunk["url"] = chunk["url"].astype(str).str.strip()
    if "description" not in chunk.columns:
        chunk["description"] = ""
    chunk["description"] = chunk["description"].astype(str)
    url_chunks.append(chunk["url"])
    text_chunks.append(chunk["description"].apply(_strip))
    rows_done += len(chunk)
    elapsed = time.time() - t0
    print(f"  {rows_done:,} rows  ({elapsed:.0f}s)", end="\r")

urls  = pd.concat(url_chunks).tolist()
texts = pd.concat(text_chunks).tolist()
print(f"\n  Done — {len(urls):,} articles in {time.time()-t0:.1f}s")

# ── Step 2: TF-IDF ────────────────────────────────────────────────────────────
print("\n[2/3] Building TF-IDF index…")
t1 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                      min_df=2, max_features=50_000)
mat = vec.fit_transform(texts)
print(f"  Done — matrix {mat.shape} in {time.time()-t1:.1f}s")

# ── Step 3: save ──────────────────────────────────────────────────────────────
print("\n[3/3] Saving to disk…")
t2 = time.time()
joblib.dump(vec,   os.path.join(CACHE_DIR, f"{key}.vec.pkl"))
save_npz(          os.path.join(CACHE_DIR, f"{key}.mat.npz"), mat)
joblib.dump(urls,  os.path.join(CACHE_DIR, f"{key}.urls.pkl"))
joblib.dump(texts, os.path.join(CACHE_DIR, f"{key}.texts.pkl"))
print(f"  Done in {time.time()-t2:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n✅ Cache built. Files in .plag_cache/:")
for f in os.listdir(CACHE_DIR):
    size = os.path.getsize(os.path.join(CACHE_DIR, f)) / 1e6
    print(f"   {f}  ({size:.1f} MB)")

print(f"""
Next step — upload cache to EC2 (run this in a Mac terminal):
  scp -i /Users/harsh/Downloads/kapp_plag_checker.pem \\
    -r {CACHE_DIR} \\
    ubuntu@3.6.65.226:/home/ubuntu/plag_checker/

Then on EC2:
  sudo systemctl restart plag_checker
""")
