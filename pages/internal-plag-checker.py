"""
Internal Plagiarism Checker — /internal-plag-checker

Reference corpus is pre-loaded from a fixed local CSV.
Users only need to upload the new articles CSV to check.
"""

import html as html_module
import io
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from plag_utils import extract_text_from_html, fetch_text, check_article

# ── paths ─────────────────────────────────────────────────────────────────────
CORPUS_PATH = "/Users/harsh/Documents/final_data_plag.csv"
CACHE_DIR   = os.path.join(os.path.dirname(CORPUS_PATH), ".plag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Internal Plag Checker · KollegeApply",
    page_icon="🔁",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top:1.5rem !important; max-width:1350px; padding-bottom:3rem;}

section[data-testid="stSidebar"] {background:#0f172a;}
section[data-testid="stSidebar"] * {color:#e2e8f0 !important;}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stSelectbox > div {
    background:#1e293b !important; border-color:#334155 !important; color:#f1f5f9 !important;
}
section[data-testid="stSidebar"] hr {border-color:#334155 !important;}
section[data-testid="stSidebar"] label {color:#94a3b8 !important; font-size:.8rem !important;}

.hero {
    background:linear-gradient(135deg,#0c1445 0%,#1a237e 45%,#283593 100%);
    border-radius:20px; padding:1.75rem 2.25rem; margin-bottom:1.5rem;
    border:1px solid rgba(63,81,181,.4);
    box-shadow:0 20px 60px rgba(26,35,126,.3);
}
.hero h1 {color:#fff; font-size:1.9rem; margin:0 0 .4rem; font-weight:800; letter-spacing:-.5px;}
.hero p  {color:rgba(255,255,255,.8); margin:0; font-size:.97rem; line-height:1.5;}
.chips   {margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap;}
.chip    {background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2);
          color:#c5cae9 !important; padding:4px 13px; border-radius:99px;
          font-size:.78rem; font-weight:500;}

.step-label {display:flex; align-items:center; gap:.6rem; font-size:1rem; font-weight:700;
             color:#1e293b; margin:1.5rem 0 .6rem;}
.step-badge {width:1.75rem; height:1.75rem; border-radius:50%;
             background:linear-gradient(135deg,#3f51b5,#7c4dff); color:#fff;
             display:inline-flex; align-items:center; justify-content:center;
             font-size:.85rem; font-weight:700; flex-shrink:0;
             box-shadow:0 2px 8px rgba(63,81,181,.4);}

.corpus-ok {
    display:inline-flex; align-items:center; gap:.6rem;
    background:#f0fdf4; border:1px solid #86efac; border-radius:12px;
    padding:.65rem 1.1rem; font-size:.9rem; color:#14532d; font-weight:600;
    margin:.4rem 0 1rem;
}
.corpus-err {
    display:inline-flex; align-items:center; gap:.6rem;
    background:#fef2f2; border:1px solid #fca5a5; border-radius:12px;
    padding:.65rem 1.1rem; font-size:.9rem; color:#991b1b; font-weight:600;
    margin:.4rem 0 1rem;
}

.mt {background:#fff; border-radius:14px; padding:1rem 1.25rem;
     border:1px solid #e2e8f0; box-shadow:0 1px 6px rgba(0,0,0,.05); text-align:center;}
.mt .val {font-size:1.9rem; font-weight:800; color:#1e293b; line-height:1.1;}
.mt .lab {font-size:.73rem; color:#64748b; margin-top:.3rem; font-weight:500;
          text-transform:uppercase; letter-spacing:.04em;}
.mt.red   .val {color:#dc2626;}
.mt.amber .val {color:#d97706;}
.mt.green .val {color:#16a34a;}
.mt.blue  .val {color:#3f51b5;}

.callout      {background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px;
               padding:.85rem 1.1rem; font-size:.88rem; color:#1e3a5f;
               line-height:1.55; margin:.5rem 0;}
.callout b    {color:#1e40af;}
.callout.warn {background:#fffbeb; border-color:#fde68a; color:#78350f;}

.res-card        {background:#fff; border-radius:14px; padding:.9rem 1.2rem;
                  border:1px solid #e2e8f0; box-shadow:0 2px 8px rgba(0,0,0,.06);
                  margin:.4rem 0; display:flex; gap:1rem; align-items:flex-start;}
.res-card.danger {border-left:4px solid #dc2626; background:#fef2f2;}
.res-card.warn   {border-left:4px solid #d97706; background:#fffbeb;}
.res-card.ok     {border-left:4px solid #16a34a; background:#f0fdf4;}
.rcscore         {font-size:1.5rem; font-weight:800; min-width:3.5rem; text-align:center;}
.res-card.danger .rcscore {color:#dc2626;}
.res-card.warn   .rcscore {color:#d97706;}
.res-card.ok     .rcscore {color:#16a34a;}
.rcbody          {flex:1; font-size:.88rem; color:#374151; line-height:1.6;}
.rcbody a        {color:#3f51b5; text-decoration:none;}
.rcbody a:hover  {text-decoration:underline;}
.rcmeta          {font-size:.78rem; color:#94a3b8; margin-top:.2rem;}

.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#3f51b5,#7c4dff) !important;
    border:none !important; font-weight:600 !important;
    box-shadow:0 4px 14px rgba(63,81,181,.35) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def step(n, label):
    st.markdown(
        f'<div class="step-label"><span class="step-badge">{n}</span>{label}</div>',
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
        return "danger", "🔴", "Duplicate / high overlap"
    if score >= 40:
        return "warn",   "🟡", "Moderate overlap — review"
    return "ok",     "🟢", "Looks unique"


def normalise_new(df):
    """Map columns of the user-uploaded CSV to url / description."""
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
# Load corpus (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_corpus(path):
    df = pd.read_csv(path, usecols=lambda c: c in ["url","description"])
    df["url"] = df["url"].astype(str).str.strip()
    if "description" not in df.columns:
        df["description"] = ""
    df["description"] = df["description"].astype(str)
    return df


def _cache_key():
    """Return a string based on the CSV's modification time."""
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
    """
    Load or build corpus index.
    Priority: session_state (instant) → disk cache (fast) → build from scratch (slow, once only).
    """
    if "corpus_urls" in st.session_state:
        return  # already in memory this session

    key = _cache_key()

    if _disk_cache_exists(key):
        # ── fast path: load from disk ─────────────────────────────────────────
        with st.spinner("⚡ Loading corpus index from disk cache…"):
            vec, mat, urls = _load_from_disk(key)
        st.session_state["corpus_urls"] = urls
        st.session_state["corpus_vec"]  = vec
        st.session_state["corpus_mat"]  = mat
        return

    # ── slow path: build from scratch, then save to disk ─────────────────────
    df    = load_corpus(CORPUS_PATH)
    urls  = df["url"].tolist()
    descs = df["description"].tolist()
    n     = len(urls)

    # Step A — parse HTML
    st.markdown(
        '<div class="callout"><b>Building index for the first time</b> — '
        'this takes a few minutes once, then loads in seconds every time after.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Step 1 / 2 — Parsing article HTML…**")
    prog_a    = st.progress(0.0)
    counter_a = st.empty()
    texts = []
    for i, html in enumerate(descs, 1):
        texts.append(extract_text_from_html(html))
        if i % 500 == 0 or i == n:
            prog_a.progress(i / n)
            counter_a.markdown(
                f'<span style="font-size:.85rem;color:#475569;">'
                f'📄 <b>{i:,}</b> / <b>{n:,}</b> articles parsed</span>',
                unsafe_allow_html=True,
            )
    prog_a.empty()
    counter_a.empty()

    # Step B — fit TF-IDF and save
    st.markdown("**Step 2 / 2 — Fitting TF-IDF index & saving to disk…**")
    with st.spinner(f"Vectorising {n:,} articles…"):
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_features=50000,
        )
        mat = vec.fit_transform(texts)
        _save_to_disk(key, vec, mat, urls)

    st.session_state["corpus_urls"] = urls
    st.session_state["corpus_vec"]  = vec
    st.session_state["corpus_mat"]  = mat
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()
    dup_threshold = st.slider("Duplicate threshold (%)", 20, 100, 65)

    st.divider()
    run_web = st.checkbox("Also check against the open web", value=False)
    if run_web:
        n_passages   = st.slider("Passages per article", 4, 20, 8)
        web_thresh   = st.slider("Match strictness (%)", 70, 100, 85)
        own_domain   = st.text_input("Your domain (skipped)", "kollegeapply.com")
        excl_domains = st.text_area("Other ignored domains",
                                    "wikipedia.org\nyoutube.com", height=60)

    st.divider()
    top_n = st.number_input("Max results to show", 10, 5000, 200)
    st.caption(
        "**Similarity guide:**  \n"
        "🟢 **<40 %** unique  \n"
        "🟡 **40–65 %** review  \n"
        "🔴 **>65 %** duplicate"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔁 Internal Plagiarism Checker</h1>
  <p>Upload a CSV of <b>new articles</b> to instantly check them against
     the KollegeApply article database · get per-article duplicate verdicts</p>
  <div class="chips">
    <span class="chip">📚 75 k article corpus — pre-loaded</span>
    <span class="chip">📄 Upload new CSV to check</span>
    <span class="chip">⚡ HTML parsed — no URL fetching needed</span>
    <span class="chip">🔴 Per-article verdict + export</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-load corpus
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(CORPUS_PATH):
    st.markdown(
        f'<div class="corpus-err">❌ Corpus file not found at '
        f'<code>{CORPUS_PATH}</code> — contact admin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

_ensure_corpus_index()   # builds with progress bars on first run, no-op after

corpus_urls = st.session_state["corpus_urls"]
corpus_vec  = st.session_state["corpus_vec"]
corpus_mat  = st.session_state["corpus_mat"]
n_corp      = len(corpus_urls)

st.markdown(
    f'<div class="corpus-ok">📚 Reference corpus ready · '
    f'<b>{n_corp:,} articles</b> · TF-IDF index pre-built</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Upload new articles
# ─────────────────────────────────────────────────────────────────────────────
step(1, "Upload new articles to check")

st.markdown(
    '<div class="callout">Upload a CSV of the <b>new articles</b> you want to verify. '
    'Required column: <code>url</code>. '
    'Optional: <code>description</code> (HTML — if missing, fetched from URL).'
    '</div>',
    unsafe_allow_html=True,
)

new_df = None
with st.container(border=True):
    new_src = st.radio("Input method", ["📄 Upload CSV", "📋 Paste URLs"],
                       horizontal=True, key="new_src")

    if new_src == "📄 Upload CSV":
        new_file = st.file_uploader("Upload new articles CSV", type=["csv"], key="new_upload")
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
                    url_c  = mc1.selectbox("URL column",         list(raw_new.columns), key="nu")
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
                    st.success(
                        f"✅ {len(new_df):,} articles ready · "
                        f"{has_desc} have HTML · "
                        f"{len(new_df)-has_desc} will be fetched from URL"
                    )
    else:
        pasted = st.text_area("One URL per line", height=140,
                              placeholder="https://…\nhttps://…")
        if pasted.strip():
            urls = [u.strip() for u in pasted.splitlines()
                    if u.strip().startswith("http")]
            if urls:
                new_df = pd.DataFrame({"url": urls, "description": ""})
                st.success(f"✅ {len(urls)} URLs (content will be fetched from each URL)")
            else:
                st.error("No valid URLs found.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Run
# ─────────────────────────────────────────────────────────────────────────────
if new_df is not None and len(new_df) > 0:
    step(2, "Run check")

    with st.container(border=True):
        n_new   = len(new_df)
        n_fetch = int((new_df["description"].str.strip().str.len() <= 10).sum())
        est_sec = n_new * 0.5 + n_fetch * 4
        if run_web:
            est_sec += n_new * n_passages * 5

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("New articles",  n_new)
        c2.metric("Corpus size",   f"{n_corp:,}")
        c3.metric("URLs to fetch", n_fetch)
        c4.metric("Est. time",
                  f"~{max(1,round(est_sec/60))} min" if est_sec > 60
                  else f"~{int(est_sec)}s")

        if n_fetch > 0:
            st.markdown(
                '<div class="callout warn">⚠️ Articles without a <code>description</code> '
                'column will be fetched from the URL — this is slower.</div>',
                unsafe_allow_html=True,
            )

        run_btn = st.button("🚀 Run check", type="primary",
                            use_container_width=True, key="run_btn")

    if run_btn:
        # ── extract text for new articles ─────────────────────────────────────
        with st.status("📄 Preparing new articles…", expanded=True) as s2:
            prog = st.progress(0.0)
            cur  = st.empty()
            new_texts, new_wc = [], []
            for i, row in enumerate(new_df.itertuples(), 1):
                cur.markdown(
                    f'<div style="font-size:.88rem;color:#1e293b;">'
                    f'📄 Article <b>{i}</b> of <b>{n_new}</b> &nbsp;·&nbsp; '
                    f'<span style="color:#64748b;">{row.url}</span></div>',
                    unsafe_allow_html=True,
                )
                desc = str(row.description).strip()
                text = extract_text_from_html(desc) if len(desc) > 10 else fetch_text(row.url)
                new_texts.append(text)
                new_wc.append(len(text.split()))
                prog.progress(i / n_new)
            cur.empty()
            s2.update(label=f"✅ {n_new} articles ready",
                      state="complete", expanded=False)

        # ── similarity: transform new articles with pre-built corpus index ────
        with st.status("🔁 Computing similarity…", expanded=False) as s3:
            valid_new = [(i, t) for i, t in enumerate(new_texts) if len(t.split()) >= 30]
            sim_results = [(0.0, 0)] * n_new

            if valid_new and corpus_mat.shape[0] > 0:
                valid_texts_only = [t for _, t in valid_new]
                # transform only — vectorizer is already fitted on the corpus
                new_mat = corpus_vec.transform(valid_texts_only)
                sims    = cosine_similarity(new_mat, corpus_mat)
                for ni, (orig_i, _) in enumerate(valid_new):
                    best_j  = int(sims[ni].argmax())
                    best_sc = round(float(sims[ni, best_j]) * 100, 1)
                    sim_results[orig_i] = (best_sc, best_j)

            s3.update(label="✅ Similarity computed", state="complete")

        # ── optional web check ────────────────────────────────────────────────
        web_res = {}
        if run_web:
            excl = {d.strip().lower() for d in excl_domains.splitlines() if d.strip()}
            excl.add(own_domain.strip().lower().replace("www.", ""))
            with st.status(f"🌐 Web-checking {n_new} articles…", expanded=True) as sw:
                wp = st.progress(0.0)
                wt = st.empty()
                for i, (row, txt) in enumerate(zip(new_df.itertuples(), new_texts), 1):
                    wt.markdown(f"**{i}/{n_new}** · `{row.url}`")
                    try:
                        res, _ = check_article(
                            row.url, n_passages, web_thresh, excl, 25,
                            preloaded_text=txt,
                        )
                    except Exception as e:
                        res = {"plagiarism_score": None, "verdict": f"Error: {e}",
                               "top_sources": "", "matches": []}
                    web_res[row.url] = res
                    wp.progress(i / n_new)
                wt.empty()
                sw.update(label="✅ Web check done", state="complete", expanded=False)

        # ── assemble results ──────────────────────────────────────────────────
        final_rows = []
        for row, txt, wc, (sc, ci) in zip(
            new_df.itertuples(), new_texts, new_wc, sim_results
        ):
            corp_url = corpus_urls[ci] if ci < len(corpus_urls) else ""
            wr       = web_res.get(row.url, {})
            _, icon, vlabel = verdict_for(sc, dup_threshold)
            final_rows.append({
                "url":                    row.url,
                "word_count":             wc,
                "similarity_to_corpus_%": sc,
                "verdict":                f"{icon} {vlabel}",
                "matched_existing_url":   corp_url,
                "web_plag_score":         wr.get("plagiarism_score"),
                "web_verdict":            wr.get("verdict", ""),
                "top_web_sources":        wr.get("top_sources", ""),
                "_matches":               wr.get("matches", []),
            })

        st.session_state["results"] = final_rows
        st.rerun()

else:
    st.info("👆 Upload a CSV of new articles to begin.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Results
# ─────────────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    rdf = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])

    h1, h2 = st.columns([6, 1], vertical_alignment="bottom")
    with h1:
        step(3, "Results")
    with h2:
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.pop("results", None)
            st.rerun()

    # summary tiles
    n_dup = int((rdf["similarity_to_corpus_%"] >= dup_threshold).sum())
    n_rev = int(((rdf["similarity_to_corpus_%"] >= 40) &
                 (rdf["similarity_to_corpus_%"] < dup_threshold)).sum())
    n_ok  = int((rdf["similarity_to_corpus_%"] < 40).sum())
    avg_s = f"{rdf['similarity_to_corpus_%'].mean():.1f}%"

    cols = st.columns(5)
    for col, (val, lab, cls) in zip(cols, [
        (len(rdf),  "Checked",                         "blue"),
        (avg_s,     "Avg similarity",                   "blue"),
        (n_dup,     f"🔴 Duplicates (≥{dup_threshold}%)", "red"),
        (n_rev,     "🟡 Review needed",                 "amber"),
        (n_ok,      "🟢 Unique",                        "green"),
    ]):
        col.markdown(
            f'<div class="mt {cls}"><div class="val">{val}</div>'
            f'<div class="lab">{lab}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    if n_dup:
        st.error(
            f"⚠️ **{n_dup} article{'s' if n_dup>1 else ''}** "
            f"{'are' if n_dup>1 else 'is'} ≥{dup_threshold}% similar to an existing article.",
            icon="🔴",
        )

    view = st.segmented_control(
        "View",
        ["📋 All results", "🔴 Duplicates only", "🌐 Web plagiarism"],
        default="📋 All results",
        label_visibility="collapsed",
    ) or "📋 All results"

    if view in ("📋 All results", "🔴 Duplicates only"):
        show = results if view == "📋 All results" else [
            r for r in results if r["similarity_to_corpus_%"] >= dup_threshold
        ]
        show = sorted(show, key=lambda r: r["similarity_to_corpus_%"], reverse=True)
        show = show[:int(top_n)]

        if not show:
            st.success("✅ No duplicates at the current threshold.", icon="✅")
        else:
            tdf = pd.DataFrame([{
                "New Article":      r["url"],
                "Words":            r["word_count"],
                "Similarity %":     r["similarity_to_corpus_%"],
                "Verdict":          r["verdict"],
                "Closest existing": r["matched_existing_url"],
            } for r in show])

            st.dataframe(
                tdf, hide_index=True, use_container_width=True,
                column_config={
                    "New Article":      st.column_config.LinkColumn(width="large"),
                    "Words":            st.column_config.NumberColumn(width="small"),
                    "Similarity %":     st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.0f%%", width="small"),
                    "Verdict":          st.column_config.TextColumn(width="medium"),
                    "Closest existing": st.column_config.LinkColumn(width="large"),
                },
            )

            flagged = [r for r in show if r["similarity_to_corpus_%"] >= 40]
            if flagged:
                st.markdown("#### Detailed cards")
                for r in flagged:
                    sc = r["similarity_to_corpus_%"]
                    cls, icon, vlabel = verdict_for(sc, dup_threshold)
                    new_e  = html_module.escape(r["url"])
                    corp_u = html_module.escape(r["matched_existing_url"], quote=True)
                    corp_ue = html_module.escape(r["matched_existing_url"])
                    st.markdown(
                        f'<div class="res-card {cls}">'
                        f'<div class="rcscore">{sc:.0f}%</div>'
                        f'<div class="rcbody">'
                        f'<b>New:</b> <a href="{new_e}" target="_blank">{new_e}</a><br>'
                        f'<b>Matches:</b> <a href="{corp_u}" target="_blank">{corp_ue}</a>'
                        f'<div class="rcmeta">{icon} {vlabel} · {r["word_count"]:,} words</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

    else:  # web plagiarism
        if not run_web:
            st.markdown(
                '<div class="callout">Enable <b>web plagiarism check</b> in the sidebar '
                'and re-run to see web results here.</div>',
                unsafe_allow_html=True,
            )
        else:
            web_show = [r for r in results if r.get("web_plag_score") is not None]
            if not web_show:
                st.info("No web results available.")
            else:
                wdf = pd.DataFrame([{
                    "Article":         r["url"],
                    "Web plag %":      r["web_plag_score"],
                    "Web verdict":     r["web_verdict"],
                    "Top web sources": r["top_web_sources"],
                } for r in web_show]).sort_values("Web plag %", ascending=False)
                st.dataframe(
                    wdf, hide_index=True, use_container_width=True,
                    column_config={
                        "Article":     st.column_config.LinkColumn(width="large"),
                        "Web plag %":  st.column_config.ProgressColumn(
                            min_value=0, max_value=100, format="%.0f%%", width="small"),
                        "Web verdict": st.column_config.TextColumn(width="medium"),
                        "Top web sources": st.column_config.TextColumn(width="large"),
                    },
                )
                for r in web_show:
                    if r.get("_matches"):
                        with st.expander(r["url"]):
                            for m in r["_matches"]:
                                src = html_module.escape(m["source"], quote=True)
                                st.markdown(
                                    f'<div style="border-left:4px solid #ef4444;padding:.6rem 1rem;'
                                    f'background:#fef2f2;border-radius:0 10px 10px 0;margin:.4rem 0;">'
                                    f'<i>"{html_module.escape(m["passage"])}"</i><br>'
                                    f'<small style="color:#64748b;">{m["score"]:.0f}% match · '
                                    f'<a href="{src}" target="_blank">{src}</a></small>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

    # download
    st.divider()
    export = rdf.drop(columns=["_matches", "_text"], errors="ignore").copy()
    export["risk_level"] = export["similarity_to_corpus_%"].apply(
        lambda s: "Duplicate" if s >= dup_threshold else "Review" if s >= 40 else "Unique"
    )
    fname = f"plag_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    st.download_button(
        "⬇️ Download enriched results CSV",
        export.to_csv(index=False).encode("utf-8"),
        fname, "text/csv", type="primary",
    )
    st.caption(
        "Columns: url · word_count · similarity_to_corpus_% · verdict · "
        "matched_existing_url · risk_level"
        + (" · web_plag_score · web_verdict · top_web_sources" if run_web else "")
    )
