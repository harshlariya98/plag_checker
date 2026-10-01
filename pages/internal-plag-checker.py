"""
Internal Plagiarism Checker — /internal-plag-checker

Workflow:
  1. Load reference corpus  – your existing articles (CSV, SQL dump, or RDS)
  2. Upload new articles     – CSV with url / entity / description columns
  3. Run analysis            – compare each new article against the corpus
  4. View results            – only for the new articles you uploaded
"""

import html as html_module
import io
import re
import sys
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from plag_utils import (
    extract_text_from_html,
    fetch_text,
    check_article,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config & CSS
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

.corpus-badge {
    display:inline-flex; align-items:center; gap:.5rem;
    background:#f0fdf4; border:1px solid #86efac; border-radius:10px;
    padding:.55rem 1rem; font-size:.88rem; color:#14532d; font-weight:600;
    margin:.6rem 0;
}
.corpus-badge.warn {background:#fffbeb; border-color:#fde68a; color:#78350f;}

.mt {background:#fff; border-radius:14px; padding:1rem 1.25rem;
     border:1px solid #e2e8f0; box-shadow:0 1px 6px rgba(0,0,0,.05); text-align:center;}
.mt .val {font-size:1.9rem; font-weight:800; color:#1e293b; line-height:1.1;}
.mt .lab {font-size:.73rem; color:#64748b; margin-top:.3rem; font-weight:500;
          text-transform:uppercase; letter-spacing:.04em;}
.mt.red   .val {color:#dc2626;}
.mt.amber .val {color:#d97706;}
.mt.green .val {color:#16a34a;}
.mt.blue  .val {color:#3f51b5;}

.callout {background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px;
          padding:.85rem 1.1rem; font-size:.88rem; color:#1e3a5f;
          line-height:1.55; margin:.5rem 0;}
.callout b {color:#1e40af;}
.callout.warn {background:#fffbeb; border-color:#fde68a; color:#78350f;}

.res-card {
    background:#fff; border-radius:14px; padding:.9rem 1.2rem;
    border:1px solid #e2e8f0; box-shadow:0 2px 8px rgba(0,0,0,.06); margin:.4rem 0;
}
.res-card.danger {border-left:4px solid #dc2626; background:#fef2f2;}
.res-card.warn   {border-left:4px solid #d97706; background:#fffbeb;}
.res-card.ok     {border-left:4px solid #16a34a; background:#f0fdf4;}
.res-card .rcscore {font-size:1.5rem; font-weight:800; min-width:3.5rem; text-align:center;}
.res-card.danger .rcscore {color:#dc2626;}
.res-card.warn   .rcscore {color:#d97706;}
.res-card.ok     .rcscore {color:#16a34a;}
.res-card .rcbody {flex:1; font-size:.88rem; color:#374151; line-height:1.6;}
.res-card .rcbody a {color:#3f51b5; text-decoration:none;}
.res-card .rcbody a:hover {text-decoration:underline;}
.res-card .rcmeta {font-size:.78rem; color:#94a3b8; margin-top:.2rem;}

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
        return "warn", "🟡", "Moderate overlap — review"
    return "ok", "🟢", "Looks unique"


def parse_sql_file(content_bytes):
    """Extract rows from a MySQL/PostgreSQL SQL dump."""
    try:
        text = content_bytes.decode("utf-8", errors="replace")
    except Exception:
        return None, "Could not decode file as UTF-8."

    # PostgreSQL COPY … FROM stdin
    copy_m = re.search(
        r"COPY\s+\S+\s*\(([^)]+)\)\s+FROM\s+stdin;\s*\n(.*?)\n\\\\.",
        text, re.DOTALL | re.IGNORECASE,
    )
    if copy_m:
        cols = [c.strip().strip('"').strip('`') for c in copy_m.group(1).split(",")]
        rows = []
        for line in copy_m.group(2).strip().splitlines():
            vals = line.split("\t")
            if len(vals) == len(cols):
                rows.append(dict(zip(cols, vals)))
        if rows:
            return pd.DataFrame(rows), None

    # INSERT INTO … VALUES …
    insert_re = re.compile(
        r"INSERT\s+INTO\s+[`\"]?\w+[`\"]?\s*\(([^)]+)\)\s+VALUES\s*(.+?)(?:;|$)",
        re.IGNORECASE | re.DOTALL,
    )
    rows = []
    col_names = None
    for m in insert_re.finditer(text):
        if col_names is None:
            col_names = [c.strip().strip('`"\' ') for c in m.group(1).split(",")]
        for t in re.finditer(r"\(([^()]*)\)", m.group(2)):
            parts = re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", t.group(1))
            vals = [p.strip().strip("'\"").replace("\\'", "'") for p in parts]
            if len(vals) == len(col_names):
                rows.append(dict(zip(col_names, vals)))
    if rows:
        return pd.DataFrame(rows), None

    return None, (
        "Could not parse INSERT or COPY statements. "
        "Please export as CSV or ensure standard SQL dump format."
    )


def load_df(file, label="file"):
    name = file.name.lower()
    content = file.read()
    if name.endswith(".sql"):
        df, err = parse_sql_file(content)
        if err:
            st.error(f"SQL parse error ({label}): {err}")
            return None
        return df
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        st.error(f"Could not read {label}: {e}")
        return None


def normalise_df(df, require_desc=True):
    """Map columns to url / entity / description. Returns (df, error)."""
    url_col  = smart_col(df, ["url","link","page_url","article_url","slug"])
    ent_col  = smart_col(df, ["entity","entity_type","type","category","tag","section"])
    desc_col = smart_col(df, ["description","content","article","body","html",
                               "article_html","text","article_body"])
    missing = []
    if not url_col:
        missing.append("url")
    if require_desc and not desc_col:
        missing.append("description/html")
    if missing:
        return df, f"Missing columns: {missing}. Found: {list(df.columns)}"

    rename = {url_col: "url"}
    if ent_col:  rename[ent_col]  = "entity"
    if desc_col: rename[desc_col] = "description"
    df = df.rename(columns=rename)
    if "entity"      not in df.columns: df["entity"]      = "unknown"
    if "description" not in df.columns: df["description"] = ""
    keep = ["url","entity","description"] + [
        c for c in df.columns if c not in ("url","entity","description")
    ]
    return df[keep].copy(), None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    st.markdown("**🔁 Duplicate threshold**")
    dup_threshold = st.slider(
        "Flag as duplicate above (%)", 20, 100, 65,
        help="New articles above this % similarity to any existing article are flagged.",
    )

    st.divider()
    st.markdown("**🌐 Web plagiarism (optional)**")
    run_web = st.checkbox("Also check against the open web", value=False)
    if run_web:
        n_passages   = st.slider("Passages per article", 4, 20, 8)
        web_thresh   = st.slider("Match strictness (%)", 70, 100, 85)
        own_domain   = st.text_input("Your domain (skipped)", "kollegeapply.com")
        excl_domains = st.text_area("Other ignored domains",
                                    "wikipedia.org\nyoutube.com", height=70)

    st.divider()
    st.markdown("**📊 Display**")
    top_n = st.number_input("Max results to show", 10, 5000, 200)
    st.caption(
        "**Similarity guide:**  \n"
        "🟢 **<40 %** — unique  \n"
        "🟡 **40–65 %** — review  \n"
        "🔴 **>65 %** — duplicate"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔁 Internal Plagiarism Checker</h1>
  <p>Load your existing article database once as a <b>reference corpus</b> ·
     then upload any new CSV to check against it ·
     get per-article duplicate verdicts in seconds</p>
  <div class="chips">
    <span class="chip">📚 Reference corpus from CSV / SQL / RDS</span>
    <span class="chip">📄 Check new CSV each run</span>
    <span class="chip">⚡ HTML parsed — no URL fetching needed</span>
    <span class="chip">🔴 Per-article verdict</span>
    <span class="chip">⬇️ Enriched CSV export</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load reference corpus
# ─────────────────────────────────────────────────────────────────────────────
step(1, "Load your existing articles (reference corpus)")

st.markdown(
    '<div class="callout">Upload your <b>full existing article database</b> once. '
    'It stays in session — you do not need to re-upload between checks.<br>'
    'Required columns: <code>url</code>, <code>description</code> (HTML). '
    'Optional: <code>entity</code> / <code>entity_type</code>.</div>',
    unsafe_allow_html=True,
)

corp_src = st.radio(
    "Source",
    ["📄 CSV file", "🗄️ SQL dump (.sql)", "🛢️ RDS / database"],
    horizontal=True, key="corp_src",
)

corpus_df = st.session_state.get("corpus_df")

with st.container(border=True):
    # ── CSV / SQL ─────────────────────────────────────────────────────────────
    if corp_src in ("📄 CSV file", "🗄️ SQL dump (.sql)"):
        ext = ["csv"] if "CSV" in corp_src else ["sql"]
        corp_file = st.file_uploader(
            "Upload reference corpus", type=ext, key="corp_upload",
        )
        if corp_file:
            with st.spinner("Loading…"):
                raw = load_df(corp_file, label="reference corpus")
            if raw is not None:
                df_norm, err = normalise_df(raw, require_desc=True)
                if err:
                    st.error(err)
                    st.write("Columns found:", list(raw.columns))
                else:
                    df_norm["url"]         = df_norm["url"].astype(str).str.strip()
                    df_norm["entity"]      = df_norm["entity"].astype(str).str.strip()
                    df_norm["description"] = df_norm["description"].astype(str)
                    st.session_state["corpus_df"]    = df_norm
                    st.session_state["corpus_texts"] = None  # mark for rebuild
                    corpus_df = df_norm
                    st.success(f"✅ {len(df_norm):,} articles loaded as reference corpus")

    # ── RDS ──────────────────────────────────────────────────────────────────
    else:
        c1, c2, c3 = st.columns([2, 3, 1])
        db_type  = c1.selectbox("DB type", ["PostgreSQL", "MySQL"], key="corp_dbtype")
        host     = c2.text_input("Host", placeholder="mydb.xxx.rds.amazonaws.com", key="corp_host")
        port     = c3.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306, key="corp_port")
        c4, c5   = st.columns(2)
        dbname   = c4.text_input("Database", key="corp_dbname")
        user     = c5.text_input("Username", key="corp_user")
        password = st.text_input("Password", type="password", key="corp_pw")
        query    = st.text_area(
            "Query",
            "SELECT url, entity_type AS entity, description\nFROM articles\nWHERE is_published = true;",
            height=90, key="corp_query",
        )
        col_btn, col_clr = st.columns([2, 1])
        if col_btn.button("🔌 Connect & load", type="primary", key="corp_fetch"):
            if not all([host, dbname, user, password]):
                st.error("Fill in all connection fields.")
            else:
                conn_str = (
                    f"postgresql+psycopg2://{user}:{password}@{host}:{int(port)}/{dbname}"
                    if db_type == "PostgreSQL"
                    else f"mysql+pymysql://{user}:{password}@{host}:{int(port)}/{dbname}"
                )
                with st.spinner("Connecting…"):
                    try:
                        from sqlalchemy import create_engine, text as sqltxt
                        eng = create_engine(conn_str, connect_args={"connect_timeout": 15})
                        with eng.connect() as conn:
                            raw = pd.read_sql_query(sqltxt(query), conn)
                        df_norm, err = normalise_df(raw, require_desc=True)
                        if err:
                            st.error(err)
                        else:
                            df_norm["url"]         = df_norm["url"].astype(str).str.strip()
                            df_norm["entity"]      = df_norm["entity"].astype(str).str.strip()
                            df_norm["description"] = df_norm["description"].astype(str)
                            st.session_state["corpus_df"]    = df_norm
                            st.session_state["corpus_texts"] = None
                            corpus_df = df_norm
                            st.success(f"✅ {len(df_norm):,} articles loaded from RDS")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
        if col_clr.button("🗑️ Clear corpus", key="corp_clear"):
            for k in ("corpus_df", "corpus_texts", "results"):
                st.session_state.pop(k, None)
            st.rerun()

# corpus status
corpus_df = st.session_state.get("corpus_df")
if corpus_df is not None:
    top_ents = ", ".join(
        f"{e} ({n})"
        for e, n in corpus_df["entity"].value_counts().head(5).items()
    )
    st.markdown(
        f'<div class="corpus-badge">📚 Reference corpus ready: '
        f'<b>{len(corpus_df):,} articles</b> · {top_ents}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="corpus-badge warn">⚠️ No reference corpus loaded yet.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Upload new articles to check
# ─────────────────────────────────────────────────────────────────────────────
step(2, "Upload new articles to check")

st.markdown(
    '<div class="callout">Upload the <b>new articles</b> you want to verify against your corpus. '
    'Columns: <code>url</code> (required), <code>entity</code> (optional), '
    '<code>description</code> (HTML — optional; omit and we fetch from the URL). '
    'Results will be generated <b>only for this CSV</b>.</div>',
    unsafe_allow_html=True,
)

new_df = None
with st.container(border=True):
    new_src = st.radio("Source", ["📄 Upload CSV", "📋 Paste URLs"],
                       horizontal=True, key="new_src")

    if new_src == "📄 Upload CSV":
        new_file = st.file_uploader("Upload new articles CSV",
                                    type=["csv"], key="new_upload")
        if new_file:
            raw_new = load_df(new_file, label="new articles")
            if raw_new is not None:
                df_norm2, err2 = normalise_df(raw_new, require_desc=False)
                if err2:
                    st.warning(f"Column mapping issue: {err2}")
                    st.write("Columns found:", list(raw_new.columns))
                    mc1, mc2, mc3 = st.columns(3)
                    url_c  = mc1.selectbox("URL column",         list(raw_new.columns), key="nu")
                    ent_c  = mc2.selectbox("Entity column",      list(raw_new.columns), key="ne")
                    desc_c = mc3.selectbox("Description column",
                                           ["(none)"] + list(raw_new.columns), key="nd")
                    rename = {url_c: "url", ent_c: "entity"}
                    if desc_c != "(none)":
                        rename[desc_c] = "description"
                    df_norm2 = raw_new.rename(columns=rename)
                    if "entity"      not in df_norm2.columns: df_norm2["entity"]      = "unknown"
                    if "description" not in df_norm2.columns: df_norm2["description"] = ""
                new_df = df_norm2
                if new_df is not None:
                    new_df["url"]         = new_df["url"].astype(str).str.strip()
                    new_df["entity"]      = new_df["entity"].astype(str).str.strip()
                    new_df["description"] = new_df["description"].astype(str).fillna("")
                    has_desc = (new_df["description"].str.strip().str.len() > 10).sum()
                    st.success(
                        f"✅ {len(new_df):,} new articles · "
                        f"{has_desc} have HTML · "
                        f"{len(new_df)-has_desc} will be fetched"
                    )
    else:
        pasted = st.text_area("One URL per line", height=140,
                              placeholder="https://…\nhttps://…")
        if pasted.strip():
            urls = [u.strip() for u in pasted.splitlines()
                    if u.strip().startswith("http")]
            if urls:
                new_df = pd.DataFrame({
                    "url": urls, "entity": "unknown", "description": ""
                })
                st.success(f"✅ {len(urls)} URLs (content will be fetched)")
            else:
                st.error("No valid URLs found (must start with http).")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Run
# ─────────────────────────────────────────────────────────────────────────────
can_run = corpus_df is not None and new_df is not None and len(new_df) > 0

if can_run:
    step(3, "Run check")

    with st.container(border=True):
        n_new    = len(new_df)
        n_corpus = len(corpus_df)
        n_fetch  = int((new_df["description"].str.strip().str.len() <= 10).sum())
        est_sec  = n_new * 0.4 + n_fetch * 4
        if run_web:
            est_sec += n_new * n_passages * 5

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("New articles",    n_new)
        c2.metric("Corpus size",     f"{n_corpus:,}")
        c3.metric("URLs to fetch",   n_fetch)
        c4.metric("Est. time",
                  f"~{max(1,round(est_sec/60))} min" if est_sec > 60
                  else f"~{int(est_sec)}s")

        if n_fetch > 0:
            st.markdown(
                '<div class="callout warn">⚠️ Articles without a description will be '
                'fetched from their URL. Add a <code>description</code> column with HTML '
                'to avoid network calls.</div>',
                unsafe_allow_html=True,
            )

        run_btn = st.button("🚀 Run check", type="primary",
                            use_container_width=True, key="run_btn")

    if run_btn:
        # 3a — parse corpus HTML
        corpus_texts = st.session_state.get("corpus_texts")
        if corpus_texts is None:
            with st.status("⚡ Parsing reference corpus HTML…", expanded=False) as s:
                corpus_texts = [
                    extract_text_from_html(h) for h in corpus_df["description"]
                ]
                st.session_state["corpus_texts"] = corpus_texts
                s.update(
                    label=f"✅ Corpus parsed ({len(corpus_texts):,} articles)",
                    state="complete",
                )

        # 3b — extract text for new articles
        with st.status("📄 Preparing new articles…", expanded=True) as s2:
            prog = st.progress(0.0)
            cur  = st.empty()
            new_texts = []
            new_wc    = []
            for i, row in enumerate(new_df.itertuples(), 1):
                cur.markdown(f"**{i}/{n_new}** · `{row.url}`")
                desc = str(row.description).strip()
                text = extract_text_from_html(desc) if len(desc) > 10 else fetch_text(row.url)
                new_texts.append(text)
                new_wc.append(len(text.split()))
                prog.progress(i / n_new)
            cur.empty()
            s2.update(label=f"✅ {n_new} articles ready",
                      state="complete", expanded=False)

        # 3c — TF-IDF: new vs corpus
        with st.status("🔁 Computing similarity against corpus…", expanded=False) as s3:
            all_texts   = corpus_texts + new_texts
            valid_mask  = [len(t.split()) >= 30 for t in all_texts]
            valid_texts = [t for t, ok in zip(all_texts, valid_mask) if ok]
            n_corp_ok   = sum(valid_mask[:n_corpus])

            sim_results = []  # list of (best_score_pct, corpus_row_idx)

            if len(valid_texts) >= 2 and n_corp_ok > 0:
                vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english",
                                      min_df=1, max_features=60000)
                mat = vec.fit_transform(valid_texts)
                corp_mat     = mat[:n_corp_ok]
                corp_orig    = [i for i, ok in enumerate(valid_mask[:n_corpus]) if ok]
                new_mat      = mat[n_corp_ok:]

                if new_mat.shape[0] > 0:
                    sims = cosine_similarity(new_mat, corp_mat)
                    ni = 0
                    for orig_i in range(n_new):
                        if valid_mask[n_corpus + orig_i]:
                            best_j  = int(sims[ni].argmax())
                            best_sc = round(float(sims[ni, best_j]) * 100, 1)
                            sim_results.append((best_sc, corp_orig[best_j]))
                            ni += 1
                        else:
                            sim_results.append((0.0, 0))
                else:
                    sim_results = [(0.0, 0)] * n_new
            else:
                sim_results = [(0.0, 0)] * n_new

            s3.update(label="✅ Similarity computed", state="complete")

        # 3d — optional web check
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
                        res = {"plagiarism_score": None,
                               "verdict": f"Error: {e}",
                               "top_sources": "", "matches": []}
                    web_res[row.url] = res
                    wp.progress(i / n_new)
                wt.empty()
                sw.update(label="✅ Web check done", state="complete", expanded=False)

        # assemble results
        final_rows = []
        for row, txt, wc, (sc, ci) in zip(
            new_df.itertuples(), new_texts, new_wc, sim_results
        ):
            corp_row = corpus_df.iloc[ci]
            wr       = web_res.get(row.url, {})
            _, icon, vlabel = verdict_for(sc, dup_threshold)
            final_rows.append({
                "url":                     row.url,
                "entity":                  row.entity,
                "word_count":              wc,
                "similarity_to_corpus_%":  sc,
                "verdict":                 f"{icon} {vlabel}",
                "matched_existing_url":    corp_row["url"],
                "matched_existing_entity": corp_row.get("entity", ""),
                "web_plag_score":          wr.get("plagiarism_score"),
                "web_verdict":             wr.get("verdict", ""),
                "top_web_sources":         wr.get("top_sources", ""),
                "_matches":                wr.get("matches", []),
                "_text":                   txt,
            })

        st.session_state["results"] = final_rows
        st.rerun()

elif corpus_df is None:
    pass  # warning already shown
elif new_df is None:
    st.info("👆 Complete Step 2 to upload the new articles you want to check.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Results
# ─────────────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")
if results:
    rdf = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])

    h1, h2 = st.columns([6, 1], vertical_alignment="bottom")
    with h1:
        step(4, "Results")
    with h2:
        if st.button("🗑️ Clear results", use_container_width=True, key="clr_res"):
            st.session_state.pop("results", None)
            st.rerun()

    # summary tiles
    n_dup  = int((rdf["similarity_to_corpus_%"] >= dup_threshold).sum())
    n_rev  = int(((rdf["similarity_to_corpus_%"] >= 40) &
                  (rdf["similarity_to_corpus_%"] < dup_threshold)).sum())
    n_ok   = int((rdf["similarity_to_corpus_%"] < 40).sum())
    avg_s  = f"{rdf['similarity_to_corpus_%'].mean():.1f}%"

    cols = st.columns(5)
    for col, (val, lab, cls) in zip(cols, [
        (len(rdf),  "Checked",                    "blue"),
        (avg_s,     "Avg similarity",              "blue"),
        (n_dup,     f"🔴 Duplicates (≥{dup_threshold}%)", "red"),
        (n_rev,     "🟡 Review",                   "amber"),
        (n_ok,      "🟢 Unique",                   "green"),
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
            f"{'are' if n_dup>1 else 'is'} ≥{dup_threshold}% similar to an article "
            f"already in your corpus — likely duplicates or rewrites.",
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
                "Entity":           r["entity"],
                "Words":            r["word_count"],
                "Similarity %":     r["similarity_to_corpus_%"],
                "Verdict":          r["verdict"],
                "Closest existing": r["matched_existing_url"],
                "Match entity":     r["matched_existing_entity"],
            } for r in show])

            st.dataframe(
                tdf, hide_index=True, use_container_width=True,
                column_config={
                    "New Article":      st.column_config.LinkColumn(width="large"),
                    "Entity":           st.column_config.TextColumn(width="small"),
                    "Words":            st.column_config.NumberColumn(width="small"),
                    "Similarity %":     st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.0f%%", width="small"),
                    "Verdict":          st.column_config.TextColumn(width="medium"),
                    "Closest existing": st.column_config.LinkColumn(width="large"),
                    "Match entity":     st.column_config.TextColumn(width="small"),
                },
            )

            flagged = [r for r in show if r["similarity_to_corpus_%"] >= 40]
            if flagged:
                st.markdown("#### Detailed cards")
                for r in flagged:
                    sc = r["similarity_to_corpus_%"]
                    cls, icon, vlabel = verdict_for(sc, dup_threshold)
                    new_e    = html_module.escape(r["url"])
                    corp_u   = html_module.escape(r["matched_existing_url"], quote=True)
                    corp_ue  = html_module.escape(r["matched_existing_url"])
                    corp_ent = html_module.escape(str(r.get("matched_existing_entity", "")))
                    new_ent  = html_module.escape(str(r.get("entity", "")))
                    st.markdown(
                        f'<div class="res-card {cls}" '
                        f'style="display:flex;gap:1rem;align-items:flex-start;">'
                        f'<div class="rcscore">{sc:.0f}%</div>'
                        f'<div class="rcbody">'
                        f'<b>New:</b> <a href="{new_e}" target="_blank">{new_e}</a> '
                        f'<span style="background:#e0e7ff;color:#3730a3;padding:2px 8px;'
                        f'border-radius:99px;font-size:.75rem;font-weight:600;">{new_ent}</span>'
                        f'<br><b>Matches:</b> <a href="{corp_u}" target="_blank">{corp_ue}</a> '
                        f'<span style="background:#fce7f3;color:#9d174d;padding:2px 8px;'
                        f'border-radius:99px;font-size:.75rem;font-weight:600;">{corp_ent}</span>'
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
        "Columns: url · entity · word_count · similarity_to_corpus_% · verdict · "
        "matched_existing_url · matched_existing_entity · risk_level"
        + (" · web_plag_score · web_verdict · top_web_sources" if run_web else "")
    )
