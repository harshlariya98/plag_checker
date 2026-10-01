"""
Internal Plagiarism Checker — /internal-plag-checker
Loads articles from a CSV file or directly from an RDS database.
Each article is identified by: url, entity, description (stored HTML).
"""

import html as html_module
import time
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# allow importing plag_utils from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from plag_utils import (
    extract_text_from_html,
    build_passages,
    check_article,
    compute_similarity_matrix,
    build_duplicate_clusters,
    get_duplicate_pairs,
    normalize,
    domain_of,
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
.block-container {padding-top: 1.5rem !important; max-width: 1350px; padding-bottom: 3rem;}
section[data-testid="stSidebar"] {background: #0f172a;}
section[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextArea textarea {
    background: #1e293b !important; border-color: #334155 !important; color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] hr {border-color: #334155 !important;}
section[data-testid="stSidebar"] label {color: #94a3b8 !important; font-size:.8rem !important;}

.hero {
    background: linear-gradient(135deg, #0c1445 0%, #1a237e 40%, #283593 100%);
    border-radius: 20px; padding: 1.75rem 2.25rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(63,81,181,.4);
    box-shadow: 0 20px 60px rgba(26,35,126,.3);
}
.hero h1 {color:#fff; font-size:1.9rem; margin:0 0 .4rem; font-weight:800; letter-spacing:-.5px;}
.hero p  {color:rgba(255,255,255,.8); margin:0; font-size:.97rem; line-height:1.5;}
.chips   {margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap;}
.chip    {
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2);
    color:#c5cae9 !important; padding:4px 13px; border-radius:99px; font-size:.78rem; font-weight:500;
}

.step-label {display:flex; align-items:center; gap:.6rem; font-size:1rem; font-weight:700;
             color:#1e293b; margin:1.5rem 0 .6rem;}
.step-badge {
    width:1.75rem; height:1.75rem; border-radius:50%;
    background:linear-gradient(135deg,#3f51b5,#7c4dff);
    color:#fff; display:inline-flex; align-items:center; justify-content:center;
    font-size:.85rem; font-weight:700; flex-shrink:0;
    box-shadow:0 2px 8px rgba(63,81,181,.4);
}

.metric-tile {
    background:#fff; border-radius:14px; padding:1rem 1.25rem;
    border:1px solid #e2e8f0; box-shadow:0 1px 6px rgba(0,0,0,.05);
    text-align:center; height:100%;
}
.metric-tile .mt-val {font-size:1.9rem; font-weight:800; color:#1e293b; line-height:1.1;}
.metric-tile .mt-lab {font-size:.75rem; color:#64748b; margin-top:.3rem; font-weight:500; text-transform:uppercase; letter-spacing:.03em;}
.metric-tile.red   .mt-val {color:#dc2626;}
.metric-tile.amber .mt-val {color:#d97706;}
.metric-tile.green .mt-val {color:#16a34a;}
.metric-tile.blue  .mt-val {color:#3f51b5;}
.metric-tile.purple .mt-val {color:#7c4dff;}

.callout {
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px;
    padding:.85rem 1.1rem; font-size:.88rem; color:#1e3a5f; line-height:1.55;
}
.callout.warn {background:#fffbeb; border-color:#fde68a; color:#78350f;}
.callout.danger {background:#fef2f2; border-color:#fecaca; color:#7f1d1d;}
.callout b {color:#1e40af;}

.dup-row {
    display:flex; align-items:flex-start; gap:.75rem; padding:.75rem 1rem;
    border-radius:12px; background:#fff; border:1px solid #e2e8f0;
    box-shadow:0 1px 4px rgba(0,0,0,.05); margin:.4rem 0;
}
.dup-score {
    min-width:3rem; text-align:center; font-size:1.15rem; font-weight:800;
    padding:.1rem .3rem; border-radius:8px; background:#fef2f2; color:#dc2626;
}
.dup-score.amber {background:#fffbeb; color:#d97706;}
.dup-score.blue  {background:#eff6ff; color:#3f51b5;}
.dup-urls {flex:1; font-size:.85rem; color:#374151; line-height:1.6;}
.dup-urls a {color:#3f51b5; text-decoration:none;}
.dup-urls a:hover {text-decoration:underline;}

.entity-pill {
    display:inline-block; padding:2px 10px; border-radius:99px; font-size:.75rem;
    font-weight:600; background:#e0e7ff; color:#3730a3; margin-left:.4rem;
}

div[data-testid="stMetric"] {border:1px solid #e2e8f0; border-radius:12px; padding:.75rem 1rem;}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#3f51b5,#7c4dff) !important;
    border:none !important; font-weight:600 !important;
    box-shadow:0 4px 14px rgba(63,81,181,.35) !important;
}
div[data-testid="stRadio"] > label:first-child {display:none;}
div[data-testid="stExpander"] {border-radius:12px !important;}
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


def score_color(s):
    if s >= 70: return "red"
    if s >= 45: return "amber"
    return "blue"


def verdict_for_score(s):
    if s >= 70: return ("🔴", "High duplicate risk")
    if s >= 45: return ("🟡", "Review — moderate overlap")
    return ("🟢", "Low overlap")


@st.cache_data(ttl=300, show_spinner=False)
def load_from_rds(db_type, host, port, dbname, user, password, query):
    """Cached RDS fetch — refreshes every 5 minutes or on manual refresh."""
    if db_type == "PostgreSQL":
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    else:
        conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(conn_str, connect_args={"connect_timeout": 15})
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return df, None
    except Exception as e:
        return None, str(e)


def normalize_columns(df):
    """Map any common column name variants to url / entity / description."""
    col_map = {}
    lower = {c.lower().strip(): c for c in df.columns}
    for target, variants in {
        "url":         ["url", "link", "page_url", "article_url", "slug"],
        "entity":      ["entity", "entity_type", "type", "category", "tag"],
        "description": ["description", "content", "article", "body", "html", "article_html", "text"],
    }.items():
        for v in variants:
            if v in lower:
                col_map[lower[v]] = target
                break
    df = df.rename(columns=col_map)
    missing = [c for c in ("url", "entity", "description") if c not in df.columns]
    if missing:
        return df, f"Could not find columns: {missing}. Please map them below."
    return df[["url", "entity", "description"] + [c for c in df.columns if c not in ("url","entity","description")]].copy(), None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis settings")
    st.divider()

    st.markdown("**🔁 Duplicate threshold**")
    dup_threshold = st.slider("Flag as duplicate above (%)", 30, 100, 60,
                              help="Articles above this TF-IDF similarity are flagged as duplicates.")
    cluster_threshold = st.slider("Cluster threshold (%)", 30, 100, 65,
                                  help="Articles above this score are grouped into the same cluster.")

    st.divider()
    st.markdown("**🌐 Web plagiarism check**")
    run_web = st.checkbox("Also check against the open web", value=False,
                          help="Slower (DuckDuckGo search per passage). "
                               "Only runs on articles above the duplicate threshold OR all if unchecked.")
    if run_web:
        web_only_high = st.checkbox("Only web-check high-duplicate articles", value=True)
        n_passages = st.slider("Passages per article", 4, 20, 8)
        match_threshold = st.slider("Match strictness (%)", 70, 100, 85)
        own_domain = st.text_input("Your domain (ignored as source)", "kollegeapply.com")
        ignored_domains = st.text_area("Ignored domains (one per line)",
                                       "wikipedia.org\nyoutube.com", height=70)

    st.divider()
    st.markdown("**📊 Display**")
    top_n_pairs = st.number_input("Show top N duplicate pairs", 10, 500, 50)
    st.caption(
        "**Similarity guide:**  \n"
        "🟢 **<45%** — low overlap  \n"
        "🟡 **45–70%** — review  \n"
        "🔴 **>70%** — high duplicate risk"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔁 Internal Plagiarism Checker</h1>
  <p>Load articles from a CSV or your RDS database · parse stored HTML · find duplicate clusters · optionally check against the open web</p>
  <div class="chips">
    <span class="chip">📂 CSV or RDS</span>
    <span class="chip">🏷️ Filter by entity type</span>
    <span class="chip">⚡ No URL fetching needed</span>
    <span class="chip">🔗 Cluster duplicate groups</span>
    <span class="chip">📊 Entity-type breakdown</span>
    <span class="chip">⬇️ Enriched CSV export</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data source
# ─────────────────────────────────────────────────────────────────────────────
step(1, "Load your articles")

source = st.radio(
    "Source",
    ["📄 Upload CSV", "🛢️ RDS Database"],
    horizontal=True,
)

raw_df = None

# ── CSV ──────────────────────────────────────────────────────────────────────
if source == "📄 Upload CSV":
    with st.container(border=True):
        st.markdown(
            '<div class="callout">Upload a CSV with at least these columns: '
            '<b>url</b>, <b>entity</b>, <b>description</b>. '
            'The <code>description</code> column should contain the article HTML. '
            'Column names like <i>article_html</i>, <i>body</i>, <i>content</i>, <i>entity_type</i> '
            'are auto-detected.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            try:
                raw_df = pd.read_csv(uploaded)
                st.success(f"✅ {len(raw_df):,} rows loaded")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

# ── RDS ──────────────────────────────────────────────────────────────────────
else:
    with st.container(border=True):
        st.markdown(
            '<div class="callout">'
            'Connect to your AWS RDS database. Credentials are stored only in your '
            'browser session and never saved to disk. The query result is <b>cached for 5 minutes</b> '
            '— use the Refresh button to force a re-fetch (useful for cron-updated tables).'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        c1, c2, c3 = st.columns([2, 3, 1])
        db_type = c1.selectbox("Database type", ["PostgreSQL", "MySQL"])
        host    = c2.text_input("Host", placeholder="mydb.xxxx.us-east-1.rds.amazonaws.com")
        port    = c3.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306, step=1)

        c4, c5 = st.columns(2)
        dbname   = c4.text_input("Database name")
        user     = c5.text_input("Username")
        password = st.text_input("Password", type="password")

        st.markdown("**Query** — must return columns: `url`, `entity` (or `entity_type`), `description`")
        query = st.text_area(
            "SQL query",
            value="SELECT url, entity_type AS entity, description\nFROM articles\nWHERE is_published = true\nORDER BY updated_at DESC\nLIMIT 5000;",
            height=110,
            label_visibility="collapsed",
        )

        col_btn, col_last = st.columns([2, 3])
        fetch_btn = col_btn.button("🔌 Connect & Fetch", type="primary", use_container_width=True)
        refresh_btn = col_btn.button("🔄 Force refresh", use_container_width=True)

        if refresh_btn:
            load_from_rds.clear()
            st.rerun()

        if fetch_btn:
            if not all([host, dbname, user, password]):
                st.error("Please fill in all connection fields.")
            else:
                with st.spinner("Connecting to RDS…"):
                    df_rds, err = load_from_rds(db_type, host, port, dbname, user, password, query)
                if err:
                    st.error(f"Connection failed: {err}")
                    st.markdown(
                        '<div class="callout warn">Make sure your RDS security group allows '
                        'inbound connections from this machine\'s IP on the database port.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state["rds_df"] = df_rds
                    st.success(f"✅ {len(df_rds):,} rows fetched from RDS  ·  {datetime.now().strftime('%H:%M:%S')}")

        if "rds_df" in st.session_state:
            raw_df = st.session_state["rds_df"]
            col_last.markdown(
                f'<div style="padding:.5rem 0; font-size:.82rem; color:#64748b;">'
                f'📋 Using cached data · {len(raw_df):,} rows</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Normalise + column mapping
# ─────────────────────────────────────────────────────────────────────────────
if raw_df is not None:
    df, col_err = normalize_columns(raw_df)

    if col_err:
        st.warning(col_err)
        st.markdown("**Map your columns manually:**")
        all_cols = list(raw_df.columns)
        mc1, mc2, mc3 = st.columns(3)
        url_col  = mc1.selectbox("URL column",         all_cols, index=0)
        ent_col  = mc2.selectbox("Entity column",      all_cols, index=min(1, len(all_cols)-1))
        desc_col = mc3.selectbox("Description column", all_cols, index=min(2, len(all_cols)-1))
        df = raw_df.rename(columns={url_col:"url", ent_col:"entity", desc_col:"description"})
        df = df[["url", "entity", "description"]].copy()

    df["url"]         = df["url"].astype(str).str.strip()
    df["entity"]      = df["entity"].astype(str).str.strip()
    df["description"] = df["description"].astype(str)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Preview & filter
    # ─────────────────────────────────────────────────────────────────────────
    step(2, "Preview & filter")

    with st.container(border=True):
        entity_types = sorted(df["entity"].dropna().unique().tolist())
        c1, c2 = st.columns([3, 1])
        chosen_entities = c1.multiselect(
            "Filter by entity type", entity_types, default=entity_types,
            help="Leave all selected to check everything.",
        )
        max_rows = c2.number_input("Max articles to analyse", 100, 50000, 5000, step=500)

        df = df[df["entity"].isin(chosen_entities)].head(int(max_rows)).reset_index(drop=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Articles selected", f"{len(df):,}")
        mc2.metric("Entity types", len(chosen_entities))
        # preview description length
        preview_len = df["description"].str.len().median()
        mc3.metric("Median description size", f"{int(preview_len):,} chars")

        with st.expander(f"Preview first 10 rows"):
            prev = df[["url", "entity"]].copy()
            prev["description_preview"] = df["description"].str[:120] + "…"
            st.dataframe(prev, hide_index=True, use_container_width=True,
                         column_config={"url": st.column_config.LinkColumn()})

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Run
    # ─────────────────────────────────────────────────────────────────────────
    step(3, "Run analysis")

    with st.container(border=True):
        n = len(df)
        est_parse = max(1, round(n * 0.05))         # ~50ms per HTML parse
        est_tfidf = max(1, round(n * n * 0.00001))  # matrix computation
        st.markdown(
            f'**{n:,} articles** will be analysed.  \n'
            f'⚡ HTML parsing + TF-IDF matrix: **~{est_parse + est_tfidf}s** (fast, no web requests).  \n'
            + (f'🌐 Web plagiarism check: **~{max(1, round(n*8/60))} min** (DuckDuckGo per article).'
               if run_web else
               '💡 Web plagiarism check is **off** — enable in the sidebar for deeper analysis.')
        )
        run_btn = st.button("🚀 Run internal analysis", type="primary", use_container_width=True)

    if run_btn:
        # ── 3a: parse HTML ───────────────────────────────────────────────────
        with st.status("⚡ Parsing HTML descriptions…", expanded=True) as status:
            prog = st.progress(0.0)
            texts, word_counts = [], []
            for i, row in enumerate(df.itertuples(), 1):
                t = extract_text_from_html(row.description)
                texts.append(t)
                word_counts.append(len(t.split()))
                if i % max(1, n // 20) == 0 or i == n:
                    prog.progress(i / n)
            status.update(label=f"✅ Parsed {n:,} articles", state="complete", expanded=False)

        # ── 3b: similarity matrix + clusters ─────────────────────────────────
        with st.status("🔁 Building TF-IDF similarity matrix…", expanded=False) as status:
            sim_matrix, clusters = build_duplicate_clusters(
                df["url"].tolist(), texts, threshold=cluster_threshold / 100
            )
            dup_pairs = get_duplicate_pairs(
                df["url"].tolist(), sim_matrix, threshold=dup_threshold / 100
            )

            # assign cluster ids
            url_to_cluster = {}
            for cid, members in clusters.items():
                for idx in members:
                    url_to_cluster[df["url"].iloc[idx]] = cid

            # per-article best match
            best_match = {}
            for i, url in enumerate(df["url"]):
                row_sim = sim_matrix[i].copy()
                row_sim[i] = 0
                j = int(row_sim.argmax())
                best_match[url] = {
                    "closest_url": df["url"].iloc[j] if row_sim[j] > 0 else "",
                    "internal_similarity_%": round(float(row_sim[j]) * 100, 1),
                }

            status.update(label=f"✅ Similarity matrix done — {len(dup_pairs)} duplicate pairs found",
                          state="complete", expanded=False)

        # ── 3c: optional web check ────────────────────────────────────────────
        web_results = {}
        if run_web:
            excl = {d.strip().lower() for d in ignored_domains.splitlines() if d.strip()}
            if own_domain.strip():
                excl.add(own_domain.strip().lower().replace("www.", ""))

            to_check = df["url"].tolist()
            if web_only_high:
                to_check = [
                    url for url in to_check
                    if best_match[url]["internal_similarity_%"] >= dup_threshold
                ]

            if to_check:
                with st.status(f"🌐 Web-checking {len(to_check)} articles…", expanded=True) as ws:
                    wp = st.progress(0.0)
                    wt = st.empty()
                    for i, url in enumerate(to_check, 1):
                        wt.markdown(f"**{i}/{len(to_check)}** · `{url}`")
                        txt = texts[df["url"].tolist().index(url)]
                        try:
                            res, _ = check_article(
                                url, n_passages, match_threshold, excl, 25,
                                preloaded_text=txt,
                            )
                        except Exception as e:
                            res = {"plagiarism_score": None, "verdict": f"Error: {e}",
                                   "top_sources": "", "matches": []}
                        web_results[url] = res
                        wp.progress(i / len(to_check))
                    wt.empty()
                    ws.update(label=f"✅ Web check done for {len(to_check)} articles",
                              state="complete", expanded=False)

        # ── assemble results dataframe ─────────────────────────────────────────
        result_rows = []
        for i, row in enumerate(df.itertuples()):
            bm = best_match[row.url]
            cluster_id = url_to_cluster.get(row.url, None)
            wr = web_results.get(row.url, {})
            result_rows.append({
                "url":                   row.url,
                "entity":                row.entity,
                "word_count":            word_counts[i],
                "internal_similarity_%": bm["internal_similarity_%"],
                "closest_duplicate_url": bm["closest_url"],
                "cluster_id":            int(cluster_id) if cluster_id is not None else None,
                "web_plag_score":        wr.get("plagiarism_score"),
                "web_verdict":           wr.get("verdict", ""),
                "top_web_sources":       wr.get("top_sources", ""),
            })

        st.session_state["int_results"] = {
            "rows": result_rows,
            "pairs": dup_pairs,
            "clusters": clusters,
            "sim_matrix": sim_matrix,
            "urls": df["url"].tolist(),
            "entities": df["entity"].tolist(),
            "word_counts": word_counts,
            "run_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Results
# ─────────────────────────────────────────────────────────────────────────────
res = st.session_state.get("int_results")
if res:
    rows       = res["rows"]
    pairs      = res["pairs"]
    clusters   = res["clusters"]
    sim_matrix = res["sim_matrix"]
    urls_list  = res["urls"]
    ents_list  = res["entities"]
    run_at     = res["run_at"]

    rdf = pd.DataFrame(rows)
    scored = rdf[rdf["internal_similarity_%"].notna()]

    # ── header ───────────────────────────────────────────────────────────────
    hc1, hc2 = st.columns([6, 1], vertical_alignment="bottom")
    with hc1:
        step(4, f"Results  · {run_at}")
    with hc2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("int_results", None)
            st.rerun()

    # ── summary tiles ─────────────────────────────────────────────────────────
    n_high   = int((rdf["internal_similarity_%"] >= 70).sum())
    n_med    = int(((rdf["internal_similarity_%"] >= 45) & (rdf["internal_similarity_%"] < 70)).sum())
    n_ok     = int((rdf["internal_similarity_%"] < 45).sum())
    n_clust  = len(clusters)
    n_pairs  = len(pairs)
    avg_sim  = f"{scored['internal_similarity_%'].mean():.1f}%" if len(scored) else "–"

    tiles = [
        ("Articles", f"{len(rdf):,}", "blue"),
        ("Avg similarity", avg_sim, "blue"),
        ("🔴 High risk", n_high, "red"),
        ("🟡 Review", n_med, "amber"),
        ("🟢 Unique", n_ok, "green"),
        ("Dup clusters", n_clust, "purple"),
    ]
    cols = st.columns(6)
    for col, (label, val, color) in zip(cols, tiles):
        col.markdown(
            f'<div class="metric-tile {color}"><div class="mt-val">{val}</div>'
            f'<div class="mt-lab">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # ── tabs ─────────────────────────────────────────────────────────────────
    tab_pairs, tab_clusters, tab_entity, tab_all, tab_web = st.tabs([
        f"🔗 Duplicate pairs ({min(n_pairs, int(top_n_pairs))})",
        f"🗂️ Clusters ({n_clust})",
        "🏷️ By entity type",
        "📋 All articles",
        "🌐 Web results" if run_web else "🌐 Web results (off)",
    ])

    # ═══ TAB 1: Duplicate pairs ══════════════════════════════════════════════
    with tab_pairs:
        if not pairs:
            st.success(f"✅ No pairs found above {dup_threshold}% similarity.", icon="✅")
        else:
            st.markdown(
                f'<div class="callout">Showing <b>top {min(len(pairs), int(top_n_pairs))}</b> '
                f'of <b>{len(pairs)}</b> duplicate pairs above <b>{dup_threshold}%</b> threshold. '
                f'These article pairs share the most content — review them for near-duplicate or '
                f'spun content.</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            # threshold filter
            filt_thresh = st.slider("Show pairs above (%)", 30, 100, int(dup_threshold),
                                    key="pairs_thresh")
            show_pairs = [p for p in pairs if p["similarity_%"] >= filt_thresh][:int(top_n_pairs)]

            if not show_pairs:
                st.info(f"No pairs above {filt_thresh}%.")
            else:
                # render as rich rows
                for p in show_pairs:
                    sc = p["similarity_%"]
                    color = score_color(sc)
                    icon, label = verdict_for_score(sc)
                    # get entities
                    ent_a = ents_list[urls_list.index(p["url_a"])] if p["url_a"] in urls_list else ""
                    ent_b = ents_list[urls_list.index(p["url_b"])] if p["url_b"] in urls_list else ""
                    st.markdown(
                        f'<div class="dup-row">'
                        f'<div class="dup-score {color}">{sc:.0f}%</div>'
                        f'<div class="dup-urls">'
                        f'<a href="{html_module.escape(p["url_a"],quote=True)}" target="_blank">{html_module.escape(p["url_a"])}</a>'
                        f'<span class="entity-pill">{html_module.escape(ent_a)}</span><br>'
                        f'<a href="{html_module.escape(p["url_b"],quote=True)}" target="_blank">{html_module.escape(p["url_b"])}</a>'
                        f'<span class="entity-pill">{html_module.escape(ent_b)}</span>'
                        f'</div>'
                        f'<div style="font-size:.8rem;color:#64748b;white-space:nowrap;">{icon} {label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # also a compact table for easy export
                with st.expander("📋 Table view"):
                    pair_df = pd.DataFrame(show_pairs)
                    pair_df.insert(2, "entity_a",
                                   pair_df["url_a"].map(dict(zip(urls_list, ents_list))))
                    pair_df.insert(4, "entity_b",
                                   pair_df["url_b"].map(dict(zip(urls_list, ents_list))))
                    st.dataframe(pair_df, hide_index=True, use_container_width=True,
                                 column_config={
                                     "url_a": st.column_config.LinkColumn("Article A"),
                                     "url_b": st.column_config.LinkColumn("Article B"),
                                     "similarity_%": st.column_config.ProgressColumn(
                                         "Similarity", min_value=0, max_value=100, format="%.0f%%"),
                                 })

    # ═══ TAB 2: Clusters ═════════════════════════════════════════════════════
    with tab_clusters:
        if not clusters:
            st.success(f"✅ No duplicate clusters found above {cluster_threshold}% threshold.")
        else:
            st.markdown(
                f'<div class="callout">'
                f'<b>{n_clust} clusters</b> of articles share content above {cluster_threshold}%. '
                f'All articles in the same cluster are likely covering the same topic with overlapping wording — '
                f'consider consolidating or rewriting them.'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            for cid, member_idxs in sorted(clusters.items(), key=lambda x: -len(x[1])):
                member_urls  = [urls_list[i] for i in member_idxs]
                member_ents  = [ents_list[i]  for i in member_idxs]
                member_wc    = [res["word_counts"][i] for i in member_idxs]
                entity_set   = sorted(set(member_ents))
                avg_wc       = int(sum(member_wc) / len(member_wc))

                label = (f"Cluster {cid} — {len(member_idxs)} articles · "
                         f"entities: {', '.join(entity_set)} · avg {avg_wc:,} words")
                with st.expander(label, expanded=len(member_idxs) >= 3):
                    cluster_df = pd.DataFrame({
                        "URL": member_urls,
                        "Entity": member_ents,
                        "Words": member_wc,
                    })
                    st.dataframe(cluster_df, hide_index=True, use_container_width=True,
                                 column_config={"URL": st.column_config.LinkColumn()})

    # ═══ TAB 3: By entity type ═══════════════════════════════════════════════
    with tab_entity:
        entity_stats = []
        for ent in sorted(rdf["entity"].unique()):
            sub = rdf[rdf["entity"] == ent]
            n_dup  = int((sub["internal_similarity_%"] >= dup_threshold).sum())
            entity_stats.append({
                "Entity type":       ent,
                "Articles":          len(sub),
                "Avg similarity %":  round(sub["internal_similarity_%"].mean(), 1),
                "High risk":         n_dup,
                "High risk %":       round(n_dup / len(sub) * 100, 1) if len(sub) else 0,
                "Avg word count":    int(sub["word_count"].mean()) if "word_count" in sub else 0,
            })
        entity_df = pd.DataFrame(entity_stats).sort_values("High risk %", ascending=False)
        st.dataframe(
            entity_df, hide_index=True, use_container_width=True,
            column_config={
                "Avg similarity %": st.column_config.ProgressColumn(
                    "Avg similarity", min_value=0, max_value=100, format="%.1f%%"),
                "High risk %": st.column_config.ProgressColumn(
                    "High risk %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )

        if n_high:
            worst = entity_df.sort_values("High risk %", ascending=False).iloc[0]
            st.warning(
                f"⚠️ **{worst['Entity type']}** has the highest duplication rate: "
                f"{worst['High risk %']:.1f}% of its articles exceed {dup_threshold}% similarity.",
                icon="⚠️",
            )

    # ═══ TAB 4: All articles ═════════════════════════════════════════════════
    with tab_all:
        fc1, fc2 = st.columns([3, 1])
        risk_filter = fc1.multiselect(
            "Show risk level",
            ["🔴 High risk", "🟡 Review", "🟢 Unique"],
            default=["🔴 High risk", "🟡 Review", "🟢 Unique"],
        )
        sort_opt = fc2.selectbox("Sort by", ["Similarity ↓", "Similarity ↑", "Words ↓"])

        def risk_label(s):
            if s >= 70: return "🔴 High risk"
            if s >= 45: return "🟡 Review"
            return "🟢 Unique"

        all_df = rdf.copy()
        all_df["risk"] = all_df["internal_similarity_%"].apply(risk_label)
        all_df = all_df[all_df["risk"].isin(risk_filter)]
        if sort_opt == "Similarity ↓":
            all_df = all_df.sort_values("internal_similarity_%", ascending=False)
        elif sort_opt == "Similarity ↑":
            all_df = all_df.sort_values("internal_similarity_%", ascending=True)
        else:
            all_df = all_df.sort_values("word_count", ascending=False)

        st.dataframe(
            all_df[["url", "entity", "word_count", "internal_similarity_%",
                     "closest_duplicate_url", "cluster_id", "risk"]],
            hide_index=True, use_container_width=True,
            column_config={
                "url":                   st.column_config.LinkColumn("Article", width="large"),
                "entity":                st.column_config.TextColumn("Entity", width="small"),
                "word_count":            st.column_config.NumberColumn("Words", width="small"),
                "internal_similarity_%": st.column_config.ProgressColumn(
                    "Similarity", min_value=0, max_value=100, format="%.0f%%", width="small"),
                "closest_duplicate_url": st.column_config.LinkColumn("Closest duplicate", width="large"),
                "cluster_id":            st.column_config.NumberColumn("Cluster", width="small"),
                "risk":                  st.column_config.TextColumn("Risk", width="small"),
            },
        )

    # ═══ TAB 5: Web results ══════════════════════════════════════════════════
    with tab_web:
        web_rows = []
        if run_web:
            web_rows = [(r["url"], web_results.get(r["url"])) for r in rows
                        if web_results.get(r["url"]) is not None
                        and web_results.get(r["url"], {}).get("plagiarism_score") is not None]

        if not run_web:
            st.markdown(
                '<div class="callout">Enable <b>web plagiarism check</b> in the sidebar '
                'and re-run the analysis to see results here.</div>',
                unsafe_allow_html=True,
            )
        elif not web_rows:
            st.info("No articles were web-checked (none exceeded the duplicate threshold).")
        else:
            web_df = pd.DataFrame([
                {
                    "url": url,
                    "web_score_%": r["plagiarism_score"],
                    "verdict": r["verdict"],
                    "top_sources": r["top_sources"],
                }
                for url, r in web_rows
            ]).sort_values("web_score_%", ascending=False)

            st.dataframe(
                web_df, hide_index=True, use_container_width=True,
                column_config={
                    "url": st.column_config.LinkColumn("Article", width="large"),
                    "web_score_%": st.column_config.ProgressColumn(
                        "Web plag score", min_value=0, max_value=100, format="%.0f%%"),
                    "verdict":     st.column_config.TextColumn("Verdict"),
                    "top_sources": st.column_config.TextColumn("Top sources", width="large"),
                },
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Download enriched CSV
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### ⬇️ Download enriched CSV")
    st.markdown(
        "The download includes all original columns plus: `word_count`, `internal_similarity_%`, "
        "`closest_duplicate_url`, `cluster_id`, and web plagiarism columns if the check was run."
    )

    export_df = rdf.copy()
    export_df["risk_level"] = export_df["internal_similarity_%"].apply(
        lambda s: "High risk" if s >= 70 else "Review" if s >= 45 else "Unique"
    )
    fname = f"internal_plag_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    st.download_button(
        "⬇️ Download enriched CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        fname,
        "text/csv",
        type="primary",
    )

    # also offer duplicate pairs CSV
    if pairs:
        pairs_df = pd.DataFrame(pairs[:int(top_n_pairs)])
        pairs_df.insert(2, "entity_a", pairs_df["url_a"].map(dict(zip(urls_list, ents_list))))
        pairs_df.insert(4, "entity_b", pairs_df["url_b"].map(dict(zip(urls_list, ents_list))))
        st.download_button(
            f"⬇️ Download duplicate pairs CSV ({len(pairs)} pairs)",
            pairs_df.to_csv(index=False).encode("utf-8"),
            f"dup_pairs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )
