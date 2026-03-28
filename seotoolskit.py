import streamlit as st
import asyncio
import aiohttp
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objs as go
import logging
import io
import numpy as np

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEO Toolkit — Bulk Indexing & Internal Linking",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

# ── Constants ────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/indexing"]
ENDPOINT = "https://indexing.googleapis.com/v3/urlNotifications:publish"
URLS_PER_ACCOUNT = 200
REQUEST_TIMEOUT = 15

# ── Session State ────────────────────────────────────────────────────
if "indexing_results" not in st.session_state:
    st.session_state.indexing_results = None
if "linking_results" not in st.session_state:
    st.session_state.linking_results = None


# =====================================================================
#  PREMIUM THEME — Deep dark with warm amber accents
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-card: rgba(255, 255, 255, 0.03);
        --bg-card-hover: rgba(255, 255, 255, 0.055);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-accent: rgba(212, 175, 55, 0.3);
        --accent: #d4af37;
        --accent-glow: rgba(212, 175, 55, 0.15);
        --accent-soft: #c9a227;
        --text-primary: #e8e6e1;
        --text-secondary: rgba(232, 230, 225, 0.5);
        --text-muted: rgba(232, 230, 225, 0.3);
        --success: #4ade80;
        --error: #f87171;
        --warning: #fbbf24;
        --info: #60a5fa;
    }

    /* ── Global overrides ── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1100px !important;
    }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #0a0a0f 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] * {
        font-family: 'Outfit', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--text-secondary) !important;
    }

    /* ── Headers ── */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-primary) !important;
        font-size: 2.4rem !important;
    }
    h2 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
        border-bottom: none !important;
        padding-bottom: 0 !important;
    }
    h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    p, li, span, div {
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px var(--accent-glow) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 30px rgba(212, 175, 55, 0.25) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    .stButton > button[disabled] {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background: transparent !important;
        color: var(--accent) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: var(--accent-glow) !important;
        color: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    /* ── Text inputs & areas ── */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--border-accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ── Select boxes ── */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    /* ── File uploader ── */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px dashed var(--border-subtle) !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease !important;
    }
    .stFileUploader > div:hover {
        border-color: var(--border-accent) !important;
    }

    /* ── Metrics ── */
    div[data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--border-accent) !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }

    /* ── Dataframes ── */
    .stDataFrame {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), #e8c547) !important;
        border-radius: 10px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--border-subtle) !important;
        margin: 2rem 0 !important;
    }

    /* ── Status widget ── */
    div[data-testid="stStatusWidget"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
    }

    /* ── Custom classes ── */
    .hero-badge {
        display: inline-block;
        background: var(--accent-glow);
        border: 1px solid var(--border-accent);
        color: var(--accent);
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 640px;
        margin-bottom: 0.5rem;
    }
    .section-label {
        display: inline-block;
        color: var(--accent);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        padding: 4px 0;
        border-bottom: 1px solid var(--border-accent);
    }
    .stat-row {
        display: flex;
        gap: 12px;
        margin: 1rem 0;
    }
    .stat-item {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--accent);
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .sidebar-brand {
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .sidebar-brand-name {
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    .sidebar-brand-accent {
        color: var(--accent);
    }
    .sidebar-author {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-subtle);
    }
    .sidebar-author a {
        color: var(--accent) !important;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .sidebar-author-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 1rem 0;
    }
    .feature-item {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        transition: all 0.2s ease;
    }
    .feature-item:hover {
        border-color: var(--border-accent);
        background: var(--bg-card-hover);
    }
    .feature-icon {
        font-size: 1.3rem;
        margin-bottom: 6px;
    }
    .feature-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .feature-desc {
        font-size: 0.75rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  INDEXING FUNCTIONS
# =====================================================================

def setup_http_client(json_key_data: dict) -> str:
    try:
        from google.oauth2 import service_account as sa
        import google.auth.transport.requests
        credentials = sa.Credentials.from_service_account_info(json_key_data, scopes=SCOPES)
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
    except ImportError:
        from oauth2client.service_account import ServiceAccountCredentials
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(json_key_data, scopes=SCOPES)
        return credentials.get_access_token().access_token


async def send_url(session, token, url, action="URL_UPDATED", semaphore=None):
    content = {"url": url.strip(), "type": action}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    for attempt in range(3):
        try:
            if semaphore:
                await semaphore.acquire()
            try:
                async with session.post(ENDPOINT, json=content, headers=headers, ssl=False, timeout=30) as response:
                    result = json.loads(await response.text())
                    if response.status == 429:
                        await asyncio.sleep((2 ** attempt) * 2)
                        continue
                    return result
            finally:
                if semaphore:
                    semaphore.release()
                    await asyncio.sleep(0.5)
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            return {"error": {"code": 500, "message": f"Failed after retries: {str(e)}"}}
    return {"error": {"code": 429, "message": "Rate limited after retries"}}


async def index_urls_batch(token, urls, progress_bar, status_text, action="URL_UPDATED"):
    semaphore = asyncio.Semaphore(5)
    successful, errors_429, other_errors = 0, 0, 0
    results_log = []
    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(urls):
            result = await send_url(session, token, url, action, semaphore)
            if "error" in result:
                code = result["error"].get("code", 0)
                msg = result["error"].get("message", "Unknown")
                if code == 429:
                    errors_429 += 1
                else:
                    other_errors += 1
                results_log.append({"URL": url, "Status": "❌ Error", "Detail": f"{code}: {msg}"})
            else:
                successful += 1
                notify = result.get("urlNotificationMetadata", {}).get("latestUpdate", {}).get("notifyTime", "OK")
                results_log.append({"URL": url, "Status": "✅ Indexed", "Detail": notify})
            progress_bar.progress((i + 1) / len(urls))
            status_text.text(f"Processed {i+1} of {len(urls)}  ·  ✅ {successful}  ·  ⚠️ {errors_429}  ·  ❌ {other_errors}")
    return successful, errors_429, other_errors, results_log


# =====================================================================
#  INTERNAL LINKING FUNCTIONS
# =====================================================================

def get_urls_from_sitemap(sitemap_url: str) -> list:
    urls = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SEOToolkit/1.0)"}
        r = requests.get(sitemap_url.strip(), headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "xml")
        sitemap_tags = soup.find_all("sitemap")
        if sitemap_tags:
            for sm in sitemap_tags:
                loc = sm.find("loc")
                if loc and loc.text:
                    urls.extend(get_urls_from_sitemap(loc.text))
        else:
            urls = [loc.text.strip() for loc in soup.find_all("loc") if loc.text]
    except Exception as e:
        logging.error(f"Sitemap error {sitemap_url}: {e}")
        st.warning(f"Could not fetch: {sitemap_url}")
    return urls


def filter_urls(urls: list) -> list:
    return list(set(u for u in urls if "/page/" not in u and "category" not in u.lower()))


def fetch_page_content(url: str) -> tuple:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SEOToolkit/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None, None, None
    except Exception:
        return None, None, None
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        robots = soup.find("meta", attrs={"name": "robots"})
        if robots and "noindex" in (robots.get("content", "").lower()):
            return None, None, None
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        h = soup.find(["h1", "h2", "h3"])
        heading = h.get_text(strip=True) if h else ""
        body = soup.find("main") or soup.find("article") or soup.find("body")
        if body:
            for tag in body(["header", "nav", "footer", "aside", "script", "style", "noscript"]):
                tag.decompose()
            text = " ".join(body.stripped_strings)
        else:
            text = ""
        if len(text) < 50:
            return None, None, None
        return title, heading, text
    except Exception:
        return None, None, None


def cluster_and_link(page_contents, valid_urls):
    vectorizer = TfidfVectorizer(stop_words="english", use_idf=True, max_features=10000, min_df=2, max_df=0.95)
    tfidf = vectorizer.fit_transform(page_contents)
    if len(valid_urls) < 2:
        return [0], {}
    try:
        sim = cosine_similarity(tfidf)
        dist = np.maximum(1 - sim, 0)
        np.fill_diagonal(dist, 0)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7, metric="precomputed", linkage="average")
        clustering.fit(dist)
        labels = clustering.labels_
    except Exception:
        labels = [0] * len(valid_urls)
    df = pd.DataFrame({"url": valid_urls, "label": labels})
    plan = {}
    for label in df["label"].unique():
        cluster = df[df["label"] == label]["url"].tolist()
        for url in cluster:
            plan[url] = [l for l in cluster if l != url]
    return labels, plan


def save_to_excel_bytes(urls, labels, plan):
    clusters_df = pd.DataFrame({"URL": urls, "Cluster": labels})
    links_df = pd.DataFrame([(s, t) for s, ts in plan.items() for t in ts], columns=["Source URL", "Target URL"])
    unique = sorted(set(labels))
    summary_df = pd.DataFrame({
        "Cluster": unique,
        "Count": [list(labels).count(l) for l in unique],
        "URLs": [", ".join(clusters_df[clusters_df["Cluster"] == l]["URL"].tolist()) for l in unique],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        clusters_df.to_excel(w, sheet_name="Clusters", index=False)
        links_df.to_excel(w, sheet_name="Linking Plan", index=False)
        summary_df.to_excel(w, sheet_name="Summary", index=False)
    return buf.getvalue()


def build_network_graph(plan, labels, urls):
    G = nx.Graph()
    url_label = dict(zip(urls, labels))
    for s, ts in plan.items():
        for t in ts:
            G.add_edge(s, t)
    if not G.nodes():
        return None
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    ex, ey = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.4, color="rgba(212,175,55,0.2)"), hoverinfo="none")
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
        mode="markers", hoverinfo="text",
        text=[f"{n}<br>Cluster {url_label.get(n, '?')}<br>{G.degree(n)} links" for n in G.nodes()],
        marker=dict(
            showscale=True, colorscale=[[0, "#1a1a2e"], [0.5, "#d4af37"], [1, "#f5e6a3"]],
            color=[url_label.get(n, 0) for n in G.nodes()], size=8,
            line=dict(width=1, color="rgba(212,175,55,0.4)"),
            colorbar=dict(thickness=12, title=dict(text="Cluster", font=dict(color="#888", size=11)), tickfont=dict(color="#666")),
        ),
    )
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hovermode="closest", margin=dict(b=10, l=10, r=10, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(family="Outfit", color="#888"),
    ))
    return fig


# =====================================================================
#  SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name"><span class="sidebar-brand-accent">◆</span> SEO Toolkit</div>
        </div>
    """, unsafe_allow_html=True)

    tool = st.selectbox("Select Tool", ["Bulk API Indexing", "Internal Linking"], label_visibility="collapsed")

    st.markdown("""
        <div class="sidebar-author">
            <div class="sidebar-author-label">Crafted by</div>
            <a href="https://in.linkedin.com/in/neeraj-kumar-seo" target="_blank">Neeraj Kumar</a>
        </div>
    """, unsafe_allow_html=True)


# =====================================================================
#  TOOL 1: BULK INDEXING
# =====================================================================

if tool == "Bulk API Indexing":

    st.markdown('<div class="hero-badge">Indexing API</div>', unsafe_allow_html=True)
    st.title("Bulk URL Indexing")
    st.markdown('<p class="hero-subtitle">Submit hundreds of URLs to Google\'s Indexing API with automatic rate limiting, multi-account rotation, and detailed result tracking.</p>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown("""
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">🔑</div>
                <div class="feature-title">Multi-Account</div>
                <div class="feature-desc">Upload multiple JSON keys — 200 URLs each, auto-rotated</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Rate Limited</div>
                <div class="feature-desc">Smart throttling with exponential backoff on 429 errors</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Live Progress</div>
                <div class="feature-desc">Real-time progress bar and per-URL status tracking</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📥</div>
                <div class="feature-title">Export Results</div>
                <div class="feature-desc">Download detailed CSV report of all submissions</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-label">Step 1 — Service Account Keys</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload JSON key files", type="json", accept_multiple_files=True,
        help="Each service account supports up to 200 URL submissions per day.")

    if not uploaded_files:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color: var(--text-muted);">
            <p style="font-size: 2rem; margin-bottom: 12px;">🔑</p>
            <p style="font-size: 0.9rem;">Upload your service account JSON key files to get started</p>
            <p style="font-size: 0.75rem; margin-top: 8px;">Each key supports up to 200 URL submissions per day</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_files:
        num_files = len(uploaded_files)
        max_urls = num_files * URLS_PER_ACCOUNT
        st.markdown(f"""
            <div class="stat-row">
                <div class="stat-item"><div class="stat-value">{num_files}</div><div class="stat-label">Accounts</div></div>
                <div class="stat-item"><div class="stat-value">{max_urls}</div><div class="stat-label">Max URLs</div></div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Step 2 — URLs</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 1])
        with col_b:
            action = st.selectbox("Action", ["URL_UPDATED", "URL_DELETED"])

        url_input = st.text_area("Paste URLs (one per line)", height=180,
            placeholder="https://yoursite.com/page-1\nhttps://yoursite.com/page-2\nhttps://yoursite.com/page-3")
        urls = [u.strip() for u in url_input.split("\n") if u.strip().startswith("http")]

        csv_up = st.file_uploader("Or upload a CSV", type=["csv"], key="idx_csv")
        if csv_up:
            try:
                df = pd.read_csv(csv_up)
                url_col = df.columns[0]
                for c in df.columns:
                    if c.strip().lower() in ("url", "urls", "link"):
                        url_col = c
                        break
                csv_urls = [u.strip() for u in df[url_col].dropna().astype(str) if u.strip().startswith("http")]
                urls.extend(csv_urls)
                st.success(f"Loaded {len(csv_urls)} URLs from CSV")
            except Exception as e:
                st.error(f"CSV error: {e}")

        urls = list(dict.fromkeys(urls))
        if urls:
            st.markdown(f'<p style="color: var(--text-secondary); font-size: 0.85rem;">📋 {len(urls)} unique URLs ready</p>', unsafe_allow_html=True)
        if len(urls) > max_urls:
            st.warning(f"Trimmed to {max_urls} URLs (account limit).")
            urls = urls[:max_urls]

        st.markdown("---")
        st.markdown('<div class="section-label">Step 3 — Submit</div>', unsafe_allow_html=True)

        if st.button("◆  Start Indexing", disabled=len(urls) == 0, use_container_width=True):
            all_results = []
            t_s, t_429, t_e = 0, 0, 0
            for i, uf in enumerate(uploaded_files):
                batch = urls[i * URLS_PER_ACCOUNT : (i + 1) * URLS_PER_ACCOUNT]
                if not batch:
                    break
                st.markdown(f'<div class="section-label">Account {i+1}: {uf.name}</div>', unsafe_allow_html=True)
                try:
                    uf.seek(0)
                    token = setup_http_client(json.load(uf))
                except Exception as e:
                    st.error(f"Auth failed: {e}")
                    continue
                pb = st.progress(0)
                st_text = st.empty()
                s, e4, eo, logs = asyncio.run(index_urls_batch(token, batch, pb, st_text, action))
                t_s += s; t_429 += e4; t_e += eo
                all_results.extend(logs)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(urls))
            c2.metric("Indexed", t_s)
            c3.metric("Rate Limited", t_429)
            c4.metric("Errors", t_e)
            if all_results:
                rdf = pd.DataFrame(all_results)
                st.dataframe(rdf, use_container_width=True, hide_index=True)
                st.download_button("📥  Download Results", rdf.to_csv(index=False), "indexing_results.csv", "text/csv")


# =====================================================================
#  TOOL 2: INTERNAL LINKING
# =====================================================================

if tool == "Internal Linking":

    st.markdown('<div class="hero-badge">Content Clustering</div>', unsafe_allow_html=True)
    st.title("Internal Linking Planner")
    st.markdown('<p class="hero-subtitle">Analyze your site\'s content structure, discover topical clusters, and generate an intelligent internal linking strategy — powered by TF-IDF and hierarchical clustering.</p>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown("""
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">🗺️</div>
                <div class="feature-title">Sitemap Parsing</div>
                <div class="feature-desc">Handles sitemap indexes, nested sitemaps, and direct URL lists</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🧬</div>
                <div class="feature-title">Content Clustering</div>
                <div class="feature-desc">Groups pages by semantic similarity using TF-IDF vectors</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🕸️</div>
                <div class="feature-title">Link Suggestions</div>
                <div class="feature-desc">Auto-generates internal linking targets within each cluster</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Visual Network</div>
                <div class="feature-desc">Interactive graph visualization of your site's link topology</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Sitemap URLs</div>', unsafe_allow_html=True)
    sitemap_input = st.text_area("Enter individual sitemap URLs (not sitemap index unless intended)", height=100,
        placeholder="https://yoursite.com/post-sitemap.xml\nhttps://yoursite.com/page-sitemap.xml")
    sitemap_urls = [u.strip() for u in sitemap_input.split("\n") if u.strip().startswith("http")]

    col1, col2 = st.columns([1, 2])
    with col1:
        max_pages = st.slider("Max pages", 10, 500, 100, 10, help="Limits pages analyzed to keep processing fast.")

    st.markdown("")

    if st.button("◆  Generate Linking Plan", disabled=len(sitemap_urls) == 0, use_container_width=True):

        with st.status("Fetching URLs from sitemaps...", expanded=True) as status:
            all_urls = []
            for su in sitemap_urls:
                st.write(f"→ {su}")
                fetched = get_urls_from_sitemap(su)
                st.write(f"  Found {len(fetched)} URLs")
                all_urls.extend(fetched)
            filtered = filter_urls(all_urls)
            st.write(f"**{len(filtered)} unique URLs** after filtering")
            if len(filtered) > max_pages:
                filtered = filtered[:max_pages]
                st.write(f"Trimmed to {max_pages}")
            if len(filtered) < 2:
                st.error("Need at least 2 URLs.")
                st.stop()
            status.update(label=f"✓ {len(filtered)} URLs ready", state="complete")

        with st.status("Analyzing page content...", expanded=True) as status:
            contents, valid = [], []
            pb = st.progress(0)
            for i, url in enumerate(filtered):
                t, h, c = fetch_page_content(url)
                if c:
                    contents.append(f"{t} {h} {c}")
                    valid.append(url)
                pb.progress((i + 1) / len(filtered))
            st.write(f"**{len(valid)}** pages with content (of {len(filtered)} attempted)")
            if len(valid) < 2:
                st.error("Not enough pages with extractable content.")
                st.stop()
            status.update(label=f"✓ {len(valid)} pages analyzed", state="complete")

        with st.status("Building content clusters...", expanded=True) as status:
            labels, plan = cluster_and_link(contents, valid)
            n_clusters = len(set(labels))
            n_links = sum(len(v) for v in plan.values())
            st.write(f"**{n_clusters}** clusters discovered, **{n_links}** link suggestions")
            status.update(label=f"✓ {n_clusters} clusters, {n_links} links", state="complete")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", len(valid))
        c2.metric("Clusters", n_clusters)
        c3.metric("Link Suggestions", n_links)

        st.markdown("")
        st.markdown('<div class="section-label">Network Visualization</div>', unsafe_allow_html=True)
        fig = build_network_graph(plan, labels, valid)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("")
        st.markdown('<div class="section-label">Cluster Breakdown</div>', unsafe_allow_html=True)
        cdf = pd.DataFrame({"URL": valid, "Cluster": labels})
        for label in sorted(set(labels)):
            cl = cdf[cdf["Cluster"] == label]["URL"].tolist()
            with st.expander(f"Cluster {label}  ·  {len(cl)} pages"):
                for u in cl:
                    st.markdown(f"- `{u}`")

        st.markdown("")
        excel = save_to_excel_bytes(valid, labels, plan)
        st.download_button("📥  Download Linking Plan (Excel)", excel, "internal_linking_plan.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
