import streamlit as st
import asyncio
import aiohttp
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import json
from tqdm import tqdm
from aiohttp.client_exceptions import ServerDisconnectedError
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import plotly.graph_objs as go
import logging
import base64
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants for Indexing
SCOPES = ["https://www.googleapis.com/auth/indexing"]
ENDPOINT = "https://indexing.googleapis.com/v3/urlNotifications:publish"
URLS_PER_ACCOUNT = 200

# Functions for URL Indexing
async def send_url(session, http, url):
    content = {'url': url.strip(), 'type': "URL_UPDATED"}
    for _ in range(3):  # Retry up to 3 times
        try:
            async with session.post(ENDPOINT, json=content, headers={"Authorization": f"Bearer {http}"}, ssl=False) as response:
                return await response.text()
        except ServerDisconnectedError:
            await asyncio.sleep(2)  # Wait for 2 seconds before retrying
            continue
    return '{"error": {"code": 500, "message": "Server Disconnected after multiple retries"}}'

async def indexURL(http, urls):
    successful_urls = 0
    error_429_count = 0
    other_errors_count = 0
    tasks = []

    async with aiohttp.ClientSession() as session:
        for url in tqdm(urls, desc="Processing URLs", unit="url"):
            tasks.append(send_url(session, http, url))

        results = await asyncio.gather(*tasks)

        for result in results:
            data = json.loads(result)
            if "error" in data:
                if data["error"]["code"] == 429:
                    error_429_count += 1
                else:
                    other_errors_count += 1
            else:
                successful_urls += 1

    st.write(f"\nTotal URLs Tried: {len(urls)}")
    st.write(f"Successful URLs: {successful_urls}")
    st.write(f"URLs with Error 429: {error_429_count}")

def setup_http_client(json_key_file):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_key_file, scopes=SCOPES)
    token = credentials.get_access_token().access_token
    return token

# Functions for Internal Linking
def get_urls_from_sitemap(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    urls = [item.text for item in soup.find_all('loc')]
    return urls

def filter_urls(urls):
    filtered_urls = [url for url in urls if '/page/' not in url and 'category' not in url]
    return filtered_urls

def fetch_page_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            return None, None, None
        html_content = r.text
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None, None, None

    soup = BeautifulSoup(html_content, 'lxml')
    if soup.find('meta', attrs={'name': 'robots', 'content': 'noindex'}):
        return None, None, None
    title = soup.title.string if soup.title else ''
    heading = soup.find(['h1', 'h2', 'h3'])
    heading = heading.text.strip() if heading else ''
    content = soup.find('main') or soup.find('article') or soup.find('body')
    if content:
        for tag in content(['header', 'nav', 'footer', 'aside']):
            tag.decompose()
        text_content = ' '.join(content.stripped_strings)
    else:
        text_content = ''
    return title, heading, text_content

def calculate_similarity(contents):
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(contents)
    return tfidf_matrix, vectorizer

def cluster_urls(tfidf_matrix):
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    clustering_model.fit(tfidf_matrix.toarray())
    return clustering_model.labels_

def create_internal_linking_plan(urls, labels):
    df = pd.DataFrame({'url': urls, 'label': labels})
    linking_plan = {}

    for label in df['label'].unique():
        cluster_urls = df[df['label'] == label]['url']
        for url in cluster_urls:
            linking_plan[url] = [link for link in cluster_urls if link != url]

    return linking_plan

def save_to_excel(urls, labels, linking_plan, output_file):
    clusters_df = pd.DataFrame({'URL': urls, 'Cluster Label': labels})
    
    linking_plan_df = pd.DataFrame([(source, target) for source, targets in linking_plan.items() for target in targets],
                                   columns=['Source URL', 'Target URL'])
    
    cluster_summary = pd.DataFrame({
        'Cluster Label': [label for label in set(labels)],
        'URLs': [', '.join(clusters_df[clusters_df['Cluster Label'] == label]['URL']) for label in set(labels)]
    })

    with pd.ExcelWriter(output_file) as writer:
        clusters_df.to_excel(writer, sheet_name='URL Clusters', index=False)
        linking_plan_df.to_excel(writer, sheet_name='Internal Linking Plan', index=False)
        cluster_summary.to_excel(writer, sheet_name='Cluster Summary', index=False)

    return output_file

def visualize_internal_linking(linking_plan):
    G = nx.Graph()
    for source, targets in linking_plan.items():
        for target in targets:
            G.add_edge(source, target)

    pos = nx.spring_layout(G, k=0.3)

    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            )
        )

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        text=[node for node in G.nodes()],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Cluster',
                xanchor='left',
                titleside='right'
            ),
        ),
        textposition="bottom center"
    )

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='<br>Internal Linking Graph',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Network graph made by Plotly",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    return fig

def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Streamlit Interface
st.set_page_config(
    page_title="SEO Toolkit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    h1, h2, h3, h4, h5, h6 {color: #003566;}
    .sidebar .sidebar-content {background: #003566; color: white;}
    .css-1d391kg {color: white;}
    .css-1offfwp.e1fqkh3o4 {color: #003566;}
    .stButton>button {background-color: #003566; color: white; border-radius: 10px; padding: 10px;}
    .stButton>button:hover {background-color: #00509e; color: white;}
    .stTextInput input {border: 1px solid #003566; border-radius: 5px; padding: 5px;}
    .stFileUploader div {color: #003566;}
    .stFileUploader label {font-weight: bold;}
    .css-1e5imcs {color: #003566;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 SEO Toolkit")

st.sidebar.title("App & Tools built by")
st.sidebar.markdown("[Neeraj Kumar](https://in.linkedin.com/in/neeraj-kumar-seo)")
st.sidebar.markdown("[Abhishek Kaushal](https://www.linkedin.com/in/abhishek-kaushal-85a15818a/)")

# Tab Selection
tab = st.sidebar.selectbox("Choose a tool", ["Bulk API Indexing", "Internal Linking Using Clusters"])

if tab == "Bulk API Indexing":
    st.header("Bulk API Indexing")
    
    uploaded_files = st.file_uploader("Upload JSON key files (max 5)", type="json", accept_multiple_files=True, help="Upload up to 5 JSON files.")
    
    if uploaded_files:
        num_files = len(uploaded_files)
        st.write(f"{num_files} JSON files uploaded. You can index up to {num_files * URLS_PER_ACCOUNT} URLs.")
        
        url_input = st.text_area("Enter URLs to index (one per line)", height=200)
        urls = [url.strip() for url in url_input.split("\n") if url.strip()]

        if len(urls) > num_files * URLS_PER_ACCOUNT:
            st.warning(f"You can only index up to {num_files * URLS_PER_ACCOUNT} URLs. Please reduce the number of URLs.")
        else:
            if st.button("Start Indexing"):
                all_urls = urls[:num_files * URLS_PER_ACCOUNT]
                
                for i, uploaded_file in enumerate(uploaded_files):
                    start_index = i * URLS_PER_ACCOUNT
                    end_index = start_index + URLS_PER_ACCOUNT
                    urls_for_account = all_urls[start_index:end_index]
                    
                    with open(f"account{i+1}.json", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    json_key_file = f"account{i+1}.json"
                    http = setup_http_client(json_key_file)
                    asyncio.run(indexURL(http, urls_for_account))
                    
                    os.remove(json_key_file)

if tab == "Internal Linking Using Clusters":
    st.header("Internal Linking Using Clusters")

    sitemap_urls = st.text_area("Enter Sitemap URLs (one per line)").split('\n')

    if st.button("Generate Internal Linking Plan"):
        all_filtered_urls = []
        for sitemap_url in tqdm(sitemap_urls, desc="Fetching URLs from sitemaps"):
            urls = get_urls_from_sitemap(sitemap_url)
            filtered_urls = filter_urls(urls)
            all_filtered_urls.extend(filtered_urls)

        page_contents = []
        valid_urls = []
        for url in tqdm(all_filtered_urls, desc="Fetching page contents"):
            title, heading, content = fetch_page_content(url)
            if content:
                page_contents.append(f"{title} {heading} {content}")
                valid_urls.append(url)

        tfidf_matrix, vectorizer = calculate_similarity(page_contents)
        labels = cluster_urls(tfidf_matrix)

        linking_plan = create_internal_linking_plan(valid_urls, labels)

        excel_file = save_to_excel(valid_urls, labels, linking_plan, 'internal_linking_plan.xlsx')

        st.success("Internal linking plan generated successfully!")

        st.write("Download the internal linking plan:")
        st.markdown(download_link(open(excel_file, 'rb').read(), 'internal_linking_plan.xlsx', 'Download Excel file'), unsafe_allow_html=True)

        st.write("Visualization of the internal linking plan:")
        fig = visualize_internal_linking(linking_plan)
        st.plotly_chart(fig)
