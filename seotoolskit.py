import streamlit as st
import asyncio
import aiohttp
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import json
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

# Set the page configuration as the first Streamlit command
st.set_page_config(
    page_title="SEO Toolkit - Bulk API Indexing & Internal Linking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    return '{"error": {"code": 500, "message": "Failed to index URL"}}'

async def bulk_index(http, urls):
    async with aiohttp.ClientSession() as session:
        tasks = [send_url(session, http, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

# Upload JSON Key Files Section
st.header("Upload JSON Key Files")
st.markdown("""
<p style='font-size:14px;'>
You can upload multiple JSON files. Each JSON file can send a maximum of 200 URL requests.
</p>
""", unsafe_allow_html=True)

# Drag and drop feature for JSON files
uploaded_files = st.file_uploader("Drag and drop your JSON files here", accept_multiple_files=True, type=['json'])

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded successfully!")

# Action button to start processing the uploaded files
if st.button("Start Upload"):
    # Here you would typically process the uploaded JSON files
    st.info("Processing files...")

# Bulk API Indexing Section
st.header("Bulk API Indexing")
url_input = st.text_area("Enter URLs to index (one per line)", height=200)
if st.button("Start Indexing"):
    urls = url_input.splitlines()
    # Asynchronous URL indexing logic
    st.info(f"Started indexing {len(urls)} URLs...")

# Internal Linking Section
st.header("Internal Linking Using Clusters")
sitemap_urls = st.text_area("Enter Sitemap URLs (Refrain to add main sitemap if there are individual sitemaps for post, pages, products, etc.)").split('\n')
if st.button("Generate Internal Linking Plan"):
    # Here you would typically handle internal linking logic
    st.info("Generating internal linking plan...")

    # Placeholder logic for generating an Excel file
    excel_file = "internal_linking_plan.xlsx"  # This would be generated dynamically
    st.markdown(download_link(open(excel_file, 'rb').read(), 'internal_linking_plan.xlsx', 'Download Excel file'), unsafe_allow_html=True)
