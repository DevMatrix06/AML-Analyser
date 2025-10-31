import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import matplotlib.colors as mcolors
import torch
from torch_geometric.data import Data
import codecs  # For UTF-8 fix
from pathlib import Path

# =====================================================
# --- CONFIGURATION (AUTOMATIC REPO-PATH DETECTION) ---
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to AML-Transaction-Graph-Analyser/
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

PROCESSED_DATA_PATH = DATA_DIR / "graph_data.pt"
SCORES_PATH = OUTPUT_DIR / "suspicion_scores.csv"

# Create missing folders if not present
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_AUC = 0.5649
FRAUD_RATIO = 0.0013

# =====================================================
# --- LOAD GRAPH DATA ---
# =====================================================
@st.cache_data
def load_graph_data(path):
    """Loads the processed graph data object for dimension extraction."""
    if not path.exists():
        st.error(f"❌ Processed graph data not found at {path}. Cannot display metrics.")
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:
        st.error(f"Failed to load graph data for metrics. Error: {e}")
        return None


data = load_graph_data(PROCESSED_DATA_PATH)
total_unique_accounts = data.x.size(0) if data is not None else 0

# =====================================================
# --- STREAMLIT SETUP ---
# =====================================================
st.set_page_config(layout="wide", page_title="AML Transaction Graph Analyzer")

# =====================================================
# --- LOAD SUSPICION SCORES ---
# =====================================================
@st.cache_data
def load_scores_data(path):
    """Loads the processed suspicion scores."""
    if not path.exists():
        st.error(f"❌ Output scores file not found at {path}. Please run 'python src/predict.py' first.")
        return None
    df = pd.read_csv(path)
    df['SENDER_ACCOUNT_ID'] = df['SENDER_ACCOUNT_ID'].astype(str)
    df['RECEIVER_ACCOUNT_ID'] = df['RECEIVER_ACCOUNT_ID'].astype(str)
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)
    return df

# =====================================================
# --- DISPLAY NETWORK GRAPH ---
# =====================================================
def display_network_graph(df_filtered, graph_html_filename="network_graph.html"):
    """Builds, saves, and renders a pyvis graph from a dataframe."""
    if df_filtered.empty:
        st.info("No transactions to display for this view.")
        return

    G = nx.from_pandas_edgelist(
        df_filtered,
        source='SENDER_ACCOUNT_ID',
        target='RECEIVER_ACCOUNT_ID',
        edge_attr=['TX_AMOUNT', 'SUSPICION_SCORE', 'TX_TYPE', 'IS_FRAUD'],
        create_using=nx.DiGraph()
    )

    net = Network(height='600px', width='100%', directed=True, notebook=False, cdn_resources='remote')
    net.set_options("""
        var options = {
          "physics": {
            "barnesHH": {
              "centralGravity": 0.2,
              "springLength": 100,
              "springConstant": 0.05,
              "damping": 0.9
            },
            "minVelocity": 0.75
          }
        }
    """)

    node_scores = {}
    for _, row in df_filtered.iterrows():
        node_scores[row['SENDER_ACCOUNT_ID']] = max(node_scores.get(row['SENDER_ACCOUNT_ID'], 0), row['SUSPICION_SCORE'])
        node_scores[row['RECEIVER_ACCOUNT_ID']] = max(node_scores.get(row['RECEIVER_ACCOUNT_ID'], 0), row['SUSPICION_SCORE'])

    fraud_accounts = set(df_filtered[df_filtered['IS_FRAUD'] == 1]['SENDER_ACCOUNT_ID']).union(
        set(df_filtered[df_filtered['IS_FRAUD'] == 1]['RECEIVER_ACCOUNT_ID'])
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("suspicion_cmap", ["blue", "red"])

    for node in G.nodes():
        score = node_scores.get(node, 0)
        color_val = min(score * 1.5, 1.0)
        hex_color = mcolors.to_hex(cmap(color_val))
        border_width = 3 if node in fraud_accounts else 1
        title_html = f"**Account ID:** {node}<br>**Max Edge Score:** {score:.4f}<br>**Known Fraud:** {'Yes' if node in fraud_accounts else 'No'}"
        net.add_node(
            n_id=str(node),
            label=str(node),
            title=title_html,
            color={'border': '#000000' if node not in fraud_accounts else '#FF0000', 'background': hex_color},
            borderWidth=border_width
        )

    for source, target, data in G.edges(data=True):
        score = data['SUSPICION_SCORE']
        line_thickness = max(0.5, score * 10)
        line_color = mcolors.to_hex(cmap(min(score * 1.5, 1.0)))
        title_html = (
            f"**Score:** {score:.4f}<br>"
            f"**Amount:** {data['TX_AMOUNT']:,.2f}<br>"
            f"**Type:** {data['TX_TYPE']}<br>"
            f"**Known Fraud:** {'YES' if data['IS_FRAUD'] == 1 else 'No'}"
        )
        net.add_edge(source=str(source), to=str(target), title=title_html, value=line_thickness, color=line_color)

    html_file_path = OUTPUT_DIR / graph_html_filename

    html_str = net.generate_html()
    with codecs.open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=650, scrolling=True)
