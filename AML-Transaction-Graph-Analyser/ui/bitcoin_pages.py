import streamlit as st
import pandas as pd
import os
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import codecs
import matplotlib.colors as mcolors

# --- Configuration ---
PREDICTIONS_PATH = 'outputs/bitcoin_predictions.csv'
EDGES_PATH = 'data/raw/elliptic_txs_edgelist.csv' # Need this for graph connections
# Constants based on the Elliptic Dataset
TOTAL_TXS = 203769
TOTAL_LABELED = 4545 + 42019
TEST_F1_SCORE = 0.8497
RISK_THRESHOLD = 0.95 # Threshold for filtering high-risk addresses

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide", page_title="Elliptic Bitcoin Analyzer")

# --- Function to Load Data ---
@st.cache_data
def load_bitcoin_predictions(path):
    """Loads the prediction scores and ensures data types are ready."""
    if not os.path.exists(path):
        st.error(f"Error: Prediction file not found at {path}. Please run 'python src/predict_bitcoin.py' first.")
        return None
    df = pd.read_csv(path)
    df['txId'] = df['txId'].astype(str)
    # Ensure scores are clipped just in case of any numerical instability during prediction
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)
    return df

@st.cache_data
def load_edge_data(path):
    """Loads the edge list for graph rendering."""
    if not os.path.exists(path):
        st.error(f"Error: Edge list not found at {path}. Cannot render graph.")
        return None
    df_edges = pd.read_csv(path)
    df_edges.rename(columns={'txId1': 'source', 'txId2': 'target'}, inplace=True)
    df_edges['source'] = df_edges['source'].astype(str)
    df_edges['target'] = df_edges['target'].astype(str)
    return df_edges

# Load primary data globally
df_pred = load_bitcoin_predictions(PREDICTIONS_PATH)
df_edges = load_edge_data(EDGES_PATH)

# --- Function to Render Leaderboards ---
def render_leaderboards(df_pred):
    """
    Renders the analytical leaderboards focusing on high-risk, UNLABELED addresses.
    """
    st.subheader("🥇 Top Predicted High-Risk Addresses (Unseen)")

    # Filter only UNLABELED addresses with high prediction score
    df_risky = df_pred[
        (df_pred['SUSPICION_SCORE'] >= RISK_THRESHOLD) &
        (df_pred['class'] == 'Unlabeled')
    ].copy()

    if df_risky.empty:
        st.info(f"No unlabeled addresses meet the high-risk threshold of {RISK_THRESHOLD:.2f}.")
        return

    # --- 1. Top 10 Risky Accounts (Highest Suspicion Score) ---
    st.markdown("#### Top 10 High-Risk Unlabeled Addresses (by Illicit Score)")
    score_leaderboard = df_risky[['txId', 'SUSPICION_SCORE']].head(10).reset_index(drop=True)
    score_leaderboard.index += 1
    score_leaderboard.columns = ['Address ID', 'Illicit Score']
    st.dataframe(score_leaderboard.style.format({'Illicit Score': '{:.6f}'}), use_container_width=True)


    # --- 2. Top 10 Risky TimeSteps (Highest Count of Illicit Addresses) ---
    st.markdown("#### Top 10 Risky TimeSteps (Highest Count of Illicit Predictions)")
    hub_leaderboard = df_risky.groupby('TimeStep').size().sort_values(ascending=False).head(10).reset_index()
    hub_leaderboard.columns = ['TimeStep', 'Count of Risky Addresses']
    st.dataframe(hub_leaderboard, use_container_width=True, hide_index=True)


# --- New Graph Drawing Function for Egonet ---
def display_network_graph(df_account_edges, df_account_nodes, graph_html_filename="egonet_graph.html"):
    """Builds, saves, and renders a PyVis graph for a small, localized network."""

    if df_account_edges.empty:
        st.info("No connections to display for this address.")
        return

    # 1. Initialize Graph
    G = nx.from_pandas_edgelist(
        df_account_edges, source='source', target='target', create_using=nx.DiGraph()
    )
    # Using 'remote' resources makes the HTML file much smaller and faster to load
    net = Network(height='500px', width='100%', directed=True, cdn_resources='remote')

    # Define color map for styling
    cmap = mcolors.LinearSegmentedColormap.from_list("suspicion_cmap", ["#112A66", "#FF0000"]) # Blue to Red

    # 2. Add Nodes with Styling
    for node_id in G.nodes():
        # Handle cases where a neighbor might not have prediction data (if filtering occurred)
        node_data_list = df_account_nodes[df_account_nodes['txId'] == node_id]
        if node_data_list.empty:
            continue # Skip nodes without data
        node_data = node_data_list.iloc[0]
        score = node_data['SUSPICION_SCORE']
        is_illicit = node_data['class'] == 'Illicit'

        color_val = min(score * 1.5, 1.0)
        hex_color = mcolors.to_hex(cmap(color_val))

        title_html = (
            f"**Address ID:** {node_id}<br>"
            f"**Score:** {score:.6f}<br>"
            f"**Truth:** {node_data['class']}"
        )

        net.add_node(
            n_id=node_id,
            label=node_id,
            title=title_html,
            size=15 + score * 20, # Size based on risk
            color={'border': '#FF0000' if is_illicit else '#000000', 'background': hex_color},
            borderWidth=3 if is_illicit else 1
        )

    # 3. Add Edges
    for source, target in G.edges():
        net.add_edge(source, target, color='#999999', width=1.5)

    # 4. Render
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    html_file_path = os.path.join('outputs', graph_html_filename)
    html_content = net.generate_html()

    # Write and read with encoding fix
    with codecs.open(html_file_path, "w", encoding="utf-8") as out:
        out.write(html_content)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        final_html_content = f.read()

    st.components.v1.html(final_html_content, height=550, scrolling=True)

# ----------------------------------------------------------------------
# --- NEW PATTERN ANALYSIS FUNCTION FOR BITCOIN (with score check) ---
# ----------------------------------------------------------------------
def analyze_bitcoin_patterns(account_id, df_edges, score):
    """Heuristically determines AML patterns based on transaction node degree."""

    # --- NEW CHECK: Only analyze patterns if GNN score is high ---
    if score < 0.5:
        return "Likely Licit (Low GNN Score)"
    # --- END NEW CHECK ---

    # Calculate In-Degree and Out-Degree for the transaction node
    in_degree = df_edges[df_edges['target'] == account_id].shape[0]
    out_degree = df_edges[df_edges['source'] == account_id].shape[0]

    patterns = []

    # Define thresholds (simpler heuristics based on connectivity)
    HIGH_DEGREE_THRESHOLD = 5
    BALANCE_THRESHOLD = 2 # Difference between in/out degree to detect imbalance

    # 1. Mule Detection (Consolidation or Dispersal)
    is_mule = False # Flag to check if Mule pattern is detected
    if abs(in_degree - out_degree) > BALANCE_THRESHOLD:
        is_mule = True
        if in_degree > out_degree:
            patterns.append("Mule-like (Consolidation Transaction)")
            # 2. Smurfing Detection (High in-degree suggests many inputs)
            if in_degree > HIGH_DEGREE_THRESHOLD:
                patterns.append("Smurfing Inputs Suspected")
        else:
            patterns.append("Mule-like (Dispersal Transaction)")

    # 3. Layering Detection (Balanced throughput)
    # Trigger if it has decent traffic AND degrees are relatively balanced
    # AND it was NOT flagged as a Mule
    if not is_mule and in_degree >= 1 and out_degree >= 1:
        patterns.append("Layering/Pass-Through Transaction")

    if not patterns:
        # If score is high but no structural pattern matches
        return "Suspicious Activity Detected (GNN Score High)"

    return ", ".join(patterns)

# --- PAGE 1: Global Dashboard ---
def draw_global_dashboard(df_pred):
    st.title("₿ Elliptic Bitcoin Transaction Analyzer")
    st.markdown("### GraphSAGE Node Classification Results")

    # Metrics Panel
    PREDICTED_ILLICIT = df_pred[df_pred['SUSPICION_SCORE'] >= RISK_THRESHOLD].shape[0]
    PREDICTED_UNLABELED = df_pred[(df_pred['SUSPICION_SCORE'] >= RISK_THRESHOLD) & (df_pred['class'] == 'Unlabeled')].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Addresses Analyzed", f"{TOTAL_TXS:,}")
    col2.metric("Total Labeled Addresses", f"{TOTAL_LABELED:,}")
    col3.metric("Model F1 Score (Test)", f"{TEST_F1_SCORE:.4f}")
    col4.metric(f"New Predictions (Risk $\geq$ {RISK_THRESHOLD:.2f})", f"{PREDICTED_UNLABELED:,}")

    st.markdown("---")

    # --- HISTOGRAM ---
    st.subheader("Distribution of Predicted Illicit Scores")
    st.markdown("This histogram shows the confidence distribution of the model's predictions.")

    fig_hist = px.histogram(
        df_pred,
        x='SUSPICION_SCORE',
        nbins=20,
        title='Illicit Probability Score Distribution Across All Addresses',
        template='plotly_dark',
        range_x=[0,1] # Explicitly set range
    )
    fig_hist.update_layout(
        xaxis_title="Illicit Score (0.0 to 1.0)",
        yaxis_title="Number of Addresses",
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0.0,
            dtick = 0.05
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Leaderboard Section
    render_leaderboards(df_pred)
    st.markdown("---")

    # Time-Series Risk Analysis
    st.subheader("Time-Series Risk Analysis (Average Suspicion Score per TimeStep)")

    df_time_risk = df_pred.groupby('TimeStep')['SUSPICION_SCORE'].mean().reset_index()
    df_time_risk.columns = ['TimeStep', 'Average Suspicion Score']

    fig_line = px.line(
        df_time_risk,
        x='TimeStep',
        y='Average Suspicion Score',
        title='Average Illicit Probability by TimeStep',
        template='plotly_dark'
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # --- GLOBAL NETWORK MAP ---
    st.subheader("Interactive Suspicious Network Map (Global Top Risk)")

    max_score = df_pred['SUSPICION_SCORE'].max()
    FIXED_DEFAULT_THRESHOLD = 0.999

    suspicion_threshold = st.slider(
        "Filter Nodes by Suspicion Score Threshold:",
        min_value=float(df_pred['SUSPICION_SCORE'].min()),
        max_value=float(max_score),
        value=float(FIXED_DEFAULT_THRESHOLD),
        step=0.00001,
        format="%.5f",
        key='global_slider'
    )

    df_filtered_nodes = df_pred[df_pred['SUSPICION_SCORE'] >= suspicion_threshold].copy()
    nodes_to_keep = df_filtered_nodes['txId'].tolist()

    df_filtered_edges = df_edges[
        (df_edges['source'].isin(nodes_to_keep)) &
        (df_edges['target'].isin(nodes_to_keep))
    ].copy()

    st.write(f"Showing connections between **{len(nodes_to_keep):,}** high-risk addresses, resulting in **{len(df_filtered_edges):,}** connections.")

    display_network_graph(df_filtered_edges, df_filtered_nodes, "global_network_graph_filtered.html")


# --- PAGE 2: Account Deep Dive ---
def draw_bitcoin_deep_dive(): # Renamed function
    st.title("🔎 Investigate a Single Bitcoin Transaction (Address)")
    st.markdown("Analyze the local network (Egonet) and risk metrics for any address.")

    search_id = st.text_input("Enter Transaction ID (e.g., from the table above):")

    if search_id:
        account_id = search_id.strip()

        # 1. Get Node Data
        node_data = df_pred[df_pred['txId'] == account_id]

        if node_data.empty:
            st.error("Transaction ID not found in prediction data.")
            return

        score = node_data['SUSPICION_SCORE'].iloc[0]
        ground_truth = node_data['class'].iloc[0]

        st.markdown(f"### Deep Dive Analysis for Transaction ID: `{account_id}`")

        # 2. Get Edge Data (Egonet)
        df_account_edges = df_edges[
            (df_edges['source'] == account_id) | (df_edges['target'] == account_id)
        ].copy()

        neighbors = set(df_account_edges['source']).union(set(df_account_edges['target']))
        df_account_nodes = df_pred[df_pred['txId'].isin(neighbors)].copy()


        # 3. Display Metrics and PATTERN ANALYSIS
        st.markdown("#### Key Risk Metrics and Pattern Classification")

        col1, col2, col3 = st.columns(3)
        col1.metric("Illicit Suspicion Score", f"{score:.6f}",
                    delta_color="off", delta=f"{'Ground Truth: '+ground_truth}")
        col2.metric("Total Neighbors (Transactions)", f"{len(neighbors) - 1:,}")
        col3.metric("Number of Direct Connections", f"{len(df_account_edges):,}")

        # --- UPDATE PATTERN ANALYSIS CALL for BITCOIN (Pass the score) ---
        pattern_result = analyze_bitcoin_patterns(account_id, df_edges, score)
        st.markdown(f"**Predicted Structural Patterns:** :red[{pattern_result}]")
        st.markdown("---")

        # 4. Display Egonet Graph
        st.markdown("#### Local Transaction Graph (Egonet)")
        display_network_graph(df_account_edges, df_account_nodes, "egonet_graph_filtered.html")

        st.markdown("---")

        # 5. Display Transaction History (Corrected Merge Logic)
        st.markdown("#### Transaction Details (Egonet)")

        # Merge source node info
        df_table = df_account_edges.merge(
            df_pred[['txId', 'class', 'SUSPICION_SCORE']].rename(columns={
                'txId': 'txId_source', 'class': 'class_sender', 'SUSPICION_SCORE': 'SUSPICION_SCORE_sender'
            }),
            left_on='source', right_on='txId_source', how='left'
        ).drop(columns=['txId_source'])

        # Merge target node info
        df_table = df_table.merge(
            df_pred[['txId', 'class', 'SUSPICION_SCORE']].rename(columns={
                'txId': 'txId_target', 'class': 'class_receiver', 'SUSPICION_SCORE': 'SUSPICION_SCORE_receiver'
            }),
            left_on='target', right_on='txId_target', how='left'
        ).drop(columns=['txId_target'])

        # Final display table construction
        df_display = df_table[['source', 'target', 'SUSPICION_SCORE_sender', 'class_sender', 'SUSPICION_SCORE_receiver', 'class_receiver']].copy()
        df_display.columns = ['Sender ID', 'Receiver ID', 'Score (Sender)', 'Class (Sender)', 'Score (Receiver)', 'Class (Receiver)']

        st.dataframe(
            df_display
            .style.format({'Score (Sender)': '{:.6f}', 'Score (Receiver)': '{:.6f}'})
            , use_container_width=True
        )


# --- Main App Logic (Router) ---
def draw_bitcoin_router(): # New function name
    st.sidebar.title("₿ Elliptic Pages")
    app_mode = st.sidebar.radio(
        "Choose view:",
        ["Global Dashboard", "Address Deep Dive"] # Renamed to Address Deep Dive
    )

    if df_pred is not None and df_edges is not None:
        if app_mode == "Global Dashboard":
            draw_global_dashboard(df_pred)
        elif app_mode == "Address Deep Dive":
            draw_bitcoin_deep_dive() # Call the correct deep dive function
    else:
        st.error("Failed to load necessary data files. Please ensure you have run data_preprocessing_bitcoin.py and predict_bitcoin.py.")

