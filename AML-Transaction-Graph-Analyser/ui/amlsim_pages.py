import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import matplotlib.colors as mcolors
import torch
from torch_geometric.data import Data
import codecs

# --- Configuration (Using RELATIVE paths for reliability) ---
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'
SCORES_PATH = 'outputs/suspicion_scores.csv'
BEST_MODEL_AUC = 0.5649
FRAUD_RATIO = 0.0013

# --- Function to Load Graph Data for Global Metrics ---
@st.cache_data
def load_graph_data(path):
    """Loads the processed graph data object for dimension extraction."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception:
        return None

# --- Function to Load Suspicion Scores CSV ---
@st.cache_data
def load_scores_data(path):
    """Loads the processed suspicion scores."""
    if not os.path.exists(path):
        st.error(f"Error: Output scores file not found at {path}. Please run 'python src/predict.py' first.")
        return None
    df = pd.read_csv(path)
    # Ensure IDs are strings for consistent NetworkX/PyVis handling
    df['SENDER_ACCOUNT_ID'] = df['SENDER_ACCOUNT_ID'].astype(str)
    df['RECEIVER_ACCOUNT_ID'] = df['RECEIVER_ACCOUNT_ID'].astype(str)
    df['SUSPICION_SCORE'] = df['SUSPICION_SCORE'].clip(0, 1)
    return df

# --- Function to Render Leaderboards ---
def render_leaderboards(df_scores):
    st.subheader("📊 Top 10 High-Risk Account Leaderboards")

    RISK_THRESHOLD = 0.5075
    df_risky = df_scores[df_scores['SUSPICION_SCORE'] >= RISK_THRESHOLD].copy()

    if df_risky.empty:
        st.info(f"No transactions meet the high-risk threshold of {RISK_THRESHOLD:.4f} for leaderboard calculation.")
        return

    # --- Calculation 1: Top 10 Risky Accounts (Highest Average Score) ---
    st.markdown("#### 🥇 Top 10 Risky Accounts (by Average Suspicion Score)")
    senders = df_risky[['SENDER_ACCOUNT_ID', 'SUSPICION_SCORE']].rename(columns={'SENDER_ACCOUNT_ID': 'ACCOUNT_ID'})
    receivers = df_risky[['RECEIVER_ACCOUNT_ID', 'SUSPICION_SCORE']].rename(columns={'RECEIVER_ACCOUNT_ID': 'ACCOUNT_ID'})
    all_risky_accounts = pd.concat([senders, receivers])
    avg_score_leaderboard = all_risky_accounts.groupby('ACCOUNT_ID')['SUSPICION_SCORE'].mean().sort_values(ascending=False).head(10).reset_index()
    avg_score_leaderboard.columns = ['Account ID', 'Average Suspicion Score']
    avg_score_leaderboard['Average Suspicion Score'] = avg_score_leaderboard['Average Suspicion Score'].map('{:.5f}'.format)
    st.dataframe(avg_score_leaderboard, use_container_width=True, hide_index=True)


    # --- Calculation 2: Top 10 Risky Hubs (Most High-Suspicion Transactions Received) ---
    st.markdown("#### 🥈 Top 10 Risky Hubs (by Count of Suspicious Incoming Transactions)")
    hub_leaderboard = df_risky.groupby('RECEIVER_ACCOUNT_ID').size().sort_values(ascending=False).head(10).reset_index()
    hub_leaderboard.columns = ['Account ID', 'Incoming Suspicious Count']
    st.dataframe(hub_leaderboard, use_container_width=True, hide_index=True)


    # --- Calculation 3: Top 10 Risky Volume Targets (Highest Total $ Volume in Risky Transactions) ---
    st.markdown("#### 🥉 Top 10 Risky Volume Targets (by Total $ Received)")
    volume_leaderboard = df_risky.groupby('RECEIVER_ACCOUNT_ID')['TX_AMOUNT'].sum().sort_values(ascending=False).head(10).reset_index()
    volume_leaderboard.columns = ['Account ID', 'Total Suspicious $ Received']
    volume_leaderboard['Total Suspicious $ Received'] = volume_leaderboard['Total Suspicious $ Received'].map('${:,.2f}'.format)
    st.dataframe(volume_leaderboard, use_container_width=True, hide_index=True)


# --- Reusable Graph Drawing Function (Kept intact) ---
def display_network_graph(df_filtered, graph_html_filename):
    """
    Builds, saves, and renders a pyvis graph from a dataframe.
    """
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
            "barnesHut": {
              "centralGravity": 0.3,
              "springLength": 100,
              "springConstant": 0.04,
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
        title_html = f"Account ID: {node}<br>Max Edge Score: {score:.4f}<br>Known Fraud: {'Yes' if node in fraud_accounts else 'No'}"
        net.add_node(
            n_id=str(node),
            label=str(node),
            title=title_html,
            color={'border': '#FF0000' if node in fraud_accounts else '#000000',
                   'background': hex_color},
            borderWidth=border_width
        )

    for source, target, data in G.edges(data=True):
        score = data['SUSPICION_SCORE']
        line_thickness = max(0.5, score * 10)
        line_color = mcolors.to_hex(cmap(min(score * 1.5, 1.0)))
        title_html = (
            f"Score: {score:.4f}<br>"
            f"Amount: {data['TX_AMOUNT']:,.2f}<br>"
            f"Type: {data['TX_TYPE']}<br>"
            f"Known Fraud: {'YES' if data['IS_FRAUD'] == 1 else 'No'}"
        )
        net.add_edge(source=str(source), to=str(target), title=title_html, value=line_thickness, color=line_color)

    # --- RENDER LOGIC: Use relative outputs path ---
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    html_file_path = os.path.join('outputs', graph_html_filename)

    html_str = net.generate_html()
    with codecs.open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=650, scrolling=True)

# ----------------------------------------------------------------------
# --- PATTERN ANALYSIS FUNCTION (FINAL HEURISTICS WITH SCORE CHECK) ---
# ----------------------------------------------------------------------

def analyze_aml_patterns(df_account, net_flow, avg_score):
    """Heuristically determines the most likely AML pattern based on local graph metrics."""

    # --- NEW CHECK: Only analyze patterns if the GNN score is high ---
    if avg_score < 0.5:
        return "Likely Safe (Low GNN Score)"
    # --- END NEW CHECK ---

    # Safely determine the central account ID (needed for degree calculation)
    all_involved_ids = set(df_account['SENDER_ACCOUNT_ID']) | set(df_account['RECEIVER_ACCOUNT_ID'])
    central_account_id = pd.concat([df_account['SENDER_ACCOUNT_ID'], df_account['RECEIVER_ACCOUNT_ID']]).mode()[0] if not df_account.empty else None

    if central_account_id is None:
        return "Not enough data for analysis."

    # Calculate metrics for the central node
    in_degree = df_account[df_account['RECEIVER_ACCOUNT_ID'] == central_account_id].shape[0]
    out_degree = df_account[df_account['SENDER_ACCOUNT_ID'] == central_account_id].shape[0]

    total_sent = df_account[df_account['SENDER_ACCOUNT_ID'] == central_account_id]['TX_AMOUNT'].sum()
    total_received = df_account[df_account['RECEIVER_ACCOUNT_ID'] == central_account_id]['TX_AMOUNT'].sum()
    # net_flow is calculated based on the central account's perspective
    net_flow = total_received - total_sent
    total_volume = total_sent + total_received

    # Heuristic Thresholds
    HIGH_DEGREE_THRESHOLD = 5
    MULE_FLOW_RATIO = 0.5
    MIN_DISPERSAL_RATIO = 0.25 # Must send out at least 25% of what it receives to be a Layering candidate

    patterns = []

    # 1. MULE / CONSOLIDATION DETECTION (High imbalance between flow)
    is_mule = False
    if total_volume > 0 and abs(net_flow) > MULE_FLOW_RATIO * total_volume:
        is_mule = True
        if net_flow > 0:
            patterns.append("Mule Account (Consolidation Target)")
        else:
            patterns.append("Mule Account (Dispersal Source)")

    # 2. SMURFING DETECTION (Many incoming deposits)
    if is_mule and net_flow > 0 and in_degree > HIGH_DEGREE_THRESHOLD:
        patterns.append("Smurfing Target (Many Deposits)")

    # 3. LAYERING DETECTION (Balanced flow, high throughput)
    if in_degree >= 2 and out_degree >= 2 and total_received > 0 and total_sent >= MIN_DISPERSAL_RATIO * total_received:
        # Final Check: Layering only applies if the flow is NOT dominated by Mule behavior
        if total_volume == 0 or abs(net_flow) < 0.20 * total_volume: # Must have < 20% imbalance
            patterns.append("Layering/Pass-Through Node")


    if not patterns:
        # If score is high but no structural pattern matches, label it generally suspicious
        return "Suspicious Activity Detected (GNN Score High)"

    return ", ".join(patterns)

# ----------------------------------------------------------------------
# --- DRAW AMLSIM DASHBOARD (Global View) ---
# ----------------------------------------------------------------------

def draw_amlsim_dashboard():
    # Load the main data
    df_scores = load_scores_data(SCORES_PATH)
    # Load graph data for total account count
    data = load_graph_data(PROCESSED_DATA_PATH)
    total_unique_accounts = data.x.size(0) if data is not None else 0


    st.title("🌎 AMLSim Banking Transaction Analyzer")
    st.markdown("### GraphSAGE Edge Classification Results")

    if df_scores is not None:

        # --- Metrics Panel ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions Analyzed", f"{len(df_scores):,}")
        col2.metric("Total Unique Accounts", f"{total_unique_accounts:,}")
        col3.metric("AML Fraud Ratio", f"{FRAUD_RATIO * 100:.2f}%")
        col4.metric("Model ROC-AUC (Test)", f"{BEST_MODEL_AUC:.4f}")
        st.markdown("---")

        # --- Bar Chart ---
        st.subheader("Transactions by Suspicion Score Range (0.05 steps)")
        bins = [round(x * 0.05, 2) for x in range(21)]
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
        # Ensure the score range column exists before trying to use it
        if 'SUSPICION_SCORE' in df_scores.columns:
            # Use right=False to ensure 0.05 is included in the first bin
            df_scores['SCORE_RANGE'] = pd.cut(df_scores['SUSPICION_SCORE'], bins=bins, labels=labels, include_lowest=True, right=False)
            range_counts = df_scores['SCORE_RANGE'].value_counts().sort_index()
            st.bar_chart(range_counts)
        else:
            st.warning("Suspicion score data not available for histogram.")
        st.markdown("---")

        # --- Feature 3: Leaderboards ---
        render_leaderboards(df_scores)
        st.markdown("---")

        # --- Main Graph View ---
        st.subheader("Interactive Suspicious Network Map (Global)")
        max_score = df_scores['SUSPICION_SCORE'].max()

        # Set the fixed default value
        FIXED_DEFAULT_THRESHOLD = 0.51805

        suspicion_threshold = st.slider(
            "Suspicion Score Threshold (Show Transactions Above This Score)",
            min_value=float(df_scores['SUSPICION_SCORE'].min()),
            max_value=float(max_score),
            value=float(FIXED_DEFAULT_THRESHOLD),
            step=0.00001,
            format="%.5f"
        )

        df_filtered = df_scores[df_scores['SUSPICION_SCORE'] >= suspicion_threshold].copy()
        st.write(f"Transactions above threshold {suspicion_threshold:.5f}: {len(df_filtered):,}")

        display_network_graph(df_filtered, "global_network_graph.html")

        st.markdown("---")
        st.subheader(f"Transaction Table (Filtered: {len(df_filtered):,} edges)")
        st.dataframe(
            df_filtered[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']]
            .sort_values(by='SUSPICION_SCORE', ascending=False)
            .head(100)
        )

# ----------------------------------------------------------------------
# --- DRAW AMLSIM DEEP DIVE VIEW ---
# ----------------------------------------------------------------------
def draw_amlsim_deep_dive():
    df_scores = load_scores_data(SCORES_PATH)

    st.title("Investigate a Single Account")

    # --- STATE MANAGEMENT FIX using st.form ---
    # Initialize session state for the ID to display, if it doesn't exist
    if 'amlsim_display_id' not in st.session_state:
        st.session_state.amlsim_display_id = ''

    # Create a form to handle input and submission
    with st.form(key='amlsim_search_form'):
        # Text input is inside the form. Use a simple key. Default to last searched ID.
        search_id_input = st.text_input(
            "Enter Account ID to investigate:",
            value=st.session_state.amlsim_display_id, # Show the last searched ID
            key='amlsim_search_box'
        )
        # The submit button for the form
        submitted = st.form_submit_button("Analyze Account")

        # When the form is submitted (button click OR Enter in text input):
        if submitted:
            # Update the display ID with the value from the text box AT THE TIME OF SUBMISSION
            st.session_state.amlsim_display_id = search_id_input.strip()
            # Rerun allows the rest of the page to update with the new ID
            st.rerun() # Use rerun to force update immediately

    # The rest of the page renders based on the display state variable
    if st.session_state.amlsim_display_id:
        account_id = st.session_state.amlsim_display_id
        st.markdown(f"### Deep Dive Analysis for Account: `{account_id}`")

        df_account = df_scores[
            (df_scores['SENDER_ACCOUNT_ID'] == account_id) |
            (df_scores['RECEIVER_ACCOUNT_ID'] == account_id)
        ].copy()

        if df_account.empty:
            st.error("Account ID not found in transaction data.")
            # Clear the display ID if the account is not found
            st.session_state.amlsim_display_id = ''
            return # Stop execution for this account

        # Calculate Stats
        tx_sent = df_account[df_account['SENDER_ACCOUNT_ID'] == account_id]
        tx_received = df_account[df_account['RECEIVER_ACCOUNT_ID'] == account_id]

        total_sent = tx_sent['TX_AMOUNT'].sum()
        total_received = tx_received['TX_AMOUNT'].sum()
        net_flow = total_received - total_sent
        avg_score = df_account['SUSPICION_SCORE'].mean()

        # --- PATTERN ANALYSIS CALL ---
        pattern_result = analyze_aml_patterns(df_account, net_flow, avg_score)

        st.markdown("#### Key Metrics and Pattern Classification")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Total $ Sent", f"${total_sent:,.2f}")
        stat_col2.metric("Total $ Received", f"${total_received:,.2f}")
        stat_col3.metric("Net $ Flow", f"${net_flow:,.2f}")
        stat_col4.metric("Avg. Suspicion Score", f"{avg_score:.4f}")

        # Display Pattern Result
        st.markdown(f"**Predicted AML Patterns:** :red[{pattern_result}]")
        st.markdown("---")


        # Display Egonet Graph
        st.markdown("#### Local Transaction Graph (Egonet)")
        # Use a unique filename for the egonet graph to avoid caching issues
        display_network_graph(df_account, f"egonet_graph_{account_id}.html")

        # Display Transaction History
        st.markdown("#### Full Transaction History")
        st.dataframe(
            df_account[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'TX_TYPE', 'IS_FRAUD', 'SUSPICION_SCORE']]
            .sort_values(by='SUSPICION_SCORE', ascending=False)
        )

# ----------------------------------------------------------------------
# --- AMLSIM ROUTER FUNCTION ---
# ----------------------------------------------------------------------

def draw_amlsim_router():
    st.sidebar.title("🧠 AMLSim Pages")
    amlsim_mode = st.sidebar.radio(
        "Choose view:",
        ["Global Dashboard", "Account Deep Dive"]
    )
    # Load data for both Deep Dive and Dashboard - moved inside router to ensure freshness
    df_scores = load_scores_data(SCORES_PATH)
    if df_scores is None:
        st.error("AMLSim data is required to view this project.")
        return

    if amlsim_mode == "Global Dashboard":
        draw_amlsim_dashboard()
    elif amlsim_mode == "Account Deep Dive":
        draw_amlsim_deep_dive()

