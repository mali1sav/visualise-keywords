import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration to wide layout
st.set_page_config(page_title="Keyword Clustering (DBSCAN + UMAP)", layout="wide")

###############################################################################
# Utility Functions
###############################################################################
def parse_input_line(line: str):
    """
    Parse a single line (e.g. 'crypto wallet 600' or 'cold wallet คืออะไร\t70')
    into (keyword, volume).
    - Accepts either tab or space as a separator.
    - The last token must be an integer (search volume).
    """
    if not line or not line.strip():
        return None
    
    # Try tab split first
    parts = line.strip().split('\t')
    if len(parts) == 2 and parts[1].strip().isdigit():
        return parts[0].strip(), int(parts[1].strip())
    
    # Otherwise, space split
    parts = line.strip().split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return ' '.join(parts[:-1]), int(parts[-1])
    
    return None

def get_default_keywords():
    """
    Provides a default set of keyword-volume pairs for demonstration.
    """
    return """crypto wallet\t600
wallet crypto\t100
hot wallet\t100
hot wallet คือ\t80
cold wallet คืออะไร\t70
crypto wallet อันไหนดี\t60
คุณสมบัติของ hot wallet ทั้งหมด\t60
คุณสมบัติของ hot wallet\t60
crypto wallet app\t50
crypto wallet คือ\t50
hardware wallet ใช้ยังไง\t40
กระเป๋าเงิน crypto\t40
hardware wallet crypto\t30
กระเป๋า crypto wallet\t30
กระเป๋า crypto\t30
hot wallet คืออะไร\t20
hot wallet มีอะไรบ้าง\t10
hot wallet crypto\t10
wallet crypto แนะนํา\t10
crypto wallet คืออะไร\t10
crypto wallet มีอะไรบ้าง\t10
กระเป๋า crypto อันไหนดี\t10
type of crypto wallet\t10
"""

@st.cache_resource
def load_local_model():
    """
    Loads the all-mpnet-base-v2 model and caches it.
    """
    return SentenceTransformer("all-mpnet-base-v2")

def cluster_and_visualize(keywords, volumes, model, eps_val, min_samples_val):
    """
    1) Embeds the keywords locally via SentenceTransformer.
    2) Clusters them using DBSCAN (with user-controlled eps, min_samples).
    3) Reduces dimension using UMAP for 2D visualization.
    4) Plots the clusters with Plotly and displays the cluster membership.
    """
    st.write("### Generating embeddings...")
    embeddings = model.encode(keywords, show_progress_bar=False)

    st.write("### Clustering with DBSCAN...")
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    cluster_labels = dbscan.fit_predict(embeddings)

    unique_clusters = sorted(set(cluster_labels))
    noise_count = sum(1 for c in cluster_labels if c == -1)
    noise_percentage = (noise_count / len(cluster_labels)) * 100
    st.write(f"Detected clusters: {unique_clusters}")
    st.write(f"Noise points (outliers): {noise_count} / {len(cluster_labels)} "
             f"({noise_percentage:.2f}%)")

    st.write("### Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reduced_points = reducer.fit_transform(embeddings)

    # Create a color palette for the clusters
    color_palette = px.colors.qualitative.Bold
    fig = go.Figure()

    for cluster_id in unique_clusters:
        idxs = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
        xvals = reduced_points[idxs, 0]
        yvals = reduced_points[idxs, 1]
        hover_texts = [f"{keywords[i]}<br>Vol: {volumes[i]}" for i in idxs]
        sizes = [10 + (volumes[i] / 10) for i in idxs]

        # Assign color for cluster_id; -1 is for noise
        color_idx = cluster_id if cluster_id >= 0 else len(color_palette) - 1
        color = color_palette[color_idx % len(color_palette)]

        cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise/Outliers"

        fig.add_trace(go.Scatter(
            x=xvals,
            y=yvals,
            mode='markers+text',
            text=[keywords[i] for i in idxs],  # show just keyword
            textposition='top center',
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.7,
                line=dict(color='black', width=1)
            ),
            name=cluster_name
        ))

    fig.update_layout(
        title="Keyword Clusters (DBSCAN + UMAP)",
        legend_title="Cluster Label",
        width=1200  # Set a fixed width for the Plotly chart
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show cluster membership
    st.subheader("Cluster Analysis")
    cluster_dict = {}
    for i, c_id in enumerate(cluster_labels):
        cluster_dict.setdefault(c_id, []).append((keywords[i], volumes[i]))

    # Display cluster membership in columns
    cols = st.columns(len(unique_clusters))
    for idx, c_id in enumerate(unique_clusters):
        with cols[idx]:
            if c_id != -1:
                st.markdown(f"**Cluster {c_id}**")
            else:
                st.markdown("**Noise/Outliers**")
            items = cluster_dict[c_id]
            for kw, vol in items:
                st.write(f"- {kw}")

###############################################################################
# Streamlit App
###############################################################################
def main():
    st.title("Local Embeddings + DBSCAN + UMAP for Keyword Clustering")
    st.write("Paste your keyword-volume pairs below, then adjust parameters.")

    # Load the model
    model = load_local_model()

    # Sliders for DBSCAN parameters
    eps_val = st.slider("DBSCAN eps (distance threshold)", 0.4, 3.0, 0.5, 0.1)
    min_samples_val = st.slider("DBSCAN min_samples (minimum points per cluster)", 2, 10, 2)

    default_text = get_default_keywords()
    user_text = st.text_area("Enter keyword-volume pairs:", value=default_text, height=250)

    if st.button("Visualize"):
        lines = user_text.strip().split("\n")
        data = []
        for line in lines:
            parsed = parse_input_line(line)
            if parsed:
                data.append(parsed)
        
        if not data:
            st.warning("No valid keyword-volume pairs found.")
            return
        
        keywords, volumes = zip(*data)
        cluster_and_visualize(list(keywords), list(volumes), model, eps_val, min_samples_val)

if __name__ == "__main__":
    main()