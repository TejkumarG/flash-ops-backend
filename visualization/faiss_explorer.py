"""
FAISS Vector Store Explorer - Web UI for visualizing FAISS index.
Similar to Attu for Milvus.
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="FAISS Vector Store Explorer",
    page_icon="🔍",
    layout="wide"
)

# Load FAISS index and metadata
@st.cache_resource
def load_vector_store():
    """Load FAISS index and metadata."""
    base_path = Path(__file__).parent.parent / "data" / "embeddings"
    index_path = base_path / "table_index.faiss"
    metadata_path = base_path / "table_metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return index, metadata

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_all_vectors(index):
    """Extract all vectors from FAISS index."""
    n_vectors = index.ntotal
    vectors = np.zeros((n_vectors, index.d))
    for i in range(n_vectors):
        vectors[i] = index.reconstruct(i)
    return vectors

def visualize_embeddings_2d(vectors, metadata, method='PCA'):
    """Create 2D visualization of embeddings."""
    if method == 'PCA':
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)

    vectors_2d = reducer.fit_transform(vectors)

    df = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'table': [m['table_name'] for m in metadata],
        'category': [m.get('category', 'unknown') for m in metadata],
        'rows': [m.get('row_count', 0) for m in metadata],
        'description': [m.get('description', '')[:100] + '...' for m in metadata]
    })

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='category',
        hover_data=['table', 'rows', 'description'],
        title=f'Table Embeddings Visualization ({method})',
        width=900,
        height=600,
        size_max=15
    )

    fig.update_traces(marker=dict(size=12))

    return fig

def search_similar_tables(query, index, metadata, model, top_k=10):
    """Search for similar tables using query."""
    query_embedding = model.encode([query], show_progress_bar=False)
    distances, indices = index.search(query_embedding.astype('float32'), k=top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        table = metadata[idx]
        similarity = 1 / (1 + dist)
        results.append({
            'Rank': rank,
            'Table': table['table_name'],
            'Category': table.get('category', 'N/A'),
            'Rows': table.get('row_count', 0),
            'Similarity': f"{similarity:.3f}",
            'Distance': f"{dist:.3f}",
            'Description': table.get('description', 'N/A')
        })

    return pd.DataFrame(results)

# Main UI
st.title("🔍 FAISS Vector Store Explorer")
st.markdown("*Explore your FAISS index - similar to Attu for Milvus*")

# Load data
index, metadata = load_vector_store()

if index is None or metadata is None:
    st.error("❌ FAISS index not found. Please generate embeddings first.")
    st.code("curl -X POST 'http://localhost:8000/api/v1/embeddings/generate/default'")
    st.stop()

model = load_embedding_model()

# Sidebar - Index Statistics
st.sidebar.header("📊 Index Statistics")
st.sidebar.metric("Total Tables", index.ntotal)
st.sidebar.metric("Vector Dimensions", index.d)
st.sidebar.metric("Index Type", type(index).__name__)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Semantic Search",
    "📊 Vector Visualization",
    "📋 Browse Tables",
    "📈 Statistics"
])

# Tab 1: Semantic Search
with tab1:
    st.header("Semantic Search")
    st.markdown("Search for tables using natural language queries")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., 'employee information', 'sales data', 'customer records'",
            key="search_query"
        )
    with col2:
        top_k = st.number_input("Top K results:", min_value=1, max_value=50, value=10)

    if query:
        with st.spinner("Searching..."):
            results_df = search_similar_tables(query, index, metadata, model, top_k)
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            # Visualization of top results
            fig = px.bar(
                results_df.head(5),
                x='Table',
                y='Similarity',
                color='Category',
                title='Top 5 Matches by Similarity',
                text='Similarity'
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Vector Visualization
with tab2:
    st.header("Vector Space Visualization")
    st.markdown("2D projection of table embeddings")

    col1, col2 = st.columns([1, 3])
    with col1:
        viz_method = st.radio("Reduction Method:", ['PCA', 't-SNE'])

        if st.button("Generate Visualization"):
            with st.spinner(f"Computing {viz_method}..."):
                vectors = get_all_vectors(index)
                fig = visualize_embeddings_2d(vectors, metadata, method=viz_method)
                st.session_state['viz_fig'] = fig

    with col2:
        if 'viz_fig' in st.session_state:
            st.plotly_chart(st.session_state['viz_fig'], use_container_width=True)
        else:
            st.info("Click 'Generate Visualization' to create the plot")

# Tab 3: Browse Tables
with tab3:
    st.header("Browse All Tables")

    # Convert metadata to DataFrame
    tables_data = []
    for i, meta in enumerate(metadata):
        tables_data.append({
            'ID': i,
            'Table Name': meta['table_name'],
            'Category': meta.get('category', 'N/A'),
            'Row Count': meta.get('row_count', 0),
            'Columns': len(meta.get('columns', [])),
            'Description': meta.get('description', 'N/A')
        })

    tables_df = pd.DataFrame(tables_data)

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        search_filter = st.text_input("Filter by table name:", "")
    with col2:
        category_filter = st.multiselect(
            "Filter by category:",
            options=tables_df['Category'].unique()
        )

    # Apply filters
    filtered_df = tables_df
    if search_filter:
        filtered_df = filtered_df[
            filtered_df['Table Name'].str.contains(search_filter, case=False)
        ]
    if category_filter:
        filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Table details
    if len(filtered_df) > 0:
        selected_table_id = st.selectbox(
            "Select table to view details:",
            filtered_df['ID'].tolist(),
            format_func=lambda x: metadata[x]['table_name']
        )

        if selected_table_id is not None:
            st.subheader(f"Details: {metadata[selected_table_id]['table_name']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Category:** {metadata[selected_table_id].get('category', 'N/A')}")
                st.markdown(f"**Row Count:** {metadata[selected_table_id].get('row_count', 0):,}")
            with col2:
                st.markdown(f"**Columns:** {len(metadata[selected_table_id].get('columns', []))}")

            st.markdown(f"**Description:**")
            st.info(metadata[selected_table_id].get('description', 'No description available'))

            # Show columns
            if 'columns' in metadata[selected_table_id]:
                st.markdown("**Columns:**")
                cols_df = pd.DataFrame(metadata[selected_table_id]['columns'])
                st.dataframe(cols_df, use_container_width=True, hide_index=True)

            # Show embedding vector
            with st.expander("View Embedding Vector"):
                vec = index.reconstruct(selected_table_id)
                st.code(f"Shape: {vec.shape}\n\nFirst 20 dimensions:\n{vec[:20]}")

# Tab 4: Statistics
with tab4:
    st.header("Index Statistics & Analytics")

    # Category distribution
    categories = [m.get('category', 'unknown') for m in metadata]
    category_counts = pd.Series(categories).value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Tables by Category'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Row count distribution
        row_counts = [m.get('row_count', 0) for m in metadata]
        fig_bar = px.bar(
            x=[m['table_name'] for m in metadata],
            y=row_counts,
            title='Row Counts by Table',
            labels={'x': 'Table', 'y': 'Row Count'}
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Vector statistics
    st.subheader("Vector Statistics")
    vectors = get_all_vectors(index)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{vectors.mean():.4f}")
    col2.metric("Std Dev", f"{vectors.std():.4f}")
    col3.metric("Min", f"{vectors.min():.4f}")
    col4.metric("Max", f"{vectors.max():.4f}")

    # Similarity heatmap (for small number of tables)
    if len(metadata) <= 20:
        st.subheader("Table Similarity Heatmap")
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(vectors)
        table_names = [m['table_name'] for m in metadata]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=table_names,
            y=table_names,
            colorscale='Viridis'
        ))
        fig_heatmap.update_layout(
            title='Cosine Similarity between Tables',
            xaxis={'tickangle': 45},
            height=600
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Quick Links")
st.sidebar.markdown("[API Docs](http://localhost:8000/docs)")
st.sidebar.markdown("[Health Check](http://localhost:8000/health)")
st.sidebar.markdown("---")
st.sidebar.caption("FAISS Vector Store Explorer v1.0")
