import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from itertools import combinations
import networkx as nx
from streamlit.components.v1 import html
import time
import numpy as np

# Page configuration
st.set_page_config(page_title="Anime Genre Network", layout="wide")

# Custom styling
st.markdown("""
<style>
    .main {background-color: #000;}
    h1 {color: #1e1e1e;}
    .highlight {
        background-color: #000;
        border-radius: 4px;
        padding: 20px;
        border-left: 3px solid #4c78a8;
    }
    .stats-card {
        background-color: #000;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Page title and introduction
st.title("🎭 Anime Genre Co-occurrence Network")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('./data/anime_cleaned.csv')
    df['genre'] = df['genre'].fillna('Unknown')
    df['genre'] = df['genre'].str.split(', ')
    
    # Add year information if available
    if 'aired_from_year' in df.columns:
        df['aired_from_year'] = pd.to_numeric(df['aired_from_year'], errors='coerce')
    
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("🛠️ Network Controls")

# Co-occurrence threshold slider
threshold = st.sidebar.slider(
    "Minimum Co-occurrence Threshold", 
    min_value=5, 
    max_value=200, 
    value=30, 
    step=5,
    help="Only show connections between genres that appear together in at least this many anime"
)

# Year range filter (if available)
if 'aired_from_year' in df.columns:
    year_min = int(df['aired_from_year'].min())
    year_max = int(df['aired_from_year'].max())
    year_range = st.sidebar.slider(
        "Year Range", 
        year_min, 
        year_max, 
        (year_min, year_max),
        help="Filter anime by release year"
    )
    df_filtered = df[(df['aired_from_year'] >= year_range[0]) & (df['aired_from_year'] <= year_range[1])]
else:
    df_filtered = df

# Genre focus
all_genres = set()
for genres in df['genre']:
    if isinstance(genres, list):
        all_genres.update(genres)

focus_genre = st.sidebar.selectbox(
    "Focus on specific genre",
    ["None"] + sorted(all_genres),
    help="Highlight a specific genre and its connections"
)

# Visualization options
st.sidebar.header("🎨 Visualization Options")
viz_style = st.sidebar.radio(
    "Visualization Type",
    ["Network Graph", "Heatmap Matrix", "Chord Diagram"],
    help="Choose how to visualize genre relationships"
)

# Count genre pairs and build co-occurrence data
@st.cache_data
def build_cooccurrence_data(df, threshold=30):
    # Count pairs
    pair_counter = Counter()
    genre_counter = Counter()
    
    for genres in df['genre']:
        if isinstance(genres, list) and len(genres) > 1:
            genre_counter.update(genres)
            pairs = combinations(sorted(set(genres)), 2)
            pair_counter.update(pairs)
    
    # Filter by threshold
    filtered_pairs = {pair: count for pair, count in pair_counter.items() if count >= threshold}
    
    # Create edge list
    edges = [(g1, g2, count) for (g1, g2), count in filtered_pairs.items()]
    
    # Build graph
    G = nx.Graph()
    
    for genre, count in genre_counter.items():
        G.add_node(genre, size=count)
    
    for g1, g2, weight in edges:
        G.add_edge(g1, g2, weight=weight)
    
    return G, genre_counter, filtered_pairs, edges

G, genre_counter, filtered_pairs, edges = build_cooccurrence_data(df_filtered, threshold)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Network Visualization", "Genre Analytics", "Top Combinations"])

# Tab 1: Network visualization
with tab1:
    if not edges:
        st.warning(f"No genre pairs meet the threshold of {threshold}. Try lowering the threshold.")
    else:
        if viz_style == "Network Graph":
            # Generate NetworkX positions
            pos = nx.spring_layout(G, seed=42)
            
            # Create edge traces
            edge_x = []
            edge_y = []
            edge_text = []
            edge_width = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(f"{edge[0]} + {edge[1]}: {edge[2]['weight']} co-occurrences")
                edge_width.append(np.sqrt(edge[2]['weight']) / 2)
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node}: appears in {G.nodes[node]['size']} anime")
                node_size.append(np.sqrt(G.nodes[node]['size']) * 1.5)
                
                # Highlight focus genre if selected
                if focus_genre != "None" and node == focus_genre:
                    node_color.append("red")
                elif focus_genre != "None" and focus_genre in G and node in G.neighbors(focus_genre):
                    node_color.append("orange")
                else:
                    node_color.append("skyblue")
            
            # Create the figure
            fig = go.Figure()
            
            # Add edges
            if edge_x:
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.8, color='#888'),
                    hoverinfo='text',
                    text=edge_text,
                    mode='lines',
                    name='Co-occurrences'
                ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=1, color='#333')
                ),
                name='Genres'
            ))
            
            # Add node labels
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='text',
                text=[node if G.nodes[node]['size'] > np.percentile([G.nodes[n]['size'] for n in G.nodes()], 50) else '' for node in G.nodes()],
                textposition="top center",
                textfont=dict(size=10, color='black'),
                hoverinfo='none',
                name='Labels'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Genre Co-occurrence Network (Threshold: {threshold}+)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network stats
            st.markdown("### Network Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='stats-card'><h3>Genres</h3><p style='font-size:24px;'>{len(G.nodes())}</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='stats-card'><h3>Connections</h3><p style='font-size:24px;'>{len(G.edges())}</p></div>", unsafe_allow_html=True)
            with col3:
                avg_connections = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                st.markdown(f"<div class='stats-card'><h3>Avg. Connections</h3><p style='font-size:24px;'>{avg_connections:.1f}</p></div>", unsafe_allow_html=True)
            with col4:
                density = nx.density(G)
                st.markdown(f"<div class='stats-card'><h3>Network Density</h3><p style='font-size:24px;'>{density:.3f}</p></div>", unsafe_allow_html=True)
            
        elif viz_style == "Heatmap Matrix":
            # Create adjacency matrix
            genres = sorted(list(G.nodes()))
            matrix = np.zeros((len(genres), len(genres)))
            
            # Fill matrix with co-occurrence counts
            for i, g1 in enumerate(genres):
                for j, g2 in enumerate(genres):
                    if G.has_edge(g1, g2):
                        matrix[i][j] = G[g1][g2]['weight']
            
            # Create heatmap
            fig = px.imshow(
                matrix,
                x=genres,
                y=genres,
                labels=dict(x="Genre", y="Genre", color="Co-occurrences"),
                color_continuous_scale="Viridis",
                title=f"Genre Co-occurrence Matrix (Threshold: {threshold}+)"
            )
            
            fig.update_layout(
                height=800,
                xaxis_tickangle=-45,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_style == "Chord Diagram":
            # Prepare data for chord diagram
            genres = sorted(list(G.nodes()))
            matrix = np.zeros((len(genres), len(genres)))
            
            # Fill matrix with co-occurrence counts
            for i, g1 in enumerate(genres):
                for j, g2 in enumerate(genres):
                    if G.has_edge(g1, g2):
                        matrix[i][j] = G[g1][g2]['weight']
            
            # Create chord diagram
            fig = go.Figure(go.Heatmap(
                z=matrix,
                x=genres,
                y=genres,
                colorscale='Viridis',
                showscale=True,
                text=matrix,
                texttemplate="%{text}",
                hovertemplate='%{y} & %{x}: %{z} co-occurrences<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Genre Co-occurrence Chord Matrix (Threshold: {threshold}+)",
                height=800,
                xaxis_tickangle=-45,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Genre Analytics
with tab2:
    st.header("Genre Analytics")
    
    # Top genres by frequency
    top_genres = pd.DataFrame({
        'Genre': list(genre_counter.keys()),
        'Anime Count': list(genre_counter.values())
    }).sort_values('Anime Count', ascending=False).head(15)
    
    # Top connected genres
    if len(G.nodes()) > 0:
        connected_genres = pd.DataFrame({
            'Genre': list(G.nodes()),
            'Connections': [G.degree(node) for node in G.nodes()]
        }).sort_values('Connections', ascending=False).head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Common Genres")
            fig1 = px.bar(
                top_genres,
                x='Anime Count',
                y='Genre',
                orientation='h',
                color='Anime Count',
                color_continuous_scale='Viridis',
                title="Top 15 Most Common Genres"
            )
            fig1.update_layout(template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Most Connected Genres")
            fig2 = px.bar(
                connected_genres,
                x='Connections',
                y='Genre',
                orientation='h',
                color='Connections',
                color_continuous_scale='Viridis',
                title="Top 15 Genres with Most Connections"
            )
            fig2.update_layout(template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)
    
    # Genre-specific analysis
    st.subheader("Single Genre Analysis")
    selected_genre = st.selectbox(
        "Select a genre to analyze", 
        [""] + sorted(list(G.nodes())),
        index=0
    )
    
    if selected_genre:
        if selected_genre in G.nodes():
            # Get connected genres
            neighbors = list(G.neighbors(selected_genre))
            connections = [(n, G[selected_genre][n]['weight']) for n in neighbors]
            connections_sorted = sorted(connections, key=lambda x: x[1], reverse=True)
            
            # Create dataframe
            connections_df = pd.DataFrame(connections_sorted, columns=['Connected Genre', 'Co-occurrence Count'])
            
            # Display information
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Appears in", G.nodes[selected_genre]['size'], "anime")
                st.metric("Connected to", len(neighbors), "other genres")
                st.dataframe(connections_df, height=400)
            
            with col2:
                # Bar chart of connections
                fig = px.bar(
                    connections_df.head(10),
                    x='Co-occurrence Count',
                    y='Connected Genre',
                    orientation='h',
                    color='Co-occurrence Count',
                    color_continuous_scale='Viridis',
                    title=f"Top Genres Connected to '{selected_genre}'"
                )
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show example anime with this genre
            if 'genre' in df.columns and 'title' in df.columns:
                st.subheader(f"Example Anime with '{selected_genre}' Genre")
                genre_anime = df_filtered[df_filtered['genre'].apply(
                    lambda x: selected_genre in x if isinstance(x, list) else False
                )]
                
                cols_to_show = ['title', 'score', 'aired_from_year'] if all(col in genre_anime.columns for col in ['score', 'aired_from_year']) else ['title']
                st.dataframe(genre_anime[cols_to_show].head(10), use_container_width=True)
        else:
            st.warning(f"Genre '{selected_genre}' not found in the network. It may not meet the co-occurrence threshold of {threshold}.")

# Tab 3: Top Combinations
with tab3:
    st.header("Top Genre Combinations")
    
    # Create a dataframe of all pairs
    if filtered_pairs:
        pairs_df = pd.DataFrame([
            {'Genre 1': g1, 'Genre 2': g2, 'Co-occurrences': count}
            for (g1, g2), count in filtered_pairs.items()
        ]).sort_values('Co-occurrences', ascending=False)
        
        # Top pairs
        st.subheader("Most Common Genre Combinations")
        fig = px.bar(
            pairs_df.head(15),
            x='Co-occurrences',
            y=[f"{row['Genre 1']} + {row['Genre 2']}" for _, row in pairs_df.head(15).iterrows()],
            orientation='h',
            color='Co-occurrences',
            color_continuous_scale='Viridis',
            title="Top 15 Genre Combinations"
        )
        fig.update_layout(template='plotly_white', yaxis_title='Genre Pair')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display all pairs
        st.subheader("All Genre Combinations")
        st.dataframe(pairs_df, use_container_width=True)
        
        # Download button
        csv = pairs_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="genre_combinations.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No genre pairs meet the threshold of {threshold}. Try lowering the threshold.")

# Explanation
with tab1:
    st.markdown("""
    ### How to Interpret This Visualization

    - **Nodes (Circles)** represent individual anime genres
    - **Node Size** indicates how many anime feature that genre
    - **Edges (Lines)** show which genres commonly appear together 
    - **Edge Thickness** represents how frequently the genres co-occur
    - **Focus Mode** highlights a specific genre and its connections when selected

    This network visualization reveals storytelling patterns in anime, showing which genre combinations are most common.
    """)
