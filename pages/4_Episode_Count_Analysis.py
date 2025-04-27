import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess

# Set page configuration
st.set_page_config(page_title="Anime Episode Count Analysis", layout="wide")

# Page title and introduction
st.title("📊 Comprehensive Analysis of Anime Episode Counts")
st.markdown("""
This dashboard explores the relationship between anime episode count and all other parameters
including score, popularity, members, favorites, genres, studios, and temporal trends.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/anime_cleaned.csv")
    # Basic data cleaning
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
    df = df[df['episodes'] > 0]
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['members'] = pd.to_numeric(df['members'], errors='coerce')
    df['favorites'] = pd.to_numeric(df['favorites'], errors='coerce')
    
    # Create episode categories
    bins = [0, 1, 12, 24, 50, 100, float('inf')]
    labels = ['Movie/Special', 'Short (1-12)', 'Medium (13-24)', 'Long (25-50)', 'Very Long (51-100)', 'Ultra (>100)']
    df['episode_category'] = pd.cut(df['episodes'], bins=bins, labels=labels)
    
    # Ensure genre is list
    if 'genre' in df.columns:
        df['genre_list'] = df['genre'].fillna('Unknown').str.split(', ')
    
    return df

df = load_data()

# Sidebar for filtering
st.sidebar.header("📋 Filters")

# Year range filter if available
if 'aired_from_year' in df.columns:
    year_min = int(df['aired_from_year'].min())
    year_max = int(df['aired_from_year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range", 
        year_min, year_max, 
        (year_min, year_max)
    )
    df_filtered = df[(df['aired_from_year'] >= year_range[0]) & (df['aired_from_year'] <= year_range[1])]
else:
    df_filtered = df

# Genre filter if available
if 'genre_list' in df.columns:
    all_genres = set()
    for genres in df['genre_list'].dropna():
        all_genres.update(genres)
    
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        sorted(all_genres),
        []
    )
    
    if selected_genres:
        mask = df_filtered['genre_list'].apply(lambda x: any(genre in x for genre in selected_genres) if isinstance(x, list) else False)
        df_filtered = df_filtered[mask]

# Type filter if available
if 'type' in df.columns:
    types = df_filtered['type'].dropna().unique()
    selected_types = st.sidebar.multiselect(
        "Select Types",
        types,
        []
    )
    
    if selected_types:
        df_filtered = df_filtered[df_filtered['type'].isin(selected_types)]

# Main content
tab1, tab2 = st.tabs(["📈 Basic Analysis", "🔍 Advanced Analysis"])

def remove_outliers(df, column, z_thresh=3):
    """Remove outliers using Z-score method."""
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] >= mean - z_thresh * std) & (df[column] <= mean + z_thresh * std)]

with tab1:
    col1, col2 = st.columns([1, 2])

    # 1. Basic statistics about episodes
    with col1:
        st.header("Episode Count Statistics")
        
        # Basic metrics
        metrics_data = {
            "Average": df_filtered['episodes'].mean(),
            "Median": df_filtered['episodes'].median(),
            "Min": df_filtered['episodes'].min(),
            "Max": df_filtered['episodes'].max()
        }
        
        for label, value in metrics_data.items():
            st.metric(label, f"{value:.1f}" if label == "Average Episodes" else f"{value:.0f}")
        
        # Episode category distribution
        st.subheader("Episode Length Distribution")
        category_counts = df_filtered['episode_category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
    

    # Episode count histogram
    with col2:
        st.header("Episode Count Distribution")
        
        # Histogram
        fig = px.histogram(
            df_filtered, 
            x='episodes',
            nbins=50,
            marginal='box',
            title="Distribution of Anime Episodes",
            labels={'episodes': 'Number of Episodes', 'count': 'Number of Anime'},
            range_x=[0, df_filtered['episodes'].quantile(0.99)]  # Limit to 99th percentile to handle outliers
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
            category_counts, 
            x='Category', 
            y='Count',
            color='Category',
            labels={'Count': 'Number of Anime', 'Category': 'Episode Category'},
            title="Distribution of Anime by Episode Count"
        )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # 2. Episode count vs numerical parameters (Score, Popularity, Members, Favorites)
    st.header("Episode Count vs Numerical Parameters")

    # Select parameters for analysis
    numerical_cols = ['score', 'popularity', 'members', 'favorites']
    numerical_cols = [col for col in numerical_cols if col in df_filtered.columns]

    # Create tabs for different numerical parameters
    num_tabs = st.tabs([col.capitalize() for col in numerical_cols])

    for i, col in enumerate(numerical_cols):
        with num_tabs[i]:

            if col in df_filtered.columns:
                # Filter out nulls
                df_valid = df_filtered.dropna(subset=['episodes', col])
                
                # Remove outliers for both 'episodes' and selected 'col'
                df_valid = remove_outliers(df_valid, 'episodes')
                df_valid = remove_outliers(df_valid, col)
                
                
                    # Box plot by episode category
                fig = px.box(
                    df_valid, 
                    x='episode_category', 
                    y=col,
                    color='episode_category',
                    labels={'episode_category': 'Episode Category', col: col.capitalize()},
                    title=f"{col.capitalize()} by Episode Category"
                )
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation
            corr = df_valid['episodes'].corr(df_valid[col])
            st.info(f"**Correlation between Episode Count and {col.capitalize()}**: {corr:.3f}")

    # 3. Time trends in episode counts
    if 'aired_from_year' in df.columns:
        st.header("Episode Count Trends Over Time")
        
        # Calculate average episode count by year
        yearly_episodes = df_filtered.groupby('aired_from_year').agg(
            avg_episodes=('episodes', 'mean'),
            median_episodes=('episodes', 'median'),
            anime_count=('episodes', 'count')
        ).reset_index()
        
        # Create tabs for different views
        trend_tabs = st.tabs(["Average", "Median", "Count by Year"])
        
        # Average episodes over time
        with trend_tabs[0]:
            fig = px.line(
                yearly_episodes, 
                x='aired_from_year', 
                y='avg_episodes',
                markers=True,
                labels={'aired_from_year': 'Year', 'avg_episodes': 'Average Episodes'},
                title="Average Episode Count Over Time"
            )
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        # Median episodes over time
        with trend_tabs[1]:
            fig = px.line(
                yearly_episodes, 
                x='aired_from_year', 
                y='median_episodes',
                markers=True,
                labels={'aired_from_year': 'Year', 'median_episodes': 'Median Episodes'},
                title="Median Episode Count Over Time"
            )
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        # Anime count over time
        with trend_tabs[2]:
            fig = px.line(
                yearly_episodes, 
                x='aired_from_year', 
                y='anime_count',
                markers=True,
                labels={'aired_from_year': 'Year', 'anime_count': 'Number of Anime'},
                title="Number of Anime Releases Over Time"
            )
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # 4. Episode count by categorical parameters
    st.header("Episode Count by Categories")

    # Select up to 4 categorical columns that exist in the dataset
    possible_cats = ['type', 'studio', 'source', 'rating']
    categorical_cols = [col for col in possible_cats if col in df_filtered.columns][:4]

    if categorical_cols:
        # Create tabs for different categorical parameters
        cat_tabs = st.tabs([col.capitalize() for col in categorical_cols])
        
        for i, col in enumerate(categorical_cols):
            with cat_tabs[i]:
                # Get top categories by frequency
                top_cats = df_filtered[col].value_counts().head(10).index
                df_top = df_filtered[df_filtered[col].isin(top_cats)]
                df_top = remove_outliers(df_top,'episodes',2)
                
                # Box plot
                fig = px.box(
                    df_top, 
                    x=col, 
                    y='episodes',
                    color=col,
                    labels={col: col.capitalize(), 'episodes': 'Number of Episodes'},
                    title=f"Episode Count by {col.capitalize()} (Top 10)"
                )
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

    # 5. Genre and Episode Count Analysis
    if 'genre_list' in df_filtered.columns:
        st.header("Genre and Episode Count Analysis")
        
        # Explode the genre list to get one row per genre
        df_exploded = df_filtered.explode('genre_list')
        
        # Get top genres
        top_genres = df_exploded['genre_list'].value_counts().head(15).index
        df_top_genres = df_exploded[df_exploded['genre_list'].isin(top_genres)]
        
        # Average episodes by genre
        genre_episodes = df_top_genres.groupby('genre_list').agg(
            avg_episodes=('episodes', 'mean'),
            median_episodes=('episodes', 'median'),
            anime_count=('episodes', 'count')
        ).reset_index().sort_values('avg_episodes', ascending=False)
        
      
        
        
        # Bar chart for average episodes
        fig = px.bar(
            genre_episodes, 
            x='genre_list', 
            y='avg_episodes',
            color='anime_count',
            labels={'genre_list': 'Genre', 'avg_episodes': 'Average Episodes', 'anime_count': 'Anime Count'},
            title="Average Episode Count by Genre"
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        

    # 6. Episode count and score matrix analysis
    if 'score' in df_filtered.columns:
        st.header("Episode Count vs Score Matrix Analysis")
        
        # Create bins for both episodes and score
        episode_bins = [0, 1, 12, 24, 50, 100, float('inf')]
        episode_labels = ['Movie/Special', 'Short (1-12)', 'Medium (13-24)', 'Long (25-50)', 'Very Long (51-100)', 'Ultra (>100)']
        
        score_bins = [0, 6, 7, 8, 9, 10]
        score_labels = ['<6', '6-7', '7-8', '8-9', '9-10']
        
        df_matrix = df_filtered.copy()
        df_matrix['episode_bin'] = pd.cut(df_matrix['episodes'], bins=episode_bins, labels=episode_labels)
        df_matrix['score_bin'] = pd.cut(df_matrix['score'], bins=score_bins, labels=score_labels)
        
        # Create a cross-tabulation
        cross_tab = pd.crosstab(df_matrix['episode_bin'], df_matrix['score_bin'], normalize='all') * 100
        
        # Plot heatmap
        fig = px.imshow(
            cross_tab,
            labels=dict(x="Score Range", y="Episode Range", color="Percentage (%)"),
            x=cross_tab.columns,
            y=cross_tab.index,
            text_auto='.1f',
            aspect="auto",
            title="Heatmap: Episode Count vs Score Distribution (%)",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # Also calculate average score for each episode bin
        avg_scores = df_matrix.groupby('episode_bin')['score'].mean().reset_index()
        
        fig = px.bar(
            avg_scores, 
            x='episode_bin', 
            y='score',
            color='episode_bin',
            labels={'episode_bin': 'Episode Range', 'score': 'Average Score'},
            title="Average Score by Episode Range"
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

   
       
    # 7. Correlation Matrix
    st.header("Correlation Matrix")

    # Select numerical columns for correlation
    corr_cols = ['episodes', 'score', 'popularity', 'members', 'favorites']
    corr_cols = [col for col in corr_cols if col in df_filtered.columns]

    if len(corr_cols) > 1:
        # Create correlation matrix
        corr_matrix = df_filtered[corr_cols].corr()
        
        # Plot heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Parameter", y="Parameter", color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.index,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix Between Parameters"
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    # 9. Top anime by episode count
    st.header("Notable Anime by Episode Count")

    # Get top 10 anime with most episodes
    top_episode_anime = df_filtered.sort_values('episodes', ascending=False).head(10)
    if 'title' in top_episode_anime.columns:
        top_episode_anime_display = top_episode_anime[['title', 'episodes', 'score', 'popularity']].reset_index(drop=True)
        
        st.subheader("Top 10 Anime with Most Episodes")
        st.dataframe(top_episode_anime_display)

        # Create bar chart for top anime
        fig = px.bar(
            top_episode_anime,
            x='episodes',
            y='title' if 'title' in top_episode_anime.columns else 'anime_id',
            color='score' if 'score' in top_episode_anime.columns else None,
            orientation='h',
            labels={'episodes': 'Number of Episodes', 'title': 'Anime Title', 'score': 'Score'},
            title="Top 10 Anime by Episode Count"
        )
        fig.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # 10. Statistical summary tables
    st.header("Statistical Summary Tables")

    # Summary by episode category
    summary_by_category = df_filtered.groupby('episode_category').agg(
        anime_count=('episodes', 'count'),
        avg_score=('score', 'mean') if 'score' in df_filtered.columns else ('episodes', 'count'),
        avg_popularity=('popularity', 'mean') if 'popularity' in df_filtered.columns else ('episodes', 'count'),
        avg_members=('members', 'mean') if 'members' in df_filtered.columns else ('episodes', 'count'),
        avg_favorites=('favorites', 'mean') if 'favorites' in df_filtered.columns else ('episodes', 'count'),
    ).reset_index()

    for col in summary_by_category.columns:
        if col != 'episode_category' and col != 'anime_count':
            summary_by_category[col] = summary_by_category[col].round(2)

    st.subheader("Summary Statistics by Episode Category")
    st.dataframe(summary_by_category)

    # Correlation table for episodes and other parameters
    st.subheader("Correlation with Episode Count")

    corr_values = {}
    for col in ['score', 'popularity', 'members', 'favorites']:
        if col in df_filtered.columns:
            corr_values[col] = df_filtered['episodes'].corr(df_filtered[col])

    corr_df = pd.DataFrame(list(corr_values.items()), columns=['Parameter', 'Correlation with Episodes'])
    corr_df['Correlation with Episodes'] = corr_df['Correlation with Episodes'].round(3)
    st.dataframe(corr_df)
