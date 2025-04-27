import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(page_title="🌍 Regional Anime Preferences", layout="wide")
st.title("📊 Regional Anime Preferences Analysis")
st.markdown("Explore how anime preferences vary across different countries and regions.")

# Load data with caching
@st.cache_data
def load_data():
    df_users = pd.read_csv("./data/users_cleaned.csv")
    df_anime_lists = pd.read_csv("./data/animelists_cleaned.csv")
    df_anime = pd.read_csv("./data/anime_cleaned.csv")
    return df_users, df_anime_lists, df_anime

df_users, df_anime_lists, df_anime = load_data()

# Clean country data
if 'location' in df_users.columns:
    df_users['country'] = df_users['location'].str.extract(r'([A-Za-z\s]+)$')[0]
    df_users['country'] = df_users['country'].fillna('Unknown').str.strip()
else:
    df_users['country'] = 'Unknown'

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📺 Watch Time Analysis", "🎭 Genre Preferences", "🔍 Detailed Country Analysis"])

with tab1:
    # Watch time analysis
    watchtime_region = df_users.groupby('country')['user_days_spent_watching'].sum().sort_values(ascending=False).head(15)
    
    fig1 = px.bar(
        watchtime_region,
        x=watchtime_region.values,
        y=watchtime_region.index,
        orientation='h',
        color=watchtime_region.values,
        color_continuous_scale='Viridis',
        labels={'x': 'Total Days Spent Watching Anime', 'y': 'Country'},
        title="Total Days Spent Watching Anime (Top 15 Countries)"
    )
    fig1.update_layout(template='plotly_white')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Per capita analysis (if population data available)
    st.subheader("📊 Watch Time Statistics")
    col1, col2 = st.columns(2)
    with col1:
        total_watchtime = df_users['user_days_spent_watching'].sum()
        avg_watchtime = df_users['user_days_spent_watching'].mean()
        st.metric("Total Watch Time", f"{total_watchtime:,.0f} days")
        st.metric("Average Watch Time per User", f"{avg_watchtime:.1f} days")
    
    with col2:
        country_count = df_users['country'].nunique()
        user_count = len(df_users)
        st.metric("Countries Represented", f"{country_count:,}")
        st.metric("Total Users", f"{user_count:,}")

with tab2:
    # Merge dataframes and prepare for stratified sampling
    st.subheader("🔥 Genre Popularity by Country (Stratified Sample)")
    
    # Stratified sampling controls
    sample_size = st.slider("Total Sample Size", 5000, 50000, 10000, step=1000)
    top_n_countries = st.slider("Number of Top Countries to Show", 5, 20, 10)
    
    # Perform stratified sampling
    st.info("Using stratified sampling to ensure fair representation of each country")
    
    # Merge data
    merged = pd.merge(df_anime_lists, df_users[['username', 'country']], on='username', how='left')
    merged = pd.merge(merged, df_anime[['anime_id', 'genre']], on='anime_id', how='left')
    
    # Find top countries by activity
    top_regions = merged.groupby('country').size().sort_values(ascending=False).head(top_n_countries).index
    merged_top = merged[merged['country'].isin(top_regions)]
    
    # Stratified sampling
    @st.cache_data
    def stratified_sample(df, country_col, max_per_country):
        countries = df[country_col].unique()
        sampled_data = []
        
        for country in countries:
            country_data = df[df[country_col] == country]
            # If country has less than max_per_country, take all; else sample
            if len(country_data) > max_per_country:
                country_sample = country_data.sample(max_per_country, random_state=42)
            else:
                country_sample = country_data
            sampled_data.append(country_sample)
        
        return pd.concat(sampled_data)
    
    # Calculate samples per country based on total desired sample size
    max_per_country = sample_size // len(top_regions)
    sampled_merged = stratified_sample(merged_top, 'country', max_per_country)
    
    # Process genre data
    sampled_merged['genre'] = sampled_merged['genre'].fillna('Unknown').str.split(', ')
    sampled_merged = sampled_merged.explode('genre')
    
    # Group by country and genre
    genre_region = sampled_merged.groupby(['country', 'genre']).size().reset_index(name='count')
    
    # Create a pivot table for the heatmap
    heatmap_data = genre_region.pivot_table(index='genre', columns='country', values='count', fill_value=0)
    
    # Option to normalize data
    normalize = st.checkbox("Show as percentage of each country's total")
    if normalize:
        heatmap_data = heatmap_data / heatmap_data.sum(axis=0) * 100
        
    # Filter to show only genres that appear in at least X countries
    min_countries = st.slider("Show genres that appear in at least X countries", 1, top_n_countries, 3)
    genre_country_count = (heatmap_data > 0).sum(axis=1)
    filtered_genres = genre_country_count[genre_country_count >= min_countries].index
    filtered_heatmap = heatmap_data.loc[filtered_genres]
    
    # Sort genres by total popularity
    sorted_genres = filtered_heatmap.sum(axis=1).sort_values(ascending=False).index
    filtered_heatmap = filtered_heatmap.loc[sorted_genres]
    
    # Create interactive heatmap
    fig2 = px.imshow(
        filtered_heatmap,
        labels=dict(x="Country", y="Genre", color="Popularity" if not normalize else "Percentage (%)"),
        aspect="auto",
        title=f"Genre Popularity by Country (Top {top_n_countries} Countries, Stratified Sample)",
        color_continuous_scale="Viridis"
    )
    fig2.update_layout(template='plotly_white', height=800)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Top genres for each country
    st.subheader("🏆 Top Genres by Country")
    
    top_genres_per_country = (
        genre_region
        .sort_values(['country', 'count'], ascending=[True, False])
        .groupby('country')
        .head(5)
        .reset_index(drop=True)
    )
    
    # Create a table with colored bars
    fig3 = go.Figure()
    
    countries = top_genres_per_country['country'].unique()
    for country in countries:
        country_data = top_genres_per_country[top_genres_per_country['country'] == country]
        fig3.add_trace(go.Bar(
            name=country,
            x=country_data['genre'],
            y=country_data['count'],
            text=country_data['count'],
            textposition='auto'
        ))
    
    fig3.update_layout(
        title='Top 5 Genres by Country',
        xaxis_title='Genre',
        yaxis_title='Count',
        template='plotly_white',
        legend_title='Country',
        barmode='group'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    # Detailed country analysis
    st.subheader("🔍 Country-Specific Analysis")
    
    selected_country = st.selectbox(
        "Select a country to analyze",
        options=top_regions,
        index=0
    )
    
    if selected_country:
        country_data = merged[merged['country'] == selected_country]
        users_count = country_data['username'].nunique()
        anime_count = country_data['anime_id'].nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Users", f"{users_count:,}")
        with col2:
            st.metric("Anime Watched", f"{anime_count:,}")
        with col3:
            avg_per_user = anime_count / users_count if users_count > 0 else 0
            st.metric("Avg. Anime per User", f"{avg_per_user:.1f}")
        
        # Genre preferences for selected country
        country_genres = sampled_merged[sampled_merged['country'] == selected_country]
        genre_counts = country_genres.groupby('genre').size().reset_index(name='count')
        genre_counts = genre_counts.sort_values('count', ascending=False).head(10)
        
        fig4 = px.pie(
            genre_counts,
            values='count',
            names='genre',
            title=f"Top Genres in {selected_country}",
            hole=0.4
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # If anime has score data, show average scores by genre for this country
        if 'score' in df_anime.columns:
            st.subheader(f"Average Scores by Genre in {selected_country}")
            
            # Merge score data
            country_scores = pd.merge(
                country_data[['anime_id', 'username']],
                df_anime[['anime_id', 'genre', 'score']],
                on='anime_id'
            )
            
            # Process genre data
            country_scores['genre'] = country_scores['genre'].fillna('Unknown').str.split(', ')
            country_scores = country_scores.explode('genre')
            
            # Calculate average scores
            genre_scores = country_scores.groupby('genre').agg(
                avg_score=('score', 'mean'),
                count=('anime_id', 'count')
            ).reset_index()
            
            # Filter to genres with enough data
            genre_scores = genre_scores[genre_scores['count'] >= 5].sort_values('avg_score', ascending=False)
            
            fig5 = px.bar(
                genre_scores.head(10),
                x='genre',
                y='avg_score',
                color='avg_score',
                labels={'avg_score': 'Average Score', 'genre': 'Genre'},
                title=f"Highest Rated Genres in {selected_country}",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig5, use_container_width=True)

# Add a data table in an expander
with st.expander("📊 View Sample Data"):
    st.dataframe(sampled_merged.head(1000))

# Download option
if 'genre_region' in locals():
    csv = genre_region.to_csv(index=False)
    st.download_button(
        label="Download Genre-Region Data as CSV",
        data=csv,
        file_name="anime_genre_by_region.csv",
        mime="text/csv"
    )
