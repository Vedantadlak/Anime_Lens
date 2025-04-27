import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="Anime Seasonal Patterns", page_icon="📅")

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
</style>
""", unsafe_allow_html=True)

# Page title and introduction
st.title("🍂 Anime Seasonal Release Patterns")
st.markdown("""
<div class="highlight">
Explore how anime releases, ratings, and popularity fluctuate across different seasons. 
Discover which seasons bring the highest-rated shows and how seasonal distribution has evolved over time.
</div>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/anime_cleaned.csv")
    # Clean and extract seasonal data
    df = df.dropna(subset=['premiered'])
    df[['season', 'season_year']] = df['premiered'].str.split(' ', expand=True)
    df = df.dropna(subset=['season_year','season'])
    df['season_year'] = df['season_year'].astype(int)
    
    # Convert score and popularity to numeric
    for col in ['score', 'popularity', 'members', 'favorites']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure standard season naming and ordering
    season_order = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    df = df[df['season'].isin(season_order.keys())]
    df['season_order'] = df['season'].map(season_order)
    
    return df

df = load_data()

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Release Trends", "⭐ Ratings Analysis", "🔥 Seasonal Heatmap", "📊 Comparative View"])

# Sidebar filters
st.sidebar.header("📋 Filter Options")

# Year range filter
min_year = int(df['season_year'].min())
max_year = int(df['season_year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Season selection
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons)

# Additional filters
if 'genre' in df.columns:
    all_genres = sorted(set([genre for genres in df['genre'].dropna() for genre in genres.split(', ')]))
    selected_genres = st.sidebar.multiselect("Filter by Genre", all_genres)
    
    if selected_genres:
        genre_filter = df['genre'].apply(lambda x: any(genre in str(x).split(', ') for genre in selected_genres) if pd.notnull(x) else False)
        df = df[genre_filter]



# Filter dataset based on selections
filtered_df = df[
    (df['season_year'] >= year_range[0]) &
    (df['season_year'] <= year_range[1]) &
    (df['season'].isin(selected_seasons))
]

# Calculate aggregates for visualization
release_trend = (
    filtered_df.groupby(['season_year', 'season', 'season_order'])['title']
    .count()
    .reset_index()
    .rename(columns={'title': 'anime_count'})
    .sort_values(['season_year', 'season_order'])
)

seasonal_scores = (
    filtered_df.groupby(['season_year', 'season'])['score']
    .mean()
    .reset_index()
    .sort_values(['season_year', 'season'])
)

seasonal_popularity = (
    filtered_df.groupby(['season_year', 'season'])['popularity']
    .mean()
    .reset_index()
    .sort_values(['season_year', 'season'])
)

# Seasonal statistics for the entire period
overall_seasonal_stats = filtered_df.groupby('season').agg(
    avg_score=('score', 'mean'),
    avg_popularity=('popularity', 'mean'),
    total_anime=('title', 'count')
).reset_index()

# TAB 1: Release Trends
with tab1:
    st.header("Anime Releases by Season Over Time")
    
    normalize = st.checkbox("Normalize by Year (Show Percentage)", False)
    
    if normalize:
        yearly_totals = release_trend.groupby('season_year')['anime_count'].sum().reset_index()
        release_trend = pd.merge(release_trend, yearly_totals, on='season_year', suffixes=('', '_total'))
        release_trend['percentage'] = (release_trend['anime_count'] / release_trend['anime_count_total']) * 100
        y_value = 'percentage'
        y_label = 'Percentage of Yearly Releases'
    else:
        y_value = 'anime_count'
        y_label = 'Number of Anime Released'

    # Create time-series chart 
    fig = px.line(
        release_trend, 
        x='season_year', 
        y=y_value,
        color='season',
        markers=True,
        labels={'season_year': 'Year', y_value: y_label, 'season': 'Season'},
        title=f'Seasonal Anime Releases ({year_range[0]}-{year_range[1]})'
    )
    
    fig.update_layout(
        hovermode="x unified",
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal distribution by year comparison
    st.subheader("Seasonal Distribution by Year")
    
    # Select specific years to compare
    comparison_years = st.multiselect(
        "Select years to compare",
        options=sorted(filtered_df['season_year'].unique()),
        default=list(sorted(filtered_df['season_year'].unique())[-3:]) if len(filtered_df['season_year'].unique()) >= 3 else list(sorted(filtered_df['season_year'].unique()))
    )
    
    if comparison_years:
        # Filter for selected years
        comparison_data = release_trend[release_trend['season_year'].isin(comparison_years)]
        comparison_data['season_year'] = comparison_data['season_year'].astype(str)

        
        # Create grouped bar chart
        fig_comparison = px.bar(
            comparison_data,
            x='season_year',
            y='anime_count',
            color='season',
            barmode='group',
            labels={'season_year': 'Year', 'anime_count': 'Number of Anime Released', 'season': 'Season'},
            title=f'Seasonal Release Comparison for Selected Years',
            category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
        )
        
        fig_comparison.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)

# TAB 2: Ratings Analysis
with tab2:
    st.header("Seasonal Ratings Analysis")
    
    # Create 2 column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall average score by season
        avg_score = overall_seasonal_stats[['season', 'avg_score']].sort_values('avg_score', ascending=False)
        
        fig_score = px.bar(
            avg_score,
            x='season',
            y='avg_score',
            color='season',
            labels={'season': 'Season', 'avg_score': 'Average Score'},
            title=f'Average Anime Score by Season ({year_range[0]}-{year_range[1]})',
            category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
        )
        
        fig_score.update_layout(template='plotly_white', showlegend=False)
        fig_score.update_yaxes(range=[avg_score['avg_score'].min() - 0.1, avg_score['avg_score'].max() + 0.1])
        
        st.plotly_chart(fig_score, use_container_width=True)
    
    with col2:
        # Average popularity by season (lower is better)
        avg_popularity = overall_seasonal_stats[['season', 'avg_popularity']].sort_values('avg_popularity')
        
        fig_pop = px.bar(
            avg_popularity,
            x='season',
            y='avg_popularity',
            color='season',
            labels={'season': 'Season', 'avg_popularity': 'Average Popularity Rank (Lower is Better)'},
            title=f'Average Popularity Rank by Season ({year_range[0]}-{year_range[1]})',
            category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
        )
        
        fig_pop.update_layout(template='plotly_white', showlegend=False)
        
        st.plotly_chart(fig_pop, use_container_width=True)
    
    # Score trends over time
    st.subheader("Season Score Trends Over Time")
    
    fig_score_trend = px.line(
        seasonal_scores,
        x='season_year',
        y='score',
        color='season',
        markers=True,
        labels={'season_year': 'Year', 'score': 'Average Score', 'season': 'Season'},
        title=f'Average Score by Season Over Time',
        category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
    )
    
    fig_score_trend.update_layout(
        hovermode="x unified",
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_score_trend, use_container_width=True)

    # with popularity
    fig_score_trend = px.line(
        seasonal_popularity,
        x='season_year',
        y='popularity',
        color='season',
        markers=True,
        labels={'season_year': 'Year', 'popularity': 'Average Popularity(Ranking)', 'season': 'Season'},
        title=f'Average Popularity(Ranking) by Season Over Time',
        category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
    )
    
    fig_score_trend.update_layout(
        hovermode="x unified",
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_score_trend, use_container_width=True)
    
    # Add insights about seasonal trends
    insights = filtered_df.groupby('season').agg(
        avg_episodes=('episodes', 'mean'),
        avg_members=('members', 'mean') if 'members' in filtered_df.columns else ('score', 'count'),
        top_score=('score', 'max'),
        bottom_score=('score', 'min')
    ).reset_index()
    
    st.subheader("Seasonal Insights")
    st.dataframe(insights, use_container_width=True)

# TAB 3: Seasonal Heatmap
with tab3:
    st.header("Seasonal Release Patterns Heatmap")
    
    # Create a pivot table for the heatmap
    pivot_data = release_trend.pivot_table(
        values='anime_count',
        index='season',
        columns='season_year'
    ).fillna(0)
    
    # Reorder seasons
    pivot_data = pivot_data.reindex(["Winter", "Spring", "Summer", "Fall"])
    
    # Create heatmap
    fig_heatmap = px.imshow(
        pivot_data,
        labels=dict(x="Year", y="Season", color="Anime Count"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Anime Releases by Season and Year (Heatmap)"
    )
    
    fig_heatmap.update_layout(
        xaxis_nticks=len(pivot_data.columns),
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Create score heatmap if data available
    
    if 'score' in filtered_df.columns:
        st.subheader("Seasonal Score Heatmap")
        
        # Create a pivot table for scores
        score_pivot = filtered_df.pivot_table(
            values='score',
            index='season',
            columns='season_year',
            aggfunc='mean'
        )
        
        # Reorder seasons
        score_pivot = score_pivot.reindex(["Winter", "Spring", "Summer", "Fall"])
        
        # Create heatmap
        fig_score_heatmap = px.imshow(
            score_pivot,
            labels=dict(x="Year", y="Season", color="Average Score"),
            x=score_pivot.columns,
            y=score_pivot.index,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Average Anime Score by Season and Year"
        )
        
        fig_score_heatmap.update_layout(
            xaxis_nticks=len(score_pivot.columns),
            template='plotly_white',
            height=500,
            coloraxis_colorbar=dict(title="Avg Score")
        )
        
        st.plotly_chart(fig_score_heatmap, use_container_width=True)

    if 'popularity' in filtered_df.columns:
        st.subheader("Seasonal Popularity(Ranking) Heatmap")
        
        # Create a pivot table for scores
        popularity_pivot = filtered_df.pivot_table(
            values='popularity',
            index='season',
            columns='season_year',
            aggfunc='mean'
        )
        
        # Reorder seasons
        popularity_pivot = popularity_pivot.reindex(["Winter", "Spring", "Summer", "Fall"])
        
        # Create heatmap
        fig_score_heatmap = px.imshow(
            popularity_pivot,
            labels=dict(x="Year", y="Season", color="Average Score"),
            x=popularity_pivot.columns,
            y=popularity_pivot.index,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Average Popularity(Ranking) by Season and Year"
        )
        
        fig_score_heatmap.update_layout(
            xaxis_nticks=len(popularity_pivot.columns),
            template='plotly_white',
            height=500,
            coloraxis_colorbar=dict(title="Avg Popularity(Ranking)")
        )
        
        st.plotly_chart(fig_score_heatmap, use_container_width=True)

# TAB 4: Comparative View
with tab4:
    st.header("Cross-Seasonal Analysis")
    
    # Create comparative box plots for scores
    fig_box = px.box(
        filtered_df,
        x='season',
        y='score',
        color='season',
        notched=True,
        points="all",
        labels={'season': 'Season', 'score': 'Score Distribution'},
        title=f'Score Distribution by Season ({year_range[0]}-{year_range[1]})',
        category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
    )
    
    fig_box.update_layout(
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_box, use_container_width=True)

    # with popularity
    fig_box = px.box(
        filtered_df,
        x='season',
        y='popularity',
        color='season',
        notched=True,
        points="all",
        labels={'season': 'Season', 'score': 'Score Distribution'},
        title=f'Popularity(Ranking) Distribution by Season ({year_range[0]}-{year_range[1]})',
        category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]}
    )
    
    fig_box.update_layout(
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Season comparison radar chart
    st.subheader("Season Comparison Radar Chart")
    
    # Prepare data for radar chart
    radar_stats = overall_seasonal_stats.copy()
    
    # Normalize values for radar chart (between 0 and 1)
    for col in ['avg_score', 'total_anime']:
        radar_stats[f'{col}_norm'] = (radar_stats[col] - radar_stats[col].min()) / (radar_stats[col].max() - radar_stats[col].min())
    
    # For popularity, lower is better, so invert
    if 'avg_popularity' in radar_stats.columns:
        radar_stats['avg_popularity_norm'] = 1 - (radar_stats['avg_popularity'] - radar_stats['avg_popularity'].min()) / (radar_stats['avg_popularity'].max() - radar_stats['avg_popularity'].min())
    
    # Create radar chart
    fig_radar = go.Figure()
    
    for season in radar_stats['season']:
        season_data = radar_stats[radar_stats['season'] == season]
        fig_radar.add_trace(go.Scatterpolar(
            r=[season_data['avg_score_norm'].values[0], 
               season_data['total_anime_norm'].values[0], 
               season_data['avg_popularity_norm'].values[0] if 'avg_popularity_norm' in season_data.columns else 0],
            theta=['Average Score', 'Number of Releases', 'Popularity (inverted)'],
            fill='toself',
            name=season
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Season Performance Comparison (Normalized)"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Top anime by season
    st.subheader("Top Anime by Season")
    
    selected_season = st.selectbox("Select a Season", ["Winter", "Spring", "Summer", "Fall"])
    
    top_season_anime = filtered_df[filtered_df['season'] == selected_season].sort_values('score', ascending=False).head(10)
    
    if len(top_season_anime) > 0:
        top_cols = ['title', 'score', 'season_year']
        if 'popularity' in top_season_anime.columns:
            top_cols.append('popularity')
        if 'members' in top_season_anime.columns:
            top_cols.append('members')
            
        st.dataframe(top_season_anime[top_cols], use_container_width=True)
        
        # Bar chart of top anime
        fig_top = px.bar(
            top_season_anime,
            x='score',
            y='title',
            color='season_year',
            orientation='h',
            labels={'title': 'Anime Title', 'score': 'Score', 'season_year': 'Year'},
            title=f"Top {len(top_season_anime)} {selected_season} Anime by Score"
        )
        
        fig_top.update_layout(
            template='plotly_white',
            height=500,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.write(f"No data available for {selected_season} season with current filters.")

# Footer with download option
st.markdown("---")
with st.expander("📊 View and Download Data"):
    st.dataframe(filtered_df[['title', 'season', 'season_year', 'score', 'popularity'] + 
                           (['members', 'favorites'] if all(col in filtered_df.columns for col in ['members', 'favorites']) else [])], 
                use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="anime_seasonal_data.csv",
        mime="text/csv"
    )
