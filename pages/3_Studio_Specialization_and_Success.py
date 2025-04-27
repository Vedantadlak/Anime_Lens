import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🎥 Anime Studio Insights Dashboard")

df_anime = pd.read_csv("./data/anime_cleaned.csv")
df_genre_studio = df_anime[['studio', 'genre']].dropna()
df_genre_studio['genre'] = df_genre_studio['genre'].str.split(',')
df_genre_studio = df_genre_studio.explode('genre')
df_genre_studio['genre'] = df_genre_studio['genre'].str.strip()
df_genre_studio['studio'] = df_genre_studio['studio'].str.strip()
studio_genre_counts = df_genre_studio.groupby(['genre', 'studio']).size().reset_index(name='count')

st.header("🏆 Top Studios by Genre")
top_n = st.slider("Select Top N Studios per Genre", 1, 10, 3)
genres = st.multiselect("Pick genres to show:", sorted(studio_genre_counts['genre'].unique()), default=['Action', 'Romance', 'Comedy'])
top_studios_by_genre = studio_genre_counts.sort_values(['genre', 'count'], ascending=[True, False])
top_studios = top_studios_by_genre.groupby('genre').head(top_n)
filtered = top_studios[top_studios['genre'].isin(genres)]

fig = px.bar(
    filtered,
    x='count',
    y='studio',
    color='genre',
    facet_col='genre',
    orientation='h',
    title="Top Studios by Genre",
    labels={'count': 'Anime Count', 'studio': 'Studio'}
)
fig.update_layout(template='plotly_white', showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.header("🔥 Heatmap: Anime Counts by Studio and Genre")
heatmap_data = studio_genre_counts.pivot(index='studio', columns='genre', values='count').fillna(0)
top_heatmap_studios = heatmap_data.sum(axis=1).sort_values(ascending=False).head(10).index
heatmap_data = heatmap_data.loc[top_heatmap_studios]

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu'
    )
)
fig_heatmap.update_layout(
    title="Top 10 Studios: Genre Spread",
    xaxis_title="Genre",
    yaxis_title="Studio",
    template='plotly_white'
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.header("🌞 Sunburst, Treemap & Sankey Visualizations")
sunburst_df = studio_genre_counts[studio_genre_counts['count'] > 2]
tab1, tab2, tab3 = st.tabs(["Sunburst Chart", "Treemap", "Sankey Diagram"])

with tab1:
    fig = px.sunburst(
        sunburst_df, path=['genre', 'studio'], values='count', color='genre',
        title='Sunburst: Genres → Studios'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.treemap(
        sunburst_df, path=['genre', 'studio'], values='count',
        title='Treemap: Studio Dominance by Genre'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
        # Load and clean data
    anime_df = pd.read_csv('./data/anime_cleaned.csv')
    anime_df['studio'] = anime_df['studio'].fillna('Unknown')
    anime_df.loc[anime_df['studio'].str.strip() == '', 'studio'] = 'Unknown'
    anime_df['genre_list'] = anime_df['genre'].str.split(', ')
    anime_exploded = anime_df.explode('genre_list').dropna(subset=['genre_list'])

    # Top studios and genres
    top_studios = anime_exploded['studio'].value_counts().head(10).index.tolist()
    top_genres = anime_exploded['genre_list'].value_counts().head(6).index.tolist()

    df = anime_exploded[
        anime_exploded['studio'].isin(top_studios) &
        anime_exploded['genre_list'].isin(top_genres)
    ].copy()

    # Rating buckets (10.0 to 0.0)
    df['rating_bucket'] = (df['score'] // 1).astype(int).clip(0, 10).astype(str) + '.0'
    rating_nodes = [f"{i}.0" for i in range(10, -1, -1)]

    # Node order and mapping
    studio_nodes = top_studios
    genre_nodes = top_genres
    all_nodes = studio_nodes + genre_nodes + rating_nodes
    node_map = {name: i for i, name in enumerate(all_nodes)}

    # Pastel color palettes
    studio_colors = [
        '#A2C5F5', '#6B7BD6', '#B7A2F5', '#F5A2D6', '#F5B7A2',
        '#F5E6A2', '#F5F3A2', '#A2F5B7', '#A2F5E6', '#A2F5F3'
    ]
    genre_colors = ['#A2E6F5', '#B7A2F5', '#D6A2F5', '#F5A2E6', '#F5A2B7', '#F5A2A2']
    rating_colors = [
        '#F5CBA2', '#F5E6A2', '#D6F5A2', '#A2F5B7', '#A2F5E6',
        '#A2C5F5', '#A2B7F5', '#B7A2F5', '#D6A2F5', '#F5A2E6', '#F5A2B7'
    ]

    def hex_to_rgba(hex_color, alpha=0.25):
        hex_color = hex_color.lstrip('#')
        return f'rgba({int(hex_color[0:2],16)},{int(hex_color[2:4],16)},{int(hex_color[4:6],16)},{alpha})'

    studio_color_map = {studio: color for studio, color in zip(studio_nodes, studio_colors)}
    genre_color_map = {genre: color for genre, color in zip(genre_nodes, genre_colors)}
    rating_color_map = {rating: color for rating, color in zip(rating_nodes, rating_colors)}

    # Aggregate for Sankey: studio→genre→rating
    studio_genre_rating = df.groupby(['studio', 'genre_list', 'rating_bucket']).size().reset_index(name='count')

    sources, targets, values, link_colors, customdata = [], [], [], [], []

    # Studio→Genre links
    for _, row in studio_genre_rating.iterrows():
        sources.append(node_map[row['studio']])
        targets.append(node_map[row['genre_list']])
        values.append(row['count'])
        link_colors.append(hex_to_rgba(studio_color_map[row['studio']], 0.25))
        customdata.append(f"Studio: {row['studio']}<br>Genre: {row['genre_list']}")

    # Genre→Rating links
    for _, row in studio_genre_rating.iterrows():
        sources.append(node_map[row['genre_list']])
        targets.append(node_map[row['rating_bucket']])
        values.append(row['count'])
        link_colors.append(hex_to_rgba(genre_color_map[row['genre_list']], 0.25))
        customdata.append(f"Genre: {row['genre_list']}<br>Rating: {row['rating_bucket']}<br>Studio: {row['studio']}")

    # Node colors (opaque)
    node_colors = studio_colors + genre_colors + rating_colors
    node_colors_rgba = [hex_to_rgba(c, 0.95) for c in node_colors]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=24,
            line=dict(color="rgba(255,255,255,0.0)", width=1),
            label=all_nodes,
            color=node_colors_rgba,
            hoverlabel=dict(
                bgcolor='white',
                font=dict(color='black', size=16)
            )
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=customdata,
            hovertemplate='%{customdata}<br>Count: %{value}<extra></extra>',
        )
    )])

    fig.update_layout(
    title_text="Studio → Genre → Rating",
    font_size=14,
    height=600,           # Increase height for more vertical space
    width=1200,           # Increase width if needed
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    #margin=dict(l=100, r=100, t=100, b=100)  # Increase all margins
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("🏅 Studios with Consistently High Scores")
df_scores = df_anime[['studio', 'score']].dropna()
df_scores['studio'] = df_scores['studio'].str.strip()
studio_scores = df_scores.groupby('studio').agg(
    avg_score=('score', 'mean'),
    anime_count=('score', 'count')
).reset_index()
min_anime = st.slider("Minimum Anime to Consider", 1, 20, 5)
studio_scores = studio_scores[studio_scores['anime_count'] >= min_anime]
top_studios_by_score = studio_scores.sort_values('avg_score', ascending=False).head(10)
fig = px.bar(
    top_studios_by_score,
    x='avg_score',
    y='studio',
    orientation='h',
    color='avg_score',
    color_continuous_scale='viridis',
    title="Top Studios by Score",
    labels={'avg_score': 'Average Score', 'studio': 'Studio Name'}
)
fig.update_layout(template='plotly_white', showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🫧 Score vs Anime Count")
studio_stats = df_anime.groupby('studio').agg({'score': 'mean', 'anime_id': 'count'}).rename(
    columns={'anime_id': 'anime_count'}).reset_index()
top_studio_stats = studio_stats[studio_stats['anime_count'] > 10]
fig = px.scatter(
    top_studio_stats,
    x='anime_count',
    y='score',
    size='anime_count',
    color='score',
    hover_name='studio',
    title='Bubble Plot: Score vs Anime Count',
    size_max=40,
    color_continuous_scale='Viridis',
    labels={'anime_count': 'Number of Anime', 'score': 'Average Score'}
)
fig.update_layout(template='plotly_white')
st.plotly_chart(fig, use_container_width=True)
