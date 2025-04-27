import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(layout="wide", page_title="Anime Genre Evolution", page_icon="📊")

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

# Page title
st.title("🌟 Anime Genre Evolution Explorer")
st.markdown("""
<div class="highlight">
Explore how different anime genres have evolved in popularity over time.
Compare trends, discover patterns, and identify rising genres in anime history.
</div>
""", unsafe_allow_html=True)

# Loading and preprocessing the data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/anime_cleaned.csv")
    df = df.dropna(subset=['aired_from_year', 'genre'])
    df['aired_from_year'] = df['aired_from_year'].astype(int)
    df['genre'] = df['genre'].str.split(', ')
    
    for col in ['score', 'popularity', 'members']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

df = load_data()
df_exploded = df.explode('genre')

# Different Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Analysis", "📊 Yearly Comparison", "🔥 Heatmap View", "⚖️ Genre Growth"])

st.sidebar.header("📋 Visualization Controls")

# Year range filter
min_year, max_year = int(df_exploded['aired_from_year'].min()), int(df_exploded['aired_from_year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Genre selection with search
unique_genres = sorted(df_exploded['genre'].dropna().unique())
selected_genres = st.sidebar.multiselect(
    "Select Genres",
    unique_genres,
    default=["Action", "Romance", "Comedy"]
)
# Display options
normalize = st.sidebar.checkbox("Normalize by Total Anime Count", False, 
                               help="Show percentage of total anime instead of raw counts")


# Color options
color_theme = st.sidebar.selectbox("Color Theme", 
                                  ["Default", "Viridis", "Plasma", "Blues", "Rainbow"])
theme_map = {
    "Default": None,
    "Viridis": px.colors.sequential.Viridis,
    "Plasma": px.colors.sequential.Plasma,
    "Blues": px.colors.sequential.Blues,
    "Rainbow": px.colors.sequential.Rainbow
}

# Filter data based on selections
filtered = df_exploded[
    (df_exploded['aired_from_year'] >= year_range[0]) &
    (df_exploded['aired_from_year'] <= year_range[1]) &
    (df_exploded['genre'].isin(selected_genres))
]

# Prepare data for visualization
genre_trend = (
    filtered.groupby(['aired_from_year', 'genre'])['title']
    .count()
    .reset_index()
    .rename(columns={'title': 'count'})
)

# Apply normalization if selected
if normalize:
    # Get total anime per year
    total_per_year = filtered.groupby('aired_from_year')['title'].count().reset_index()
    total_per_year.columns = ['aired_from_year', 'total']
    
    # Merge with genre counts
    genre_trend = pd.merge(genre_trend, total_per_year, on='aired_from_year')
    
    # Calculate percentage
    genre_trend['percentage'] = (genre_trend['count'] / genre_trend['total']) * 100
    value_column = 'percentage'
    value_label = 'Percentage of Anime (%)'
else:
    value_column = 'count'
    value_label = 'Number of Anime Released'

# Tab 1: Trend Line Chart
with tab1:
    st.header("Genre Popularity Trends Over Time")
    
   
    plot_column = value_column
    
    # Create trend line chart
    fig = px.line(
        genre_trend,
        x='aired_from_year',
        y=plot_column,
        color='genre',
        markers=True,
        labels={'aired_from_year': 'Year', plot_column: value_label},
        title='Anime Genre Popularity Trends',
        color_discrete_sequence=theme_map[color_theme],
        hover_data=['count'] if normalize else None,
    )
    
    fig.update_layout(
        hovermode="x unified",
        legend_title_text='Genre',
        template='plotly_white',
        height=600,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics below the chart
    stats_cols = st.columns(len(selected_genres))
    
    for i, genre in enumerate(selected_genres):  # Limit to 4 stats boxes
        genre_data = genre_trend[genre_trend['genre'] == genre]
        if not genre_data.empty:
            latest_year = genre_data['aired_from_year'].max()
            latest_count = genre_data[genre_data['aired_from_year'] == latest_year][value_column].values[0]
            
            # Calculate growth vs previous year
            if latest_year > year_range[0]:
                prev_year_data = genre_data[genre_data['aired_from_year'] == latest_year - 1]
                if not prev_year_data.empty:
                    prev_count = prev_year_data[value_column].values[0]
                    yoy_change = ((latest_count / prev_count) - 1) * 100
                    stats_cols[i % len(selected_genres)].metric(f"{genre}", 
                                           f"{int(latest_count)}", 
                                           f"{yoy_change:+.1f}% vs. prev. year")
                else:
                    stats_cols[i % len(selected_genres)].metric(f"{genre}", 
                                           f"{int(latest_count)}")
            else:
                stats_cols[i % len(selected_genres)].metric(f"{genre}", 
                                     f"{int(latest_count)}")

# Tab 2: Yearly Comparison Bar Chart
with tab2:
    st.header("Genre Comparison by Year")
    
    # Select specific years to compare
    num_years = st.slider("Number of years to compare", 2, 10, 2)
    available_years = sorted(filtered['aired_from_year'].unique())
    
    # Try to select last N years, or as many as available
    default_years = available_years[-num_years:] if len(available_years) >= num_years else available_years
    
    selected_years = st.multiselect(
        "Select years to compare",
        options=available_years,
        default=default_years
    )
    
    if selected_years:
        # Filter for selected years
        year_genre_data = genre_trend[genre_trend['aired_from_year'].isin(selected_years)]

        # Ensure years are treated as string (categorical)
        year_genre_data['aired_from_year'] = year_genre_data['aired_from_year'].astype(str)

        # Normalize data if selected
        if normalize:
            total_per_year = year_genre_data.groupby('aired_from_year')[value_column].transform('sum')
            year_genre_data[value_column] = (year_genre_data[value_column] / total_per_year) * 100

        # Sort years manually for x-axis order
        year_genre_data['aired_from_year'] = pd.Categorical(
            year_genre_data['aired_from_year'],
            categories=[str(y) for y in sorted(selected_years)],
            ordered=True
        )

        # Grouped bar chart with genres side-by-side per year
        bar_fig = px.bar(
            year_genre_data,
            x='aired_from_year',
            y=value_column,
            color='genre',
            barmode='group',
            labels={
                'aired_from_year': 'Year',
                'genre': 'Genre',
                value_column: value_label
            },
            title='Genre Distribution by Year',
            color_discrete_sequence=theme_map[color_theme],
            height=500
        )

        bar_fig.update_layout(
            template='plotly_white',
            xaxis_title='Year',
            yaxis_title=value_label,
            legend_title='Genre'
        )

        st.plotly_chart(bar_fig, use_container_width=True)
        
        # Show percentage change table
        if len(selected_years) >= 2:
            st.subheader("Genre Change Analysis")
            
            # Sort years
            selected_years = sorted(selected_years)
            start_year = str(selected_years[0])
            end_year = str(selected_years[-1])
            
            # Get data for first and last year
            start_data = year_genre_data[year_genre_data['aired_from_year'] == start_year]
            end_data = year_genre_data[year_genre_data['aired_from_year'] == end_year]
            
            # Merge to calculate change
            change_df = pd.merge(
                start_data[['genre', value_column]], 
                end_data[['genre', value_column]], 
                on='genre', 
                suffixes=('_start', '_end')
            )
            
            # Calculate change
            change_df['absolute_change'] = change_df[f'{value_column}_end'] - change_df[f'{value_column}_start']
            change_df['percent_change'] = ((change_df[f'{value_column}_end'] / change_df[f'{value_column}_start']) - 1) * 100
            
            # Display the change table
            change_display = change_df.sort_values('percent_change', ascending=False).copy()
            change_display.columns = ['Genre', f'{start_year} Value', f'{end_year} Value', 'Change', '% Change']
            
            # Format percentage
            change_display['% Change'] = change_display['% Change'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(change_display, use_container_width=True)
    else:
        st.warning("Please select at least one year to display the comparison.")

# Tab 3: Heatmap View
with tab3:
    st.header("Genre Popularity Heatmap")
    
    # Create pivot table for heatmap
    pivot_data = genre_trend.pivot(index='genre', columns='aired_from_year', values=value_column)
    
    # Fill NA values with 0
    pivot_data = pivot_data.fillna(0)
    
    # Create heatmap
    heat_fig = px.imshow(
        pivot_data,
        labels=dict(x="Year", y="Genre", color=value_label),
        title="Genre Popularity Heatmap",
        color_continuous_scale="Viridis" if color_theme == "Default" else color_theme.lower(),
        height=600
    )
    
    heat_fig.update_layout(
        template='plotly_white',
        xaxis_nticks=20,
    )
    
    st.plotly_chart(heat_fig, use_container_width=True)
    
    # Show top genre for each year
    st.subheader("Dominant Genre by Year")
    
    # Calculate top genre for each year
    top_by_year = pd.DataFrame({
        'Year': pivot_data.columns,
        'Top Genre': pivot_data.idxmax()
    })
    
    # Get counts of years a genre was dominant
    genre_dominance = top_by_year['Top Genre'].value_counts().reset_index()
    genre_dominance.columns = ['Genre', 'Years as Dominant']
    
    # Create horizontal bar chart
    dom_fig = px.bar(
        genre_dominance.head(10),
        y='Genre',
        x='Years as Dominant',
        color='Genre',
        orientation='h',
        labels={'Years as Dominant': 'Number of Years as Dominant Genre'},
        title='Genres with Most Years as Dominant',
        color_discrete_sequence=theme_map[color_theme],
        height=400
    )
    
    dom_fig.update_layout(
    template='plotly_white',
    yaxis={'categoryorder': 'total descending'},  
    margin=dict(l=20, r=20, t=40, b=20)
)
    
    st.plotly_chart(dom_fig, use_container_width=True)

# Tab 4: Genre Growth Analysis
with tab4:
    st.header("Genre Growth Analysis")
    
    # Calculate growth between first and last year
    growth_data = []
    
    for genre in selected_genres:
        genre_data = genre_trend[genre_trend['genre'] == genre].sort_values('aired_from_year')
        
        if len(genre_data) >= 2:
            # Get first and last data points
            first_year = genre_data['aired_from_year'].min()
            last_year = genre_data['aired_from_year'].max()
            
            first_value = genre_data[genre_data['aired_from_year'] == first_year][value_column].values[0]
            last_value = genre_data[genre_data['aired_from_year'] == last_year][value_column].values[0]
            
            # Calculate compound annual growth rate
            years_diff = last_year - first_year
            if years_diff > 0 and first_value > 0:
                cagr = (((last_value / first_value) ** (1 / years_diff)) - 1) * 100
            else:
                cagr = None
                
            growth_data.append({
                'Genre': genre,
                'First Year': first_year,
                'Last Year': last_year,
                'Initial Value': first_value,
                'Final Value': last_value,
                'Change': last_value - first_value,
                'CAGR (%)': cagr
            })
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        
        # Sort by growth rate
        growth_df = growth_df.sort_values('CAGR (%)', ascending=False)
        
        # Display as a table
        st.dataframe(growth_df, use_container_width=True)
        
        # Visualize the growth rates
        growth_fig = px.bar(
            growth_df,
            y='Genre',
            x='CAGR (%)',
            color='Genre',
            orientation='h',
            labels={'CAGR (%)': 'Compound Annual Growth Rate (%)'},
            title='Genre Growth Rates',
            color_discrete_sequence=theme_map[color_theme],
            height=500
        )
        
        growth_fig.update_layout(
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(growth_fig, use_container_width=True)
        
        # Create a bubble chart showing initial value, final value, and growth
        bubble_fig = px.scatter(
            growth_df,
            x='Initial Value',
            y='Final Value',
            size=growth_df['Change'].abs(),
            color='Genre',
            size_max=50,
            text='Genre',
            hover_name='Genre',
            labels={
                'Initial Value': f'Initial Value ({year_range[0]}-{first_year})',
                'Final Value': f'Final Value ({last_year}-{year_range[1]})'
            },
            title='Genre Growth Bubble Chart',
            height=600
        )
        
        # Add reference line (y=x)
        bubble_fig.add_shape(
            type="line",
            x0=growth_df['Initial Value'].min(),
            y0=growth_df['Initial Value'].min(),
            x1=growth_df['Final Value'].max(),
            y1=growth_df['Final Value'].max(),
            line=dict(color="gray", width=1, dash="dash")
        )
        
        bubble_fig.update_layout(
            template='plotly_white'
        )
        
        st.plotly_chart(bubble_fig, use_container_width=True)
    else:
        st.warning("Not enough data to calculate growth rates. Try selecting more genres or a wider year range.")

# Data table in expander
with st.expander("📊 View and Download Data"):
    st.dataframe(genre_trend, use_container_width=True)
    
    # Add download button
    csv = genre_trend.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="anime_genre_trends.csv",
        mime="text/csv"
    )
