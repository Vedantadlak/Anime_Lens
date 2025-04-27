import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import shap

# Page configuration
st.set_page_config(
    page_title="Anime Success Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        background-color: #000;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #000;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c78a8;
        color: white;
    }
    h1, h2, h3 {
        color: #1e1e1e;
    }
    .highlight {
        background-color: #000;
        border-radius: 4px;
        padding: 20px;
        border-left: 3px solid #4c78a8;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🎯 Interactive Anime Success Prediction")
st.markdown("""
<div class="highlight">
This app uses machine learning to predict whether an anime will be successful based on its characteristics. 
Success is defined as having a score of 7.5 or higher on MyAnimeList.
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/anime_cleaned.csv")
    return df

df_anime = load_data()

# Data preprocessing
def preprocess_data(df):
    # Define success threshold
    df['successful'] = df['score'] >= 7
    
    # Handle missing values
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').fillna(0)
    df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce').fillna(0)
    df['aired_from_year'] = pd.to_numeric(df['aired_from_year'], errors='coerce').fillna(2000)
    df['members'] = pd.to_numeric(df['members'], errors='coerce').fillna(0)
    
    # Create log transformations for skewed features
    df['log_members'] = np.log1p(df['members'])
    
    # Genre encoding
    df['genre'] = df['genre'].fillna('Unknown').astype(str).str.split(', ')
    genres_dummies = df['genre'].explode().str.get_dummies().groupby(level=0).sum()
    
    # Studio encoding (only top studios)
    df['studio'] = df['studio'].fillna('Unknown')
    top_studios = df['studio'].value_counts().nlargest(20).index
    df['studio'] = df['studio'].where(df['studio'].isin(top_studios), 'Other')
    studio_dummies = pd.get_dummies(df['studio'], prefix='studio')
    
    # Feature creation
    if 'favorites' in df.columns and 'members' in df.columns:
        df['fav_member_ratio'] = df['favorites'] / df['members']
    
    # Select base features
    base_features = df[['episodes', 'duration_min', 'aired_from_year', 'log_members']]
    if 'fav_member_ratio' in df.columns:
        base_features['fav_member_ratio'] = df['fav_member_ratio']
    
    # Combine all features
    features = pd.concat([
        base_features,
        genres_dummies,
        studio_dummies
    ], axis=1)
    
    # Handle any remaining NaN values
    features = features.fillna(0)
    
    return features, df['successful']

# Main model building function
def build_model(X_train, X_test, y_train, y_test):
    # Model training
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    threshold = 0.6 
    y_pred = (y_proba >= threshold).astype(int)
    
    return clf, y_pred, y_proba

# Execute preprocessing
features, labels = preprocess_data(df_anime)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf, y_pred, y_proba = build_model(X_train, X_test, y_train, y_test)

# Create tabs for organization
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔮 Make Predictions", "💡 Feature Importance"])

with tab1:
    st.header("Model Performance")
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    accuracy = report['accuracy']
    precision = report['True']['precision']
    recall = report['True']['recall']
    f1 = report['True']['f1-score']

    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Accuracy", f"{accuracy:.2f}")
    metrics_cols[1].metric("Precision", f"{precision:.2f}")
    metrics_cols[2].metric("Recall", f"{recall:.2f}")
    metrics_cols[3].metric("F1 Score", f"{f1:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix heatmap
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Not Successful', 'Successful'],
        y=['Not Successful', 'Successful'],
        colorscale='Blues',
        showscale=False,
        text=cm,
        texttemplate="%{text}",
    ))
    
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(side="bottom"),
        width=500, 
        height=400,
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    
    fig_roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig_roc.update_layout(
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    
    st.plotly_chart(fig_roc)
    
    # Cross-validation scores
    st.subheader("Cross-Validation Performance")
    cv_scores = cross_val_score(clf, features, labels, cv=5)
    
    fig_cv = go.Figure(data=[
        go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores,
            marker_color='royalblue'
        )
    ])
    
    fig_cv.add_shape(
        type="line",
        x0=-0.5,
        x1=len(cv_scores)-0.5,
        y0=cv_scores.mean(),
        y1=cv_scores.mean(),
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig_cv.add_annotation(
        x=len(cv_scores)-1,
        y=cv_scores.mean(),
        text=f"Mean: {cv_scores.mean():.3f}",
        showarrow=True,
        arrowhead=2,
    )
    
    fig_cv.update_layout(
        title="Cross-Validation Accuracy Scores",
        xaxis_title="Validation Fold",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0.5, 1]),
    )
    
    st.plotly_chart(fig_cv)

with tab2:
    st.header("Make Your Own Predictions")
    st.markdown("Adjust the parameters below to predict if an anime with these characteristics would be successful.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic anime characteristics
        episodes = st.slider("Number of Episodes", 1, 500, 12)
        duration = st.slider("Episode Duration (minutes)", 1, 60, 24)
        year = st.slider("Release Year", 1990, 2025, 2020)
        members = st.slider("Expected Member Count", 100, 1000000, 50000)
        
        # Calculate log members
        log_members = np.log1p(members)
        
        # Genre selection
        all_genres = [col for col in features.columns if col not in [
            'episodes', 'duration_min', 'aired_from_year', 'log_members', 'fav_member_ratio'
        ] and not col.startswith('studio_')]
        
        selected_genres = st.multiselect("Select Genres", all_genres, default=["Action"])
    
    with col2:
        # Studio selection
        studio_cols = [col for col in features.columns if col.startswith('studio_')]
        studios = [col.replace('studio_', '') for col in studio_cols]
        selected_studio = st.selectbox("Select Studio", studios)
        
        # Optional: Favorite to member ratio if available
        if 'fav_member_ratio' in features.columns:
            fav_ratio = st.slider("Favorites to Member Ratio", 0.0, 0.5, 0.05, 0.01)
        else:
            fav_ratio = 0.05
            
        # Prediction button
        predict_btn = st.button("Predict Success", type="primary", use_container_width=True)
        
        # Prepare input data
        if predict_btn:
            # Create input dataframe with same structure as training features
            input_data = pd.DataFrame(0, index=[0], columns=features.columns)
            
            # Set values
            input_data.loc[0, 'episodes'] = episodes
            input_data.loc[0, 'duration_min'] = duration
            input_data.loc[0, 'aired_from_year'] = year
            input_data.loc[0, 'log_members'] = log_members
            
            if 'fav_member_ratio' in input_data.columns:
                input_data.loc[0, 'fav_member_ratio'] = fav_ratio
                
            # Set genres
            for genre in selected_genres:
                if genre in input_data.columns:
                    input_data.loc[0, genre] = 1
                    
            # Set studio
            studio_col = f"studio_{selected_studio}"
            if studio_col in input_data.columns:
                input_data.loc[0, studio_col] = 1
                
            # Make prediction
            prediction = clf.predict(input_data)[0]
            probability = clf.predict_proba(input_data)[0][1]
            
            # Display result
            result_container = st.container()
            
            with result_container:
                if prediction:
                    st.success(f"Likely to be successful with {probability:.1%} confidence!")
                else:
                    st.error(f"Not likely to be successful. Only {probability:.1%} confidence.")
                
                # Gauge chart for probability visualization
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "royalblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    title = "Success Probability",
                    height = 300,
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)

with tab3:
    st.header("Feature Importance Analysis")
    
    # Get feature importances
    importances = pd.Series(clf.feature_importances_, index=features.columns)
    top_importances = importances.nlargest(15)
    
    # Plot feature importances
    fig_imp = px.bar(
        x=top_importances.values,
        y=top_importances.index,
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        title='Top 15 Feature Importances'
    )
    
    fig_imp.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500
    )
    
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Feature importance explanation
    st.markdown("""
    ### Understanding Feature Importance
    
    - **Episodes & Duration**: Shows how episode count and length influence success
    - **Genre Features**: Indicates which genres are associated with higher success rates
    - **Studio Features**: Reveals which studios have track records of successful anime
    - **Year**: Shows how release timing affects success
    - **Member Count**: Indicates the relationship between popularity and quality rating
    """)
    
    # Feature correlations with success
    st.subheader("Feature Correlations with Success")
    
    # Add success to features for correlation
    corr_df = pd.concat([features, labels], axis=1)
    
    # Calculate correlations with success
    success_corr = corr_df.corr()['successful'].sort_values(ascending=False).drop('successful')
    top_pos_corr = success_corr.nlargest(10)
    top_neg_corr = success_corr.nsmallest(10)
    
    # Create subplots for positive and negative correlations
    fig_corr = make_subplots(rows=1, cols=2, subplot_titles=["Positive Correlations with Success", "Negative Correlations with Success"])
    
    fig_corr.add_trace(
        go.Bar(
            x=top_pos_corr.values,
            y=top_pos_corr.index,
            orientation='h',
            marker_color='green',
            name="Positive"
        ),
        row=1, col=1
    )
    
    fig_corr.add_trace(
        go.Bar(
            x=top_neg_corr.values,
            y=top_neg_corr.index,
            orientation='h',
            marker_color='red',
            name="Negative"
        ),
        row=1, col=2
    )
    
    fig_corr.update_layout(
        height=500,
        title_text="Correlations between Features and Success",
        showlegend=False
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Add sidebar with additional information
with st.sidebar:
    st.header("About this Model")
    st.markdown("""
    **Model Type**: Random Forest Classifier
    
    **Training Data**: MyAnimeList dataset with 
    anime released between 2000 and 2021
    
    **Success Definition**: MAL score ≥ 7.5
    
    **Feature Categories**:
    - Basic metrics (episodes, duration)
    - Content descriptors (genres)
    - Production information (studio)
    - Popularity indicators (member count)
    
    This model helps studios predict potential success
    before committing to full production.
    """)

    
    # Model parameters
    st.markdown("### Model Parameters")
    st.code("""
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
    """)
