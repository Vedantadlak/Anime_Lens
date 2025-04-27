import streamlit as st
import base64

def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local image
image_path = "utils/anime-style-character-space.jpg"
bg_image_base64 = get_base64_of_bin_file(image_path)

# Inject CSS with base64 background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        transition: background 0.5s;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True
) 


st.markdown(
    f"""
    <style>
    .stApp {{

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        transition: background 0.5s;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }}
    /* Glassmorphism effect for main container */
    .anime-glass {{
        background: rgba(255, 255, 255, 0.80);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-top: 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.25);
        border: 1.5px solid rgba(255, 255, 255, 0.35);
        max-width: 680px;
        margin-left: auto;
        margin-right: auto;
    }}
    /* Anime-style header */
    .anime-title {{
        font-family: 'Luckiest Guy', 'Segoe UI', cursive;
        font-size: 2.8rem;
        color: #e75480;
        text-shadow: 2px 2px 10px #fff6, 0 2px 8px #e75480;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        text-align: center;
    }}
    .anime-subtitle {{
        font-size: 1.25rem;
        color: #333;
        text-align: center;
        margin-bottom: 1.3rem;
    }}
    .anime-badge {{
        display: inline-block;
        background: #f9c5d1;
        color: #a8325e;
        border-radius: 18px;
        padding: 0.4em 1.1em;
        font-weight: bold;
        font-size: 1.1em;
        margin: 0.5em 0.2em;
        box-shadow: 0 1px 6px #e7548033;
    }}
    </style>
    <!-- Google Fonts for anime style -->
    <link href="https://fonts.googleapis.com/css2?family=Luckiest+Guy&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

# --- Main content inside a glassmorphic container ---
st.markdown(
    """
    <div class="anime-glass">
        <div class="anime-title">AnimeLens</div>
        <div class="anime-subtitle">
            Dive into the world of anime with interactive data visualizations and ML-powered insights.<br>
            <span class="anime-badge">Trends</span>
            <span class="anime-badge">Genres</span>
            <span class="anime-badge">Studios</span>
            <span class="anime-badge">Predictions</span>
        </div>
        <hr style="border:1.2px solid #e75480; margin: 1.5em 0;">
        <ul style="font-size:1.1em; line-height:1.7;color:#a8325e;">
            <li><b>Genre Popularity Over Time</b>: See which genres are rising or fading.</li>
            <li><b>Seasonal Release Patterns</b>: Explore how seasons affect anime trends.</li>
            <li><b>Studio Specialization</b>: Discover the studios behind your favorite genres.</li>
            <li><b>Regional Preferences</b>: Find out what's trending worldwide.</li>
            <li><b>Success Prediction</b>: Try our ML model to predict anime hits!</li>
            <li><b>Genre Networks</b>: Visualize how genres blend in storytelling.</li>
        </ul>
        <hr style="border:1.2px solid #e75480; margin: 1.5em 0;">
        <div style="text-align:center; margin-top:1.5em;">
            <span style="font-size:1.12em; color:#a8325e;">
                Select a section from the left sidebar to begin your anime data journey!<br>
                <span style="font-size:2em;">👈</span>
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style='text-align:center; color:#fff; margin-top:2em; text-shadow:0 1px 10px #0007;'>
        <em>“To know sorrow is not terrifying. What is terrifying is to know you can't go back to happiness you could have.”</em><br>
        <span style="font-size:0.95em;">— Matsumoto Rangiku, <b>Bleach</b></span>
    </div>
    """,
    unsafe_allow_html=True
)
