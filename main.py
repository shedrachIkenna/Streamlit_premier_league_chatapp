import streamlit as st 
import os
from dotenv import load_dotenv
import pandas as pd 
import plotly.express as px
import plotly.graph_objs as go
import os
from dotenv import load_dotenv


from rag_analyzer import PremierLeagueRAGAnalyzer
from ui_components import (
    display_match_prediction, 
    display_team_form, 
    display_head_to_head,
    sidebar_navigation
)

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "football_stats_db")
}

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Premier League Insights",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize the RAG Analyzer
    analyzer = PremierLeagueRAGAnalyzer(DB_PARAMS)

    # Main application title
    st.markdown('<h1 class="title">🏆 Premier League Insights</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    selected_feature = sidebar_navigation()

    # Main content area
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    if selected_feature == "Match Prediction":
        display_match_prediction(analyzer)
    elif selected_feature == "Team Form":
        display_team_form(analyzer)
    elif selected_feature == "Head to Head":
        display_head_to_head(analyzer)
    elif selected_feature == "Natural Language Query":
        display_natural_language_query(analyzer)

    st.markdown('</div>', unsafe_allow_html=True)

def display_natural_language_query(analyzer):
    st.subheader("🤔 Natural Language Query")
    
    query = st.text_area(
        "Ask any question about Premier League teams, matches, or performance", 
        height=150
    )
    
    if st.button("Get Insights", key="query_btn"):
        if query:
            with st.spinner("Analyzing your query..."):
                response = analyzer.process_query(query)
                
                if 'response' in response:
                    st.markdown("### 🔍 Analysis Result")
                    st.write(response['response'])
                    
                    if 'relevant_matches' in response:
                        st.markdown("### 📊 Relevant Matches")
                        for match in response['relevant_matches']:
                            st.markdown(f"- {match}")
                else:
                    st.error("Sorry, could not process the query.")
        else:
            st.warning("Please enter a query!")

if __name__ == "__main__":
    main()

