import streamlit as st 
import plotly.express as px 
import plotly.graph_objs as go 

def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.title("⚽ Premier League Analytics")
    
    # Teams list for dropdowns
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City", 
        "Manchester United", "Tottenham", "Newcastle", "Brighton",
        "Aston Villa", "Wolves", "Crystal Palace", "Fulham",
        "Brentford", "Nottingham Forest", "Everton", "Bournemouth",
        "Luton", "Burnley", "Sheffield United"
    ]
    
    # Sidebar selection
    selected_feature = st.sidebar.radio(
        "Choose Analysis Type",
        ["Match Prediction", "Team Form", "Head to Head", "Natural Language Query"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Additional info or resources
    st.sidebar.info(
        "🔍 Get deep insights into Premier League teams and matches!\n"
        "Use the features to predict match outcomes, analyze team performance, "
        "and get comprehensive football analytics."
    )
    
    return selected_feature