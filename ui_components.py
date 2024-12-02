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

def display_match_prediction(analyzer):
    """Display match prediction UI"""
    st.subheader("🏆 Match Outcome Predictor")
    
    # Team selection columns
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", [
            "Liverpool", "Manchester City", "Arsenal", "Chelsea", 
            "Manchester United", "Tottenham"
        ], index=0)
    
    with col2:
        away_team = st.selectbox("Away Team", [
            "Chelsea", "Manchester United", "Arsenal", "Liverpool", 
            "Manchester City", "Tottenham"
        ], index=0)
    
    if st.button("Predict Match", key="predict_btn"):
        with st.spinner("Generating match prediction..."):
            prediction = analyzer.predict_match(home_team, away_team)
            
            # Probability Chart
            prob_data = {
                'Outcome': ['Home Win', 'Away Win', 'Draw'],
                'Probability': [
                    prediction['home_win_probability'],
                    prediction['away_win_probability'],
                    prediction['draw_probability']
                ]
            }
            
            # Plotly probability bar chart
            fig = px.bar(
                prob_data, 
                x='Outcome', 
                y='Probability', 
                title=f'{home_team} vs {away_team} Win Probabilities',
                labels={'Probability': 'Win Probability (%)'}
            )
            st.plotly_chart(fig)
            
            # Expected Goals
            st.markdown("### 📊 Expected Goals")
            goals_data = {
                'Team': [home_team, away_team],
                'Expected Goals': [
                    prediction['expected_goals']['home'], 
                    prediction['expected_goals']['away']
                ]
            }
            fig_goals = px.bar(
                goals_data, 
                x='Team', 
                y='Expected Goals', 
                title='Expected Goals Comparison'
            )
            st.plotly_chart(fig_goals)
            
            # AI Match Analysis
            st.markdown("### 🤖 AI Match Analysis")
            st.write(prediction['explanation'])
