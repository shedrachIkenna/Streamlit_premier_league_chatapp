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
        home_team = st.selectbox("Home Team", ['Liverpool', 'Everton', 'WestHamUnited',
       'CrystalPalace', 'BrightonandHoveAlbion', 'ManchesterCity',
       'Bournemouth', 'Southampton', 'TottenhamHotspur',
       'AstonVilla', 'Arsenal', 'NewcastleUnited',
       'LeicesterCity', 'Chelsea', 'WolverhamptonWanderers',
       'ManchesterUnited', 'Fulham',
       'Brentford', 'NottinghamForest', 'IpswichTown'], index=0)
    
    with col2:
        away_team = st.selectbox("Away Team", ['Liverpool', 'Everton', 'WestHamUnited',
       'CrystalPalace', 'BrightonandHoveAlbion', 'ManchesterCity',
       'Bournemouth', 'Southampton', 'TottenhamHotspur',
       'AstonVilla', 'Arsenal', 'NewcastleUnited',
       'LeicesterCity', 'Chelsea', 'WolverhamptonWanderers',
       'ManchesterUnited', 'Fulham',
       'Brentford', 'NottinghamForest', 'IpswichTown'], index=0)
    
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

def display_team_form(analyzer):
    """Display team form analysis"""
    st.subheader("📈 Team Performance Analysis")
    
    team = st.selectbox("Select Team", [
        "Liverpool", "Manchester City", "Arsenal", "Chelsea", 
        "Manchester United", "Tottenham"
    ])
    
    if st.button("Analyze Team", key="team_form_btn"):
        with st.spinner("Analyzing team performance..."):
            team_form = analyzer.get_team_form(team)
            
            # Recent Results Visualization
            st.markdown(f"### 🏅 {team} Recent Form")
            results_map = {'W': 'green', 'D': 'gray', 'L': 'red'}
            result_colors = [results_map.get(r, 'gray') for r in team_form['recent_results']]
            
            fig = go.Figure(data=[go.Bar(
                x=[f'Match {i+1}' for i in range(len(team_form['recent_results']))],
                y=[1]*len(team_form['recent_results']),
                marker_color=result_colors
            )])
            fig.update_layout(
                title='Recent Match Results (Green: Win, Gray: Draw, Red: Loss)',
                xaxis_title='Matches',
                yaxis_title='Result'
            )
            st.plotly_chart(fig)
            
            # Performance Metrics
            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Goals Scored (Avg)", f"{team_form['goals_scored_avg']:.2f}")
            
            with col2:
                st.metric("Goals Conceded (Avg)", f"{team_form['goals_conceded_avg']:.2f}")
            
            with col3:
                st.metric("Expected Goals (Avg)", f"{team_form['xg_avg']:.2f}")

def display_head_to_head(analyzer):
    """Display head-to-head comparison"""
    st.subheader("🤝 Head-to-Head Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("First Team", [
            "Liverpool", "Manchester City", "Arsenal", "Chelsea", 
            "Manchester United", "Tottenham"
        ], index=0)
    
    with col2:
        team2 = st.selectbox("Second Team", [
            "Chelsea", "Manchester United", "Arsenal", "Liverpool", 
            "Manchester City", "Tottenham"
        ], index=1)
    
    if st.button("Compare Teams", key="h2h_btn"):
        with st.spinner("Comparing team performances..."):
            h2h_data = analyzer.get_head_to_head(team1, team2)
            
            # Wins Comparison
            st.markdown("### 🏆 Head-to-Head Wins")
            wins_data = {
                'Outcome': [f'{team1} Wins', f'{team2} Wins', 'Draws'],
                'Count': [
                    h2h_data['summary'][f'{team1}_wins'], 
                    h2h_data['summary'][f'{team2}_wins'], 
                    h2h_data['summary']['draws']
                ]
            }
            
            fig = px.pie(
                wins_data, 
                values='Count', 
                names='Outcome', 
                title=f'Head-to-Head Matches: {team1} vs {team2}'
            )
            st.plotly_chart(fig)
            
            # Recent Matches
            st.markdown("### 📅 Recent Matches")
            for match in h2h_data['matches']:
                st.markdown(
                    f"**{match['date']}**: {match['home_team']} {match['score']} {match['away_team']}"
                )