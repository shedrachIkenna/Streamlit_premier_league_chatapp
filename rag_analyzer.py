import numpy as np 
import pandas as pd 
import psycopg2 
import torch 
from typing import Dict, List 
import ollama 
from sentence_transformers import SentenceTransformer
import faiss 
import sqlalchemy
from sqlalchemy import create_engine
import warnings


# Suppress pandas SQLAlchemy warnings
warnings.filterwarnings('ignore', category=UserWarning)

class PremierLeagueRAGAnalyzer:
    def __init__(self, db_params: Dict[str, str]):
        # Database connection parameters 
        self.db_params = db_params 

        # Embedding model for semantic search 
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Ollama llama3 configuration 
        self.llm_model = "llama3"

        # Vector store for semantic search 
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.stored_matches = []

        # Initialize vector store with recent matches 
        self._initialize_vector_store()


    def _standardize_team_name(self, team_name: str) -> str:
        """
        Standardize team names using the provided mapping
        
        Args:
            team_name (str): Original team name from the database
        
        Returns:
            str: Standardized team name
        """
        # Create the mapping dictionary
        map_values = {
            'Leicester City': 'LeicesterCity',
            'Crystal Palace': 'CrystalPalace',
            'Norwich City': 'NorwichCity',
            'Aston Villa': 'AstonVilla',
            'Brighton': 'BrightonandHoveAlbion',
            'Manchester United': 'ManchesterUnited',
            'Leeds United': 'LeedsUnited',
            'Manchester City': 'ManchesterCity',
            'Newcastle': 'NewcastleUnited',
            'Tottenham': 'TottenhamHotspur',
            'Westham': 'WestHamUnited', 
            'Wolves': 'WolverhamptonWanderers',
            'Luton Town': 'LutonTown',
            'Nottingham Forest': 'NottinghamForest',
            "Sheffield United": 'SheffieldUnited',
            "Leeds United": "LeedsUnited",
            "Norwich City": "NorwichCity",
            "West Brom": "WestBromwichAlbion",
            "Ipswich": "IpswichTown",
        }
        
        return map_values.get(team_name, team_name)


    def _get_connection(self):
        """Create and return database connection""" 
        return psycopg2.connect(**self.db_params)

    def _initialize_vector_store(self):
        """Load recent matches and create embedding for semantic search""" 
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT *
                    FROM "PremierLeague"
                    WHERE date >= NOW() - INTERVAL '6 months'
                    ORDER BY date DESC
                """
                matches_df = pd.read_sql_query(query, conn)
                
                # Standardize team and opponent names
                matches_df['team'] = matches_df['team'].apply(self._standardize_team_name)
                matches_df['opponent'] = matches_df['opponent'].apply(self._standardize_team_name)

            # Rest of the method remains the same
        except Exception as e:
            print(f"Error initializing vector store: {e}")

    def _convert_match_to_text(self, match: pd.Series) -> str:

        """Convert match data to searchable text"""
        return f"""
        Match between {match['team']} and {match['opponent']} on {match['date']}
        Venue: {match['venue']}
        Score: {match['team']} {match['gf']}-{match['ga']} {match['opponent']}
        Team Performance:
        - Shots: {match['sh']} (On Target: {match['sot']})
        - Expected Goals: {match['xg']} vs {match['xga']}
        Result: {match['result']}
        """
    def _retrieve_relevant_matches(self, query: str, k: int = 3) -> List[Dict]:
        """Semantic search for relevant matches"""
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        return [self.stored_matches[idx] for idx in indices[0] if idx != -1]

    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using Ollama Llama 3"""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {
                        'role': 'system', 
                        'content': 'You are a helpful Premier League football analyst.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 500
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "Sorry, I couldn't generate a response at the moment."

    def get_team_form(self, team: str, last_n_matches: int = 5) -> Dict:
        """Calculate team's recent form and statistics"""
        team = self._standardize_team_name(team)
        query = """
            SELECT date, team, opponent, venue, gf, ga, sh, sot, xg, xga, result
            FROM "PremierLeague"
            WHERE team = %s
            ORDER BY date DESC
            LIMIT %s
        """
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[team, last_n_matches])

            # Standardize team and opponent names
            df['team'] = df['team'].apply(self._standardize_team_name)
            df['opponent'] = df['opponent'].apply(self._standardize_team_name)
            
        if df.empty:
            return {"error": "No data found for this team"}
            
        # Calculate form metrics
        form_metrics = {
            "recent_results": df['result'].tolist(),
            "goals_scored_avg": df['gf'].mean(),
            "goals_conceded_avg": df['ga'].mean(),
            "shots_avg": df['sh'].mean(),
            "shots_on_target_avg": df['sot'].mean(),
            "xg_avg": df['xg'].mean(),
            "xga_avg": df['xga'].mean(),
            "last_matches": df[['date', 'opponent', 'result', 'gf', 'ga']].to_dict('records')
        }
        
        return form_metrics

    def get_head_to_head(self, team1: str, team2: str, last_n_matches: int = 5) -> Dict:
        """Get head-to-head statistics between two teams"""
        query = """
            SELECT *
            FROM "PremierLeague"
            WHERE (team = %s AND opponent = %s) OR (team = %s AND opponent = %s)
            ORDER BY date DESC
            LIMIT %s
        """
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[team1, team2, team2, team1, last_n_matches])
            df['team'] = df['team'].apply(self._standardize_team_name)
            df['opponent'] = df['opponent'].apply(self._standardize_team_name)
            
        if df.empty:
            return {"error": "No head-to-head data found"}
            
        h2h_stats = {
            "matches": [],
            "summary": {
                f"{team1}_wins": 0,
                f"{team2}_wins": 0,
                "draws": 0
            }
        }
        
        for _, match in df.iterrows():
            match_info = {
                "date": match['date'].strftime('%Y-%m-%d'),
                "home_team": match['team'] if match['venue'] == 'Home' else match['opponent'],
                "away_team": match['opponent'] if match['venue'] == 'Home' else match['team'],
                "score": f"{match['gf']}-{match['ga']}" if match['venue'] == 'Home' else f"{match['ga']}-{match['gf']}"
            }
            h2h_stats["matches"].append(match_info)
            
            # Update win/loss/draw counts
            if match['result'] == 'W':
                if match['team'] == team1:
                    h2h_stats["summary"][f"{team1}_wins"] += 1
                else:
                    h2h_stats["summary"][f"{team2}_wins"] += 1
            elif match['result'] == 'L':
                if match['team'] == team1:
                    h2h_stats["summary"][f"{team2}_wins"] += 1
                else:
                    h2h_stats["summary"][f"{team1}_wins"] += 1
            else:
                h2h_stats["summary"]["draws"] += 1
                
        return h2h_stats

    def predict_match(self, home_team: str, away_team: str) -> Dict:
        """Predict match outcome based on recent form and historical data"""
        # Get recent form for both teams
        home_form = self.get_team_form(home_team, last_n_matches=10)
        away_form = self.get_team_form(away_team, last_n_matches=10)
        h2h = self.get_head_to_head(home_team, away_team, last_n_matches=5)

        print(f"This is the home_form: {home_form}")
        print(f"This is the away_team form: {away_form}")
        
        if "error" in home_form or "error" in away_form:
            return {"error": "Insufficient data for prediction"}
        
        # Calculate basic probability based on recent form
        home_xg_strength = home_form["xg_avg"]
        home_xga_strength = home_form["xga_avg"]
        away_xg_strength = away_form["xg_avg"]
        away_xga_strength = away_form["xga_avg"]
        
        # Adjust for home advantage
        home_advantage = 1.1  # 10% boost for home team
        
        # Calculate win probabilities
        home_win_prob = (home_xg_strength * home_advantage / away_xga_strength) / 2
        away_win_prob = (away_xg_strength / (home_xga_strength * home_advantage)) / 2
        draw_prob = 1 - (home_win_prob + away_win_prob)
        
        # Normalize probabilities
        total = home_win_prob + away_win_prob + draw_prob
        home_win_prob /= total
        away_win_prob /= total
        draw_prob /= total
        
        prediction = {
            "home_win_probability": round(home_win_prob * 100, 1),
            "away_win_probability": round(away_win_prob * 100, 1),
            "draw_probability": round(draw_prob * 100, 1),
            "expected_goals": {
                "home": round(home_xg_strength, 2),
                "away": round(away_xg_strength, 2)
            },
            "recent_form": {
                "home": home_form["recent_results"],
                "away": away_form["recent_results"]
            },
            "head_to_head": h2h
        }
        
        # Add natural language explanation using LLM
        explanation_prompt = f"""
        Provide a detailed betting analysis for the match between {home_team} and {away_team}.
        
        Match Prediction Details:
        - {home_team} Win Probability: {prediction['home_win_probability']}%
        - {away_team} Win Probability: {prediction['away_win_probability']}%
        - Draw Probability: {prediction['draw_probability']}%
        
        {home_team} Recent Form: {prediction['recent_form']['home']}
        {away_team} Recent Form: {prediction['recent_form']['away']}
        
        Provide insights into the match prediction, highlighting key factors
        that influenced the probabilities and potential betting strategies.
        """
        
        prediction['explanation'] = self._generate_llm_response(explanation_prompt)
        
        return prediction

    def process_query(self, query: str) -> Dict:
        """Process natural language queries about Premier League teams"""
        query = query.lower()
        
        # Retrieve semantically relevant matches
        relevant_matches = self._retrieve_relevant_matches(query)
        
        # Extract team names
        teams = self._extract_teams_from_query(query)
        
        # Prepare additional context for LLM
        context = "\n".join([match['text'] for match in relevant_matches])
        
        # Standard query processing
        if "prediction" in query or "odds" in query or "chances" in query:
            if len(teams) == 2:
                return self.predict_match(teams[0], teams[1])
            return {"error": "Please specify two teams for prediction"}
            
        if "form" in query or "performance" in query:
            if len(teams) == 1:
                return self.get_team_form(teams[0])
            return {"error": "Please specify one team for form analysis"}
            
        if "head to head" in query or "versus" in query or "vs" in query:
            if len(teams) == 2:
                return self.get_head_to_head(teams[0], teams[1])
            return {"error": "Please specify two teams for head-to-head comparison"}
        
        # Use LLM to generate a response for more complex queries
        llm_prompt = f"""
        You are a Premier League football analyst. 
        Analyze the following query using the context of recent match data.

        Query: {query}

        Relevant Match Context:
        {context}

        Provide a detailed, insightful analysis based on the query and the match context.
        """
        
        # Generate LLM response for complex queries
        llm_response = self._generate_llm_response(llm_prompt)
        
        return {
            "response": llm_response,
            "relevant_matches": [match['text'] for match in relevant_matches]
        }

    def _extract_teams_from_query(self, query: str) -> List[str]:
        """
        Advanced team name extraction with multiple matching strategies
        
        Supports:
        - Case-insensitive matching
        - Partial matches
        - Common variations and abbreviations
        """
        # Comprehensive team mapping with variations
        team_mappings = {
            'Arsenal': ['Arsenal', 'Ars'],
            'Chelsea': ['Chelsea', 'Chels'],
            'Liverpool': ['Liverpool', 'Liv'],
            'Manchester City': ['Manchester City', 'Man City', 'City'],
            'Manchester United': ['Manchester United', 'Man Utd', 'United'],
            'Tottenham': ['Tottenham', 'Spurs', 'Tottenham Hotspur', 'TottenhamHotspur'],
            'Newcastle Utd': ['Newcastle', 'Newcastle United', 'Newcastle Utd'],
            'Brighton': ['Brighton', 'Brighton and Hove Albion'],
            'Aston Villa': ['Aston Villa', 'Villa'],
            'Wolves': ['Wolves', 'Wolverhampton', 'Wolverhampton Wanderers'],
            'Crystal Palace': ['Crystal Palace', 'Palace'],
            'Fulham': ['Fulham'],
            'Brentford': ['Brentford'],
            'Nottingham Forest': ['Nottingham Forest', 'Forest'],
            'Everton': ['Everton'],
            'Bournemouth': ['Bournemouth', 'AFC Bournemouth'],
            'Luton Town': ['Luton', 'Luton Town'],
            'Burnley': ['Burnley'],
            'Sheffield Utd': ['Sheffield United', 'Sheffield Utd']
        }
        
        # Normalize query
        normalized_query = query.lower()
        
        matched_teams = []
        
        for canonical_name, variations in team_mappings.items():
            # Check for complete or partial matches
            if any(variation.lower() in normalized_query for variation in variations):
                matched_teams.append(canonical_name)
        
        return matched_teams