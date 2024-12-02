import numpy as np 
import pandas as pd 
import psycopg2 
import torch 
from typing import Dict, List 
import ollama 
from sentence_transformer import SenetenceTransformers
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

            # Create embeddings for matches
            for _, match in matches_df.iterrows():
                match_text = self._convert_match_to_text(match)
                embedding = self.embedding_model.encode([match_text])[0]
                
                self.index.add(np.array([embedding]).astype('float32'))
                self.stored_matches.append({
                    'text': match_text,
                    'match_data': match.to_dict()
                })
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