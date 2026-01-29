import json
import os
from slm_manager import SLMManager

class SQLOptimizer:
    """
    [Rule 24: Advanced Text-to-SQL Protocol]
    Optimizes SQL generation using:
    1. Schema Filtering
    2. Business Glossary Grounding
    3. Multi-stage Reasoning (Thinking Step)
    4. Few-shot Best Practices
    """
    def __init__(self):
        self.slm = SLMManager()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.glossary_path = os.path.join(self.base_dir, "data/epl_business_glossary.json")
        self.examples_path = os.path.join(self.base_dir, "data/epl_sql_best_practices.sql")
        
        # Load constraints
        self.glossary = self._load_json(self.glossary_path)
        with open(self.examples_path, 'r') as f:
            self.examples = f.read()

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    def generate_optimized_sql(self, user_query: str):
        """Generates SQL using the 4-step Advanced Protocol."""
        
        # Step 1: Schema Context Selection (Only relevant tables)
        schema_context = """
        Table: fixtures (fixture_id, date, home_team, away_team, status)
        Table: live_stats (fixture_id, timestamp, home_possession, away_possession, home_shots, away_shots, home_goals, away_goals)
        Table: odds (fixture_id, timestamp, bookmaker, home_win_odds, draw_odds, away_win_odds)
        Table: predictions (fixture_id, timestamp, home_win_prob, draw_prob, away_win_prob, value_bet_side, value_bet_edge)
        """

        # Step 2: Multi-stage Thinking Prompt
        prompt = f"""
        [Advanced Text-to-SQL Protocol]
        User Query: "{user_query}"
        
        S1: Schema Info:
        {schema_context}
        
        S2: Business Glossary:
        {json.dumps(self.glossary, ensure_ascii=False)}
        
        S3: Best Practices (Few-shot):
        {self.examples}
        
        Task: 
        1. List the tables needed.
        2. Define JOIN conditions.
        3. Write the optimized DuckDB SQL.
        
        Format the output clearly with 'THOUGHT' and 'SQL' sections.
        """
        
        print("🧠 [SQL Thinking] Generating optimized query path...")
        return self.slm.query(prompt, system_prompt="You are a Lead SQL Architect specializing in DuckDB.")

if __name__ == "__main__":
    optimizer = SQLOptimizer()
    result = optimizer.generate_optimized_sql("최근 가치가 가장 높은 베팅(Value Bet) 3개를 알려줘")
    print(result)
