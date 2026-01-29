import json
import os

class NeuroSymbolicVerifier:
    """
    [PCL-Reasoner Inspired] Neuro-Symbolic Verification Engine.
    Combines LLM Probabilistic Reasoning (Neuro) with Hard Football Logic (Symbolic).
    """
    def __init__(self):
        # Symbolic Rules (The "Fixed Textbook" from PCL paper)
        self.rules = {
            "key_injury": -0.15,
            "home_advantage": 0.10,
            "away_fatigue": -0.05,
            "manager_new": -0.10,
            "derby_match": 0.05
        }

    def verify_prediction(self, raw_analysis: str, probability: float, context: dict) -> dict:
        """
        Grades the prediction based on symbolic logic.
        Returns a verification score and a logic correction.
        """
        logic_score = 1.0
        warnings = []
        
        # 1. Injury Logic Check
        injured_count = context.get("injured_count", 0)
        if injured_count > 3 and probability > 0.8:
            logic_score -= 0.2
            warnings.append("⚠️ [Logic Collision] High win probability despite critical injuries.")
            
        # 2. Home/Away Consistency
        is_home = context.get("is_home", True)
        if not is_home and probability > 0.85:
            logic_score -= 0.1
            warnings.append("🚨 [Rarity Alert] Exceptionally high probability for an Away team.")
            
        # 3. Reasoning Trace Quality (PCL-style CoT check)
        keywords = ["부상", "전술", "흐름", "xG", "데이터"]
        found_keywords = sum(1 for k in keywords if k in raw_analysis)
        if found_keywords < 2:
            logic_score -= 0.3
            warnings.append("📉 [Low CoT] Reasoning path lacks analytical depth (Symbolic penalty).")

        # 4. [Rule 22.1] Inference Efficiency Check
        # Penalize if reasoning is too verbose/redundant for the complexity of the query
        token_count = len(raw_analysis.split())
        if token_count > 500 and logic_score > 0.9:
            logic_score -= 0.05
            warnings.append("⚡ [IaaS Warning] High redundancy detected. Consider distilling rationale for cost efficiency.")

        status = "Verified" if logic_score >= 0.7 else "Unstable"
        
        return {
            "status": status,
            "logic_score": round(logic_score, 2),
            "warnings": warnings,
            "final_prob_adj": round(probability * logic_score, 2)
        }

    def select_best_path(self, reasoning_paths: list) -> str:
        """
        [PCL: Grading] Selects the most logically consistent reasoning path among multiple options.
        """
        scores = []
        for path in reasoning_paths:
            # Score based on density of data points and causal indicators (because, since, so)
            causal_words = ["때문에", "따라서", "결과적으로", "이유는"]
            score = sum(1 for w in causal_words if w in path)
            scores.append((score, path))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

# Singleton instance
ns_verifier = NeuroSymbolicVerifier()
