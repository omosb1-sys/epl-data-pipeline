import json
import os
from collections import Counter
from datetime import datetime

class ContextGear:
    """
    [Personalized Context Gear]
    Learns from user interaction patterns to provide proactive and personalized analysis.
    """
    def __init__(self, memory_path="data/user_context_memory.json"):
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "episodes" not in data: data["episodes"] = []
                    return data
            except:
                return {"interactions": [], "preferences": {}, "episodes": []}
        return {"interactions": [], "preferences": {}, "episodes": []}

    def _save_memory(self):
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def record_interaction(self, query: str, matched_plugin: str = None):
        """
        [STITCH Protocol] Records interaction with contextual grounding.
        1. Event Type Tagging
        2. Key Entity Extraction
        3. Thematic Episode Segmenting
        """
        # 1. Event Type & Entity Extraction (Internal logic)
        event_type = "Analysis" if "분석" in query or "종합" in query else ("Prediction" if "예측" in query else "Inquiry")
        entities = []
        teams = ["토트넘", "맨유", "맨시티", "아스널", "리버풀", "첼시", "손흥민", "황희찬", "이강인"]
        for t in teams:
            if t in query: entities.append(t)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "plugin": matched_plugin,
            "event_type": event_type,
            "entities": entities
        }
        self.memory["interactions"].append(entry)
        
        # 2. Episode Segmentation (Thematic Scope)
        self._segment_episodes(entry)
        
        # Keep limits
        if len(self.memory["interactions"]) > 100:
            self.memory["interactions"] = self.memory["interactions"][-100:]
            
        self._update_preferences()
        self._save_memory()

    def _segment_episodes(self, last_entry):
        """
        [STITCH: Thematic Scope] Clusters interactions into episodes.
        If the topic changes significantly or time passes, start a new episode.
        """
        if not self.memory["episodes"]:
            self._start_new_episode(last_entry)
            return

        current_ep = self.memory["episodes"][-1]
        
        # Topic Match Check
        topic_match = any(e in current_ep["thematic_scope"] for e in last_entry["entities"])
        
        # Time Gap Check (30 mins)
        last_time = datetime.fromisoformat(current_ep["end_time"])
        now = datetime.fromisoformat(last_entry["timestamp"])
        time_gap = (now - last_time).total_seconds() > 1800

        if topic_match and not time_gap:
            # Continue episode
            current_ep["events"].append(last_entry)
            current_ep["end_time"] = last_entry["timestamp"]
            # Merge entities
            current_ep["thematic_scope"] = list(set(current_ep["thematic_scope"] + last_entry["entities"]))
        else:
            # Start new chapter
            self._start_new_episode(last_entry)

    def _start_new_episode(self, entry):
        new_ep = {
            "id": f"EP_{len(self.memory['episodes']) + 1}",
            "start_time": entry["timestamp"],
            "end_time": entry["timestamp"],
            "thematic_scope": entry["entities"] if entry["entities"] else ["General"],
            "events": [entry]
        }
        self.memory["episodes"].append(new_ep)
        # Keep only last 5 episodes
        if len(self.memory["episodes"]) > 5:
            self.memory["episodes"] = self.memory["episodes"][-5:]

    def _update_preferences(self):
        """[SOTA Update] 단순 키워드 매핑을 넘어 SLM을 이용한 톤/스타일 심층 분석"""
        from slm_manager import SLMManager
        slm = SLMManager()
        
        queries = [i["query"] for i in self.memory["interactions"][-10:]] # 최근 10개로 한정
        if not queries: return
        
        prompt = f"""
        Analyze the following user queries to determine:
        1. Tone & Manner: (e.g., Professional, Casual, Data-heavy, Concise)
        2. Preferred Metrics: (e.g., Goals, xG, Tactics, Transfers)
        3. Persona: What kind of analyst does this user prefer to talk to?
        
        Queries:
        {chr(10).join(queries)}
        
        Return the result in JSON format: {{"tone": "...", "metrics": ["..."], "persona": "..."}}
        """
        
        try:
            response = slm.query(prompt, system_prompt="You are a Stylistic & Behavioral Analyst.")
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                self.memory["preferences"]["persona_style"] = analysis
        except:
            self.memory["preferences"]["persona_style"] = {"tone": "Analytical", "metrics": ["Stats"], "persona": "Data Scientist"}

        # Legacy topic extraction (Top 3 overall)
        found_teams = []
        for i in self.memory["interactions"]:
            found_teams.extend(i.get("entities", []))
        self.memory["preferences"]["top_topics"] = [t for t, count in Counter(found_teams).most_common(3)]

    def get_personalized_prompt_prefix(self) -> str:
        """[STITCH] Generates grounded context string."""
        active_ep = self.memory["episodes"][-1] if self.memory["episodes"] else None
        prefs = self.memory.get("preferences", {})
        style = prefs.get("persona_style", {})
        
        context = []
        if active_ep:
            context.append(f"Active Episode: {active_ep['id']} (Topic: {', '.join(active_ep['thematic_scope'])})")
        
        if style:
            context.append(f"Style: {style.get('tone')} | Persona: {style.get('persona')}")
            
        if context:
            return f"[STITCH Context: {' | '.join(context)}]"
        return ""

    def get_routing_bias(self) -> dict:
        """Returns active episode topics for prioritized routing."""
        if not self.memory["episodes"]: return []
        return self.memory["episodes"][-1]["thematic_scope"]

# Singleton instance
context_gear = ContextGear()

# Singleton instance
context_gear = ContextGear()
