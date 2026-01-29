import os
import importlib.util
import streamlit as st
import json
from slm_manager import SLMManager
from context_gear import context_gear
from audit_logger import audit_logger
import time

class PluginManager:
    """
    Antigravity Plugin Manager: Dynamically loads analysis modules from the 'plugins/' directory.
    """
    def __init__(self, plugin_dir="plugins"):
        self.plugin_dir = plugin_dir
        self.plugins = {}
        self.slm = SLMManager()
        self.gear = context_gear
        self.load_plugins()

    def load_plugins(self):
        """Scans the plugin directory and imports available plugins."""
        if not os.path.exists(self.plugin_dir):
            os.makedirs(self.plugin_dir)
            return

        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                plugin_name = filename[:-3]
                file_path = os.path.join(self.plugin_dir, filename)
                
                spec = importlib.util.spec_from_file_location(plugin_name, file_path)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    # Check for required interface
                    if hasattr(module, "get_metadata"):
                        metadata = module.get_metadata()
                        self.plugins[plugin_name] = {
                            "module": module,
                            "metadata": metadata
                        }
                except Exception as e:
                    print(f"Error loading plugin {plugin_name}: {e}")

    def get_plugin_names(self):
        """Returns a list of display names for active plugins."""
        return [p["metadata"]["display_name"] for p in self.plugins.values()]

    def get_plugin_by_display_name(self, display_name):
        """Retrieves plugin module info by its display name."""
        for p in self.plugins.values():
            if p["metadata"]["display_name"] == display_name:
                return p
        return None

    def render_plugin_ui(self, display_name, **kwargs):
        """Renders the Streamlit UI of the selected plugin."""
        plugin = self.get_plugin_by_display_name(display_name)
        if plugin and hasattr(plugin["module"], "render_ui"):
            plugin["module"].render_ui(**kwargs)
        else:
            st.warning(f"Plugin '{display_name}' has no UI implementation.")

    def get_plugin_intelligence(self, display_name, **kwargs):
        """Fetches structured JSON data from a plugin for AI agents."""
        plugin = self.get_plugin_by_display_name(display_name)
        if plugin and hasattr(plugin["module"], "get_intelligence"):
            return plugin["module"].get_intelligence(**kwargs)
        return {"error": "Plugin not found or intelligence not supported"}

    def route_request(self, query: str):
        """
        [Amazon Orchestration] Routes the user query to the most relevant plugin.
        Uses metadata descriptions and names for matching.
        """
        query = query.lower()
        scores = []
        for name, p in self.plugins.items():
            metadata = p["metadata"]
            score = 0
            # weight name, display_name and description
            targets = [
                metadata.get("name", ""),
                metadata.get("display_name", ""),
                metadata.get("description", "")
            ]
            for target in targets:
                if any(word in target.lower() for word in query.split()):
                    score += 1
            if score > 0:
                scores.append((score, metadata["display_name"]))
        
        if scores:
            # Sort by score descending
            scores.sort(key=lambda x: x[0], reverse=True)
            return scores[0][1] # Return the best display_name
        return None

    def semantic_route_request(self, query: str) -> str:
        """
        [Advanced Orchestration] Uses local SLM (Phi-3.5/Qwen) to route based on intent.
        Fallbacks to keyword search if SLM is unavailable.
        """
        plugin_list = [p["metadata"]["display_name"] for p in self.plugins.values()]
        plugin_desc = [f"- {p['metadata']['display_name']}: {p['metadata'].get('description', '')}" for p in self.plugins.values()]
        
        prompt = f"""
        User Query: "{query}"
        User Context: {self.gear.get_personalized_prompt_prefix()}
        
        Available Analysis Agents:
        {chr(10).join(plugin_desc)}
        
        Task: Pick the MOST RELEVANT agent for this query. Return ONLY the display name of the agent.
        If none are highly relevant, return "None".
        """
        
        system_prompt = "You are an expert Multi-Agent Router for EPL analysis."
        
        try:
            # Try semantic first
            response = self.slm.query(prompt, system_prompt=system_prompt, temperature=0.1)
            for name in plugin_list:
                if name.lower() in response.lower():
                    return name
        except:
            pass
            
        # Fallback to keyword
        return self.route_request(query)

    def get_chained_intelligence(self, query: str, **kwargs) -> str:
        """
        [Advanced RAG: Sumanth Strategy]
        Uses Multi-Query Translation and Hybrid Routing before synthesis.
        """
        from query_translator import query_translator, hybrid_router
        
        # 1. Query Translation (Multi-Query)
        translated_queries = query_translator.generate_multi_queries(query)
        st.caption(f"🔄 **Query Translation (RAG-Fusion)**: {len(translated_queries)}개 변형 생성됨")
        
        # 2. Dynamic Routing (Structured vs Unstructured)
        data_source = hybrid_router.route_query_type(query)
        if data_source == "SQL_DB":
            st.info("📊 **데이터 소스**: 정형 통계 데이터베이스(DuckDB) 우선 검색")
        else:
            st.info("📎 **데이터 소스**: 비정형 전술 컨텍스트 및 뉴스 검색")

            # [OpenAI Observability] Measure execution cost
            start_time = time.time()
            intel = self.get_plugin_intelligence(metadata["display_name"], **kwargs)
            duration = time.time() - start_time
            
            audit_logger.log_execution(metadata["display_name"], duration, query)
            
            if "error" not in intel:
                results[metadata["display_name"]] = intel
        
        # Synthesis using SLM
        context = json.dumps(results, ensure_ascii=False)
        user_context_str = self.gear.get_personalized_prompt_prefix()
        style = self.gear.memory.get("preferences", {}).get("persona_style", {})
        
        prompt = f"""
        Query: {query}
        User History Context: {user_context_str}
        Context from various agents: {context}
        
        Synthesize a comprehensive answer based on the intelligence from different agents.
        Adopt the user's preferred Tone: {style.get('tone', 'Analytical')} and Persona: {style.get('persona', 'Expert Analyst')}.
        Focus on metrics they prefer: {', '.join(style.get('metrics', ['General']))}.
        """
        
        system_msg = f"You are a {style.get('persona', 'Master Analyst')} synthesizing agent inputs in a {style.get('tone', 'professional')} manner."
        raw_synthesis = self.slm.query(prompt, system_prompt=system_msg)
        
        # [PCL-Reasoner] Neuro-Symbolic Verification
        from neuro_symbolic_verifier import ns_verifier
        verification = ns_verifier.verify_prediction(
            raw_synthesis, 
            probability=0.7, # Default confidence for synthesis
            context={"is_home": True, "injured_count": 2} # Dynamic context can be passed here
        )
        
        if verification["status"] == "Unstable":
            raw_synthesis += f"\n\n--- \n⚠️ **[시스템 논리 검토]** {chr(10).join(verification['warnings'])}\n"
            raw_synthesis += f"*(PCL-Reasoner 기반 논리 무결성 점수: {verification['logic_score']})*"
        
        # [PCL: Distillation] 고품질 추론 데이터 증류 및 보관 (Unsloth Fine-tuning용)
        from distillation_engine import distillation_engine
        distillation_engine.save_verified_trace(query, raw_synthesis, verification)
            
        return raw_synthesis

# Singleton Instance
@st.cache_resource
def get_plugin_manager():
    return PluginManager()
