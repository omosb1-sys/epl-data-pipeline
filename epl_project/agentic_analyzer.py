import json
import os
from slm_manager import SLMManager

class AgenticAnalyzer:
    """
    [DeepCode-Inspired Agentic Workflow]
    Implements a Plan -> Execute -> Verify loop for data analysis tasks.
    Optimized for 8GB RAM Mac using local SLM.
    """
    def __init__(self):
        self.slm = SLMManager()

    def run_workflow(self, task_description: str):
        """[Rule 23 & 27: Agentic & Native Workflow] Ground -> Plan -> Execute -> Observe -> Reflect -> Correct."""
        # 0. Native Data Grounding (Rule 27.1)
        # Scan task for data files and 'peek' inside to understand semantic context
        grounding_context = self._native_grounding_agent(task_description)
        print(f"📊 [Native Grounding] Data Context Extracted: {len(grounding_context)} bytes")

        # 1. Research & Impact Analysis
        research = self._research_agent(task_description, grounding_context)
        from mantic_search import mantic_search
        impact = mantic_search.search(task_description)
        
        # 2. Planning
        plan = self._planning_agent(task_description, research, grounding_context)
        
        # 3. Execution Loop (Max 3 attempts for Self-Healing)
        attempts = 0
        success = False
        execution_log = ""
        final_code = ""
        
        while attempts < 3 and not success:
            attempts += 1
            print(f"🚀 [Agentic Loop] Attempt {attempts}: Generating & Executing Code...")
            
            # 3.1 Coder Agent: Generate/Fix Code
            if attempts == 1:
                final_code = self._coder_agent(plan)
            else:
                final_code = self._healing_agent(final_code, execution_log)
            
            # 3.2 Executor: Run the code and Observe
            success, output, error = self._execute_and_observe(final_code)
            execution_log = f"STDOUT: {output}\nSTDERR: {error}"
            
            if success:
                print(f"✅ [Agentic Loop] Success on attempt {attempts}!")
                break
            else:
                print(f"⚠️ [Agentic Loop] Attempt {attempts} failed. Reflecting...")

        # 4. Final Review
        verified_result = self._review_agent(task_description, final_code, execution_log)
        
        # 5. [Rule 25: Autonomous Action] Connector Dispatch
        from connector_manager import connector_manager
        if success:
            action_results = connector_manager.autonomous_dispatch(verified_result)
        else:
            action_results = {"status": "Execution failed, no external action taken"}
        
        return {
            "success": success,
            "attempts": attempts,
            "code": final_code,
            "log": execution_log,
            "verification": verified_result,
            "actions": action_results
        }

    def _get_model_tier(self, complexity: str):
        """[Rule 25.2: Hybrid Intelligence] Routes task to the optimal model tier."""
        if complexity == "high":
            # [Pro Tier] For deep reasoning and cross-file changes
            return "Gemini-1.5-Pro (via Antigravity)"
        else:
            # [SLM Tier] For planning, routing, and drafting
            return self.slm.get_optimal_model()

    def _execute_and_observe(self, code: str):
        """[Action/Observation] Actually runs the code in a temp environment."""
        import subprocess
        import tempfile
        
        # Ensure we filter out Markdown if the LLM wrapped it in ```python
        clean_code = code.replace("```python", "").replace("```", "").strip()
        
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as tmp:
            tmp.write(clean_code)
            tmp_path = tmp.name
        
        try:
            result = subprocess.run(['python3', tmp_path], capture_output=True, text=True, timeout=30)
            os.unlink(tmp_path)
            if result.returncode == 0:
                return True, result.stdout, ""
            else:
                return False, result.stdout, result.stderr
        except Exception as e:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
            return False, "", str(e)

    def _healing_agent(self, failed_code: str, error_log: str):
        """[Reflection/Self-Correction] Fixes code based on error observation."""
        prompt = f"""
        [Self-Healing Mode]
        Failed Code:
        {failed_code}
        
        Error Log:
        {error_log}
        
        Goal: Analyze the error, reflect on why it happened, and output the CORRECTED Python code.
        Focus on: Library imports, file paths, and syntax.
        """
        return self.slm.query(prompt, system_prompt="You are an expert Debugging Agent.")

    def _native_grounding_agent(self, task: str):
        """[Rule 27.1: Zero-Gap Context] Automatically peeks at data files/tables."""
        # Simple extraction of likely file paths or table names from task
        import re
        files = re.findall(r'[\w\/\.]+\.(?:csv|xlsx|parquet|db|json)', task)
        
        context = ""
        for f in set(files):
            if os.path.exists(f):
                print(f"🔍 [Grounding] Peeking at {f}...")
                if f.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(f, nrows=5)
                    context += f"\nFile: {f}\nHeader: {list(df.columns)}\nSample:\n{df.to_string()}\n"
                elif f.endswith('.db'):
                    import duckdb
                    conn = duckdb.connect(f)
                    tables = conn.execute("SHOW TABLES").df()
                    context += f"\nDB: {f}\nTables: {list(tables['name'])}\n"
        return context

    def _research_agent(self, task: str, grounding: str):
        prompt = f"""
        [Senior Research Phase]
        Task: {task}
        Data Context: {grounding}
        Role: Senior Infrastructure Specialist
        Instruction: Research technical constraints for 8GB RAM Mac, library compatibilities, and data bottlenecks based on the actual data context provided.
        """
        return self.slm.query(prompt, system_prompt="You are a meticulous Research Agent.")

    def _planning_agent(self, task: str, research: str, grounding: str):
        prompt = f"""
        [Senior Planning Phase]
        Original Task: {task}
        Data Grounding: {grounding}
        Research Findings: {research}
        Role: Lead Solution Architect
        
        Instruction: 
        1. Propose 3 Distinct Options: Option A (MVP), Option B (Enterprise), Option C (Antigravity).
        2. Identify 5 Potential Failure Scenarios.
        3. Recommendation: Choose the best path based on the grounding data.
        """
        return self.slm.query(prompt, system_prompt="You are a Senior Planning Architect.")

    def _coder_agent(self, plan: str):
        prompt = f"""
        [Step 2: Coder Agent]
        Plan: {plan}
        Role: Expert Python Coder
        Instruction: Generate complete, executable Python code.
        """
        return self.slm.query(prompt, system_prompt="You are a Coder Agent.")

    def _review_agent(self, task: str, code: str, log: str):
        prompt = f"""
        [Step 3: Review & Debug Agent]
        Original Task: {task}
        Generated Code: {code}
        Execution Log: {log}
        Role: QA & Debug Expert
        Instruction: Verify the final outcome. Is it what the user wanted? Is it efficient?
        """
        return self.slm.query(prompt, system_prompt="You are a Review Agent.")

# Singleton Instance
agentic_analyzer = AgenticAnalyzer()

# Singleton Instance
agentic_analyzer = AgenticAnalyzer()
