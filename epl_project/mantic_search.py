import subprocess
import json
import os

class ManticSearch:
    """
    [Mantic: Structural Code Search Engine]
    Uses mantic.sh to perform sub-500ms structural code search.
    Provides impact analysis for code changes.
    Optimized for Local-First AI Agent workflows (8GB RAM Mac).
    """
    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def search(self, query: str, path: str = None):
        """Performs a structural search using Mantic CLI."""
        cmd = ["mantic", "search", query]
        if path:
            cmd.extend(["--path", path])
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root_dir)
            if result.returncode == 0:
                return result.stdout
            return f"Error: {result.stderr}"
        except Exception as e:
            return f"Mantic Execution Error: {e}"

    def analyze_impact(self, file_path: str):
        """Analyzes the blast radius of a change in a specific file."""
        cmd = ["mantic", "impact", file_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root_dir)
            return result.stdout
        except Exception as e:
            return f"Impact Analysis Error: {e}"

# Singleton instance
mantic_search = ManticSearch()

if __name__ == "__main__":
    # Internal test
    print("🚀 Running Mantic Structural Search...")
    print(mantic_search.search("PluginManager"))
