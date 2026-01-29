import argparse
import json
import os
import sys

# Add current directory to path to find plugin_manager
sys.path.append(os.getcwd())

from plugin_manager import PluginManager

def main():
    parser = argparse.ArgumentParser(description="Antigravity Agent Intelligence API")
    parser.add_argument("--team", required=True, help="Target team name")
    parser.add_argument("--plugin", required=True, help="Plugin name to query (e.g., tactics_plugin)")
    parser.add_argument("--format", default="json", choices=["json", "text"], help="Output format")

    args = parser.parse_args()

    # Initialize PluginManager (UI context not required here)
    pm = PluginManager()
    
    # In a real scenario, we'd load some data first. 
    # For this CLI tool, we'll simulate a minimal clubs_data if needed.
    dummy_clubs_data = [{"team_name": args.team, "manager_name": "AI Orchestrator", "power_index": 75}]

    # Fetch Intelligence
    display_name = ""
    # Find display name mapping for convenience
    for p_name, p_info in pm.plugins.items():
        if p_name == args.plugin:
            display_name = p_info["metadata"]["display_name"]
            break
    
    if not display_name:
        print(json.dumps({"error": f"Plugin '{args.plugin}' not found."}))
        return

    intel = pm.get_plugin_intelligence(display_name, selected_team=args.team, clubs_data=dummy_clubs_data)

    if args.format == "json":
        print(json.dumps(intel, indent=2, ensure_ascii=False))
    else:
        print(f"--- Intelligence Report for {args.team} ---")
        for k, v in intel.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
