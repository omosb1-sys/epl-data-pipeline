import os
import json
import requests
from slm_manager import SLMManager

class ConnectorManager:
    """
    [Rule 25: Connected Agent Architecture]
    Handles external service integrations (Slack, GitHub, Google Sheets) 
    inspired by Composio SDK and Claude Cowork.
    """
    def __init__(self):
        self.slm = SLMManager()
        self.connectors = {
            "slack": self._slack_publish,
            "github": self._github_commit,
            "reports": self._local_report_sync
        }

    def execute_action(self, service: str, payload: dict):
        """Executes an action on an external service."""
        if service in self.connectors:
            print(f"🔗 [CAA] Connecting to {service}...")
            return self.connectors[service](payload)
        return {"error": f"Service {service} not supported"}

    def _slack_publish(self, payload: dict):
        """Publishes a report to Slack via Webhook."""
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            return {"status": "Skipped", "reason": "SLACK_WEBHOOK_URL not set"}
        
        message = {
            "text": f"🚀 *Antigravity Analysis Report*\n{payload.get('message', '')}",
            "attachments": payload.get("attachments", [])
        }
        
        try:
            response = requests.post(webhook_url, json=message, timeout=10)
            return {"status": "Success", "code": response.status_code}
        except Exception as e:
            return {"status": "Error", "error": str(e)}

    def _github_commit(self, payload: dict):
        """
        Commits files to GitHub using MCP or local git.
        In this SDK context, we use a simplified git flow.
        """
        import subprocess
        try:
            subprocess.run(["git", "add", "."], capture_output=True)
            msg = payload.get("message", "Auto-commit by Antigravity Agent")
            subprocess.run(["git", "commit", "-m", msg], capture_output=True)
            # subprocess.run(["git", "push"], capture_output=True) # User might need to approve push
            return {"status": "Success", "action": "Local Commit Completed"}
        except Exception as e:
            return {"status": "Error", "error": str(e)}

    def _local_report_sync(self, payload: dict):
        """Syncs analytical results to a structured local directory."""
        sync_dir = "data/synced_reports"
        os.makedirs(sync_dir, exist_ok=True)
        filename = f"{payload.get('title', 'report')}.json"
        path = os.path.join(sync_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"status": "Success", "path": path}

    def autonomous_dispatch(self, analysis_result: str):
        """
        [Rule 25.3] Self-Routing Dispatcher.
        Decides if the result should be published externally.
        """
        prompt = f"""
        Result: {analysis_result[:500]}...
        Task: Should this result be published to 'slack', committed to 'github', or 'synced' locally?
        Return a JSON list of services: ["slack", "github"]
        """
        decision = self.slm.query(prompt, system_prompt="You are an Autonomous Action Dispatcher.")
        try:
            services = json.loads(decision)
            results = {}
            for s in services:
                results[s] = self.execute_action(s, {"message": analysis_result})
            return results
        except:
            return {"status": "No autonomous action taken"}

# Singleton Instance
connector_manager = ConnectorManager()
