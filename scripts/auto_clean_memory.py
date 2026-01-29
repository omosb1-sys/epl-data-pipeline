
import os
import time
import shutil
from datetime import datetime, timedelta

MEMORY_DIR = ".agent/memory"
TTL_DAYS = 7

def clean_memory():
    """
    Deletes files in .agent/memory that are older than TTL_DAYS (7 days).
    Enforces the 'Privacy-First Ephemeral Memory Protocol'.
    """
    # 1. Ensure memory dir exists
    if not os.path.exists(MEMORY_DIR):
        print(f"✅ Memory directory {MEMORY_DIR} does not exist. Nothing to clean.")
        return

    now = time.time()
    cutoff_time = now - (TTL_DAYS * 86400) # 7 days in seconds
    deleted_count = 0
    
    print(f"🔒 [Privacy Protocol] Scanning for memories older than {TTL_DAYS} days...")

    for root, dirs, files in os.walk(MEMORY_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                # Check file modification time
                file_mtime = os.path.getmtime(file_path)
                
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
                    file_date = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d')
                    print(f"   🗑️ Shredded: {filename} (Created: {file_date})")
            except Exception as e:
                print(f"   ⚠️ Error deleting {filename}: {e}")

    if deleted_count > 0:
        print(f"✨ Cleanup Complete. Implemented 'Right to be Forgotten' for {deleted_count} items.")
    else:
        print("✨ Memory is fresh. No expired data found.")

if __name__ == "__main__":
    clean_memory()
