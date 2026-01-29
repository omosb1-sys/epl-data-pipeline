"""
Fast Analytics Engine: Polars-based Zero-Copy Streaming
======================================================
Implements high-performance analysis without loading full datasets into RAM.
Tailored for 8GB RAM environments.

Author: Antigravity
"""
import polars as pl
import os

import polars.selectors as cs

class FastAnalyticsEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def scan_raw_data(self):
        """Returns a lazy frame for the raw Parquet data."""
        path = os.path.join(self.data_dir, "raw_data.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet file not found. Run convert_to_parquet.py first: {path}")
        return pl.scan_parquet(path)

    def get_summary_stats(self):
        """Calculates summary stats using streaming for zero RAM pressure."""
        lf = self.scan_raw_data()
        
        # Example: Efficient aggregation without loading full data
        stats = lf.select([
            pl.all().count().alias("total_rows"),
            cs.numeric().mean().name.suffix("_mean"),
            cs.numeric().max().name.suffix("_max")
        ]).collect(streaming=True)
        
        return stats

if __name__ == "__main__":
    DATA_DIR = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"
    engine = FastAnalyticsEngine(DATA_DIR)
    
    try:
        print("🔍 Scanning data via Polars Lazy API...")
        summary = engine.get_summary_stats()
        print("✅ Summary Statistics (Streaming-mode):")
        print(summary)
    except Exception as e:
        print(f"⚠️ Engine Error: {e}")
