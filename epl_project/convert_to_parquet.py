"""
Parquet Migration Script: CSV to Zero-Copy Parquet
==================================================
Converts heavy CSV files to Parquet for 10x faster loading 
and 1/5 memory footprint on Intel Mac 8GB.

Author: Antigravity
"""
import polars as pl
import os
import time

def convert_csv_to_parquet(csv_path: str):
    """Converts a CSV file to Parquet using Polars."""
    parquet_path = csv_path.replace('.csv', '.parquet')
    
    print(f"🚀 Converting {os.path.basename(csv_path)} to Parquet...")
    start_time = time.time()
    
    try:
        # Use scan_csv for memory efficiency (Lazy API)
        df = pl.scan_csv(csv_path)
        df.sink_parquet(parquet_path, compression='zstd')
        
        duration = time.time() - start_time
        old_size = os.path.getsize(csv_path) / (1024 * 1024)
        new_size = os.path.getsize(parquet_path) / (1024 * 1024)
        
        print(f"✅ Success! {os.path.basename(parquet_path)} created.")
        print(f"📊 Stats: {old_size:.1f}MB -> {new_size:.1f}MB (Reduction: {(1 - new_size/old_size)*100:.1f}%)")
        print(f"⏱️ Duration: {duration:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error converting {csv_path}: {e}")

if __name__ == "__main__":
    DATA_DIR = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data"
    files_to_convert = [
        os.path.join(DATA_DIR, "raw_data.csv"),
        os.path.join(DATA_DIR, "match_info.csv")
    ]
    
    for file_path in files_to_convert:
        if os.path.exists(file_path):
            convert_csv_to_parquet(file_path)
        else:
            print(f"⚠️ Skip: {file_path} not found.")
