import pandas as pd
import numpy as np
import os

def task1_preprocessing(file_path):
    print(f"--- Task 1: Loading and Preprocessing Data from {os.path.basename(file_path)} ---")
    
    # 1. Load Data
    try:
        # csv loading
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # 2. Check for Missing Values
    print("\n[Missing Values Before Processing]")
    print(df.isnull().sum())

    # 3. Handling Missing Values
    
    # - player_id: If missing, we can't attribute the action to a specific player. 
    #   For this analysis, we might drop them if they are few, or fill with a placeholder.
    df['player_id'] = df['player_id'].fillna(-1).astype(int)
    
    # - result_name: Often 'Pass' has 'Successful' or 'Unsuccessful'. 
    #   If it's NaN (like in Pass Received), we fill with 'Neutral' or 'Unknown'
    df['result_name'] = df['result_name'].fillna('Informational')
    
    # - team_name_ko / player_name_ko: Fill missing names with 'Unknown'
    df['team_name_ko'] = df['team_name_ko'].fillna('Unknown Team')
    df['player_name_ko'] = df['player_name_ko'].fillna('Unknown Player')
    
    # - position_name / main_position: Fill with 'Unknown'
    df['position_name'] = df['position_name'].fillna('Unknown')
    df['main_position'] = df['main_position'].fillna('Unknown')

    # 4. Data Type Conversion
    # Ensure coordinates are numeric
    coord_cols = ['start_x', 'start_y', 'end_x', 'end_y']
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("\n[Missing Values After Processing]")
    print(df.isnull().sum())

    # 5. Feature Engineering (Optional for Task 1, but good for "Reinforcement")
    # Let's create a 'is_success' boolean for easy analysis later
    df['is_success'] = df['result_name'] == 'Successful'

    print("\n--- Task 1 Completed ---")
    return df

if __name__ == "__main__":
    DATA_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/raw_data.csv"
    if os.path.exists(DATA_PATH):
        # We work on a sample for speed if needed, but let's try the full data 
        # as it's only 90MB (pandas should handle this fine in memory)
        processed_df = task1_preprocessing(DATA_PATH)
        
        # Save a sample of cleaned data to verify
        if processed_df is not None:
            output_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/cleaned_k_league_data.csv"
            processed_df.head(1000).to_csv(output_path, index=False)
            print(f"\nSample of cleaned data saved to: {output_path}")
            
            # Update spec_demo.md status
            print("\nUpdating spec_demo.md...")
    else:
        print(f"Data file not found at {DATA_PATH}")
