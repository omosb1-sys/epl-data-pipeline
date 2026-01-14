import pandas as pd
try:
    raw_data = pd.read_csv('data/raw/raw_data.csv', encoding='utf-8')
    match_info = pd.read_csv('data/raw/match_info.csv', encoding='utf-8')
    print("Raw Data Columns:", raw_data.columns.tolist())
    print("Match Info Columns:", match_info.columns.tolist())
    
    df = raw_data.merge(match_info, on='game_id', how='left')
    print("Merged DF Columns:", df.columns.tolist())
    
    game_team_stats = df.groupby(['game_id', 'team_name_ko']).size().reset_index(name='count')
    print("Initial Statistics Columns:", game_team_stats.columns.tolist())
    
    # Check if we can add home_team_id
    game_team_stats = game_team_stats.merge(
        match_info[['game_id', 'home_team_id', 'away_team_id']], 
        on='game_id', 
        how='left'
    )
    print("After Match Info Merge Columns:", game_team_stats.columns.tolist())
    
    # Check if we can add team_id
    team_id_map = df.groupby(['game_id', 'team_name_ko'])['team_id'].first().reset_index()
    print("Team ID Map Columns:", team_id_map.columns.tolist())
    
    game_team_stats = game_team_stats.merge(team_id_map, on=['game_id', 'team_name_ko'], how='left')
    print("Final Statistics Columns:", game_team_stats.columns.tolist())
    
except Exception as e:
    print(f"Error: {e}")
