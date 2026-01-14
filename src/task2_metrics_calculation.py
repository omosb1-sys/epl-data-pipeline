import pandas as pd
import os

def calculate_pass_metrics(df):
    """
    Task 2: Calculate pass success rate metrics for teams and top players.
    """
    print("--- Task 2: Calculating Pass Success Metrics ---")
    
    # Filter for only 'Pass' type actions
    pass_df = df[df['type_name'] == 'Pass'].copy()
    print(f"Total passes to analyze: {len(pass_df)}")

    # 1. Team Level Metrics
    team_metrics = pass_df.groupby('team_name_ko').agg(
        total_passes=('type_name', 'count'),
        successful_passes=('is_success', 'sum')
    )
    team_metrics['pass_success_rate'] = (team_metrics['successful_passes'] / team_metrics['total_passes'] * 100).round(2)
    team_metrics = team_metrics.sort_values(by='pass_success_rate', ascending=False)

    # 2. Player Level Metrics (Top 10 by total passes)
    player_metrics = pass_df.groupby(['player_name_ko', 'team_name_ko']).agg(
        total_passes=('type_name', 'count'),
        successful_passes=('is_success', 'sum')
    )
    player_metrics['pass_success_rate'] = (player_metrics['successful_passes'] / player_metrics['total_passes'] * 100).round(2)
    
    # Filter for players with at least 100 passes to ensure statistical significance
    top_players = player_metrics[player_metrics['total_passes'] >= 100].sort_values(by='pass_success_rate', ascending=False)

    print("\n[Top 5 Teams by Pass Success Rate]")
    print(team_metrics.head(5))

    print("\n[Top 5 Players by Pass Success Rate (min. 100 passes)]")
    print(top_players.head(5))

    return team_metrics, top_players

if __name__ == "__main__":
    # Import logic from Task 1
    from task1_data_loading import task1_preprocessing
    
    DATA_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/raw_data.csv"
    
    # Run Preprocessing first (Context preservation)
    df = task1_preprocessing(DATA_PATH)
    
    if df is not None:
        team_stats, player_stats = calculate_pass_metrics(df)
        
        # Save results for next task
        team_stats.to_csv("/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/team_pass_metrics.csv")
        player_stats.to_csv("/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/player_pass_metrics.csv")
        print("\nMetrics saved for Task 3 (Visualization).")
