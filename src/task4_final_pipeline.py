import os
import datetime
from task1_data_loading import task1_preprocessing
from task2_metrics_calculation import calculate_pass_metrics
from task3_visualization import visualize_pass_metrics

def run_full_pipeline():
    print(f"==================================================")
    print(f"🚀 K-League Data Analysis Pipeline Started")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==================================================\n")

    # 1. Constants
    DATA_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/raw/raw_data.csv"
    TEAM_STATS_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/team_pass_metrics.csv"
    PLAYER_STATS_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/player_pass_metrics.csv"
    
    # 2. Execute Steps
    try:
        # Step 1: Preprocessing
        print("[Step 1/3] Preprocessing raw data...")
        df = task1_preprocessing(DATA_PATH)
        if df is None: raise Exception("Task 1 failed")

        # Step 2: Calculation
        print("\n[Step 2/3] Calculating metrics...")
        team_stats, player_stats = calculate_pass_metrics(df)
        team_stats.to_csv(TEAM_STATS_PATH)
        player_stats.to_csv(PLAYER_STATS_PATH)

        # Step 3: Visualization
        print("\n[Step 3/3] Generating visual charts...")
        chart_path = visualize_pass_metrics(TEAM_STATS_PATH, PLAYER_STATS_PATH)

        # Final Report Summary
        print(f"\n==================================================")
        print(f"✅ Analysis Pipeline Completed Successfully!")
        print(f"--------------------------------------------------")
        print(f"Result Chart: {chart_path}")
        print(f"Team Stats: {TEAM_STATS_PATH}")
        print(f"Player Stats: {PLAYER_STATS_PATH}")
        print(f"==================================================")

        # Generate a simple text report
        report_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/reports/docs/final_analysis_report.txt"
        with open(report_path, "w") as f:
            f.write("K-League Pass Analysis Report\n")
            f.write(f"Generated at: {datetime.datetime.now()}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Top Team: {team_stats.index[0]} ({team_stats['pass_success_rate'].iloc[0]}%)\n")
            f.write(f"Top Player: {player_stats.index[0][0]} / {player_stats.index[0][1]} ({player_stats['pass_success_rate'].iloc[0]}%)\n")
            f.write(f"\nVisual analysis saved at: {chart_path}\n")
        
        print(f"Final summary report saved to: {report_path}")

    except Exception as e:
        print(f"\n❌ Pipeline failed during execution: {e}")

if __name__ == "__main__":
    run_full_pipeline()
