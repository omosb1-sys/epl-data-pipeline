import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'AppleGothic'  # Support Korean on Mac
plt.rcParams['axes.unicode_minus'] = False

def visualize_pass_metrics(team_stats_path, player_stats_path):
    """
    Task 3: Create visualization charts for team and player pass metrics.
    """
    print("--- Task 3: Creating Visualizations ---")
    
    # 1. Load Metrics calculated in Task 2
    try:
        team_stats = pd.read_csv(team_stats_path)
        player_stats = pd.read_csv(player_stats_path)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return

    # 2. Create Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # --- Plot Team Metrics ---
    sns.barplot(
        x='pass_success_rate', 
        y='team_name_ko', 
        data=team_stats.sort_values('pass_success_rate', ascending=False),
        palette='viridis',
        ax=ax1
    )
    ax1.set_title('K-리그 팀별 패스 성공률 (%)', fontsize=16, fontweight='bold')
    ax1.set_xlim(80, 95)  # Zoom in to see differences
    ax1.set_xlabel('성공률 (%)')
    ax1.set_ylabel('팀명')

    # Add labels to bars
    for i, v in enumerate(team_stats.sort_values('pass_success_rate', ascending=False)['pass_success_rate']):
        ax1.text(v + 0.2, i, f"{v}%", color='black', va='center', fontweight='bold')

    # --- Plot Player Metrics (Top 10) ---
    top_10_players = player_stats.head(10).copy()
    top_10_players['player_label'] = top_10_players['player_name_ko'] + " (" + top_10_players['team_name_ko'] + ")"
    
    sns.barplot(
        x='pass_success_rate', 
        y='player_label', 
        data=top_10_players,
        palette='magma',
        ax=ax2
    )
    ax2.set_title('K-리그 선수별 패스 성공률 TOP 10 (100회 이상 시도)', fontsize=16, fontweight='bold')
    ax2.set_xlim(90, 100) # Zoom in
    ax2.set_xlabel('성공률 (%)')
    ax2.set_ylabel('선수명 (팀)')

    # Add labels to bars
    for i, v in enumerate(top_10_players['pass_success_rate']):
        ax2.text(v + 0.1, i, f"{v}%", color='black', va='center', fontweight='bold')

    plt.tight_layout()
    
    # 3. Save the result
    output_img = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/k_league_pass_analysis.png"
    plt.savefig(output_img, dpi=300)
    print(f"Visualization saved to: {output_img}")
    
    return output_img

if __name__ == "__main__":
    TEAM_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/team_pass_metrics.csv"
    PLAYER_PATH = "/Users/sebokoh/데이터분석연습/데이콘/data/processed/player_pass_metrics.csv" # Fixed path from task 2 bug
    
    # Task 2에서 저장된 경로를 확인 (이전 스텝에서 player_stats.to_csv 경로가 상위일 수 있어 보정)
    if not os.path.exists(PLAYER_PATH):
        PLAYER_PATH = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/data/processed/player_pass_metrics.csv"

    if os.path.exists(TEAM_PATH) and os.path.exists(PLAYER_PATH):
        visualize_pass_metrics(TEAM_PATH, PLAYER_PATH)
    else:
        print("Required CSV files from Task 2 not found.")
