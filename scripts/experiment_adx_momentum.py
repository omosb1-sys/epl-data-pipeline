
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Settings & Data Loaidng ---
FILE_PATH = 'data/processed/team_match_results.csv'
LOOKBACK = 5  # K-League Patch Size (Usually 5 games is a good trend indicator)

# Set Korean Font for Mac (AppleGothic)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(path):
    if not os.path.exists(path):
        # Fallback to absolute path if relative fails
        path = os.path.join(os.getcwd(), path)
        
    df = pd.read_csv(path)
    # Ensure sorted by game_id to simulate time
    df = df.sort_values(by='game_id')
    return df

# --- 2. ADX Calculation Function (The Core Logic) ---
def calculate_adx(df, lookback=14):
    """
    Calculates ADX, +DI, -DI based on Synthetic High, Low, Close.
    Expected columns in df: 'high', 'low', 'close'
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    # Logic: If UpMove > DownMove and UpMove > 0, then +DM = UpMove, else 0
    # Note: Stock market logic is slightly different, let's stick to standard Wilder's ADX
    # Standard:
    # UpMove = High[i] - High[i-1]
    # DownMove = Low[i-1] - Low[i] (Note the order!)
    
    # Let's use the vectorized approach from the article but verify direction
    # The article used:
    # plus_dm = high.diff()
    # minus_dm = low.diff() -> This is wrong for standard ADX. 
    # Standard ADX DownMove is (Previous Low - Current Low).
    # If using stock library: +DM = max(H-H_prev, 0), -DM = max(L_prev - L, 0)
    
    # Let's implement rigorous Wilder's DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # Initialize DM series
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    # +DM conditions
    plus_mask = (up_move > down_move) & (up_move > 0)
    plus_dm[plus_mask] = up_move[plus_mask]
    
    # -DM conditions
    minus_mask = (down_move > up_move) & (down_move > 0)
    minus_dm[minus_mask] = down_move[minus_mask]
    
    # True Range (TR)
    # TR = Max(|H-L|, |H-Cp|, |L-Cp|)
    tr1 =  high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothing (Wilder's Smoothing is often alpha=1/lookback, or similar)
    # The article used rolling mean for ATR, but Wilder uses smoothing.
    # Let's use Simple Moving Average (rolling mean) as per article for simplicity in demo
    atr = tr.rolling(lookback).mean()
    
    # Smoothed DM
    plus_dm_smooth = plus_dm.rolling(lookback).mean()
    minus_dm_smooth = minus_dm.rolling(lookback).mean()
    
    # DI Calculation
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX (Smoothed DX)
    adx = dx.rolling(lookback).mean()
    
    return plus_di, minus_di, adx

# --- 3. Main Execution & Visualization ---
def run_experiment():
    print("⚽️ Loading K-League Data...")
    df = load_data(FILE_PATH)
    
    # Target Teams to Analyze (Top 3 Contenders 2024 + 1 Struggling Team)
    target_teams = ['Ulsan HD FC', 'Gangwon FC', 'Jeonbuk Hyundai Motors'] 
    
    plt.figure(figsize=(15, 10))
    
    for i, team_name in enumerate(target_teams):
        print(f"Analyzing {team_name}...")
        team_df = df[df['team'] == team_name].copy()
        
        # --- SYNTHETIC OHLC CREATION ---
        # Close: Cumulative Points (The 'Price' of the team)
        team_df['close'] = team_df['points'].cumsum()
        
        # High: Close + Goals For (Upside Potential displayed in match)
        # Low: Close - Goals Against (Downside Risk displayed in match)
        # Using a multiplier to exaggerate volatility for better ADX visualization if needed
        # But let's stick to 1.0 first.
        team_df['high'] = team_df['close'] + team_df['goals_for']
        team_df['low'] = team_df['close'] - team_df['goals_against']
        
        # Calculate ADX
        plus_di, minus_di, adx = calculate_adx(team_df, lookback=LOOKBACK)
        
        team_df['+DI'] = plus_di
        team_df['-DI'] = minus_di
        team_df['ADX'] = adx
        
        # Visualization
        # Plot 1: Price (Cumulative Points)
        ax1 = plt.subplot(len(target_teams), 2, i*2 + 1)
        ax1.plot(team_df.index, team_df['close'], label='Cumulative Points (Price)', color='black')
        ax1.fill_between(team_df.index, team_df['low'], team_df['high'], alpha=0.2, color='gray', label='Goal Volatility (H-L)')
        ax1.set_title(f"{team_name} - Performance Trend")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ADX & DI
        ax2 = plt.subplot(len(target_teams), 2, i*2 + 2)
        ax2.plot(team_df.index, team_df['ADX'], label='ADX (Strength)', color='purple', linewidth=2)
        ax2.plot(team_df.index, team_df['+DI'], label='+DI (Bull/Atk)', color='red',  linestyle='--', alpha=0.7)
        ax2.plot(team_df.index, team_df['-DI'], label='-DI (Bear/Def)', color='blue', linestyle='--', alpha=0.7)
        
        # Threshold Line
        ax2.axhline(25, color='gray', linestyle=':', label='Strong Trend Filter (25)')
        ax2.set_title(f"Momentum Strength (ADX)")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # --- Insight Extraction (Last Match) ---
        last = team_df.iloc[-1]
        print(f"   [Last {LOOKBACK} Games] ADX: {last['ADX']:.2f}")
        if last['ADX'] > 20:
             trend = "📈 Trending" if last['+DI'] > last['-DI'] else "📉 Crashing"
             print(f"   Status: {trend} (Strong Direction)")
        else:
             print(f"   Status: 🦀 Ranging (No clear direction, volatile)")
        print("-" * 30)

    # Save to disk
    output_path = 'reports/adx_experiment_result.jpg'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
