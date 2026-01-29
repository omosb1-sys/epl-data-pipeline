
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set font for Korean support (Mac OS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class USHowitzerMarketAnalysis:
    """
    Class specifically designed to analyze the potential of Hanwha K9 export to the US
    following the cancellation of the ERCA (Extended Range Cannon Artillery) program.
    """
    
    def __init__(self):
        # Comparison Data: K9A2 vs M109A7 (Current) vs Potential Competitors
        self.competitor_specs = {
            'K9A2 Thunder': {
                'Range (km)': 40, # Base bleed, can be 50+ with RAP
                'Rate of Fire (rpm)': 9, # Burst
                'Automation Level': 9, # High (Auto-loader)
                'Cost ($M)': 4.5, # Est
                'Tech Maturity': 9, # Proven
                'Local Production': 0.7 # Subject to partnership
            },
            'M109A7 Paladin': {
                'Range (km)': 30,
                'Rate of Fire (rpm)': 4,
                'Automation Level': 5, # Manual/Semi
                'Cost ($M)': 14.0, # High per unit cost for upgrades
                'Tech Maturity': 10, # Fully fielded
                'Local Production': 1.0 # Native
            },
            'RCH 155 (Boxer)': {
                'Range (km)': 54, # V-LAP
                'Rate of Fire (rpm)': 9,
                'Automation Level': 8, # Remote
                'Cost ($M)': 12.0, # Est high
                'Tech Maturity': 7, # Newer
                'Local Production': 0.5 # Low U.S. footprint
            }
        }
    
    def plot_radar_comparison(self):
        """Generates a Radar Chart comparing key specifications."""
        print("Creating Radar Chart for Specification Comparison...")
        
        categories = ['Range (km)', 'Rate of Fire (rpm)', 'Automation Level', 'Tech Maturity']
        
        # Normalize data for radar chart (0-1 scale)
        df = pd.DataFrame(self.competitor_specs).T
        df_normalized = df[categories] / df[categories].max()
        
        # Data prep
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        colors = ['#FF6B6B', '#4D96FF', '#6BCB77'] # Colors for K9, Paladin, RCH
        
        for i, (name, data) in enumerate(df_normalized.iterrows()):
            values = data.values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=colors[i])
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, fontweight='bold')
        
        # Y-labels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=10)
        plt.ylim(0, 1.0)
        
        plt.title('SPH-M Competitor Analysis: K9A2 vs Competitors', size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        plt.savefig('k9_vs_competitors_radar.png', bbox_inches='tight', dpi=300)
        print("Saved: k9_vs_competitors_radar.png")
        plt.close()

    def run_monte_carlo_us_bid(self, n_simulations: int = 10000):
        """
        Simulates the US Army SPH-M selection score.
        Score = w1*Performance + w2*Cost + w3*LocalProduction + w4*TimeDelivery
        """
        print(f"Running {n_simulations} Monte Carlo simulations for US Bid...")
        
        # Weights (Hypothetical US Army Selection Criteria)
        # US prioritizes Domestic Production widely, but needs Capability fast (Time)
        weights = {
            'Performance': 0.35,
            'Cost Efficiency': 0.20,
            'Local Production': 0.30,
            'Time to Field': 0.15
        }
        
        results = []
        
        for _ in range(n_simulations):
            sim_data = {}
            
            # 1. K9A2 Simulation
            # Performance: High (Proven) +/- noise
            k9_perf = np.random.normal(90, 5) 
            # Cost: Lower than US native options -> High Score
            k9_cost = np.random.normal(85, 5) 
            # Local Prod: Biggest Risk. Range depends on 'Hanwha Defense USA' success
            # Scenario: 50% chance of great partnership, 50% chance of struggles
            if np.random.random() > 0.5:
                k9_local = np.random.normal(80, 10) # Good partnership
            else:
                k9_local = np.random.normal(50, 10) # Struggle
            # Time: Very fast to field
            k9_time = np.random.normal(95, 3) 
            
            sim_data['K9A2'] = (
                k9_perf * weights['Performance'] +
                k9_cost * weights['Cost Efficiency'] +
                k9_local * weights['Local Production'] +
                k9_time * weights['Time to Field']
            )

            # 2. M109A7 (Incumbent) Simulation
            m109_perf = np.random.normal(60, 5) # Lower range/rate of fire
            m109_cost = np.random.normal(50, 5) # Expensive
            m109_local = 100 # Perfect score
            m109_time = 100 # Already there
            
            sim_data['M109A7'] = (
                m109_perf * weights['Performance'] +
                m109_cost * weights['Cost Efficiency'] +
                m109_local * weights['Local Production'] +
                m109_time * weights['Time to Field']
            )
            
            # 3. RCH 155 (Tech Leader)
            rch_perf = np.random.normal(95, 5)
            rch_cost = np.random.normal(60, 5)
            rch_local = np.random.normal(40, 10) # Low localization initially
            rch_time = np.random.normal(60, 10) # Need new lines
            
            sim_data['RCH 155'] = (
                rch_perf * weights['Performance'] +
                rch_cost * weights['Cost Efficiency'] +
                rch_local * weights['Local Production'] +
                rch_time * weights['Time to Field']
            )
            
            results.append(sim_data)
            
        df_results = pd.DataFrame(results)
        
        # Calculate Win Probabilities
        wins = df_results.idxmax(axis=1).value_counts(normalize=True) * 100
        print("\n[Simulation Results] Win Probability:")
        print(wins)
        
        # Plot Distributions
        plt.figure(figsize=(12, 6))
        sns.kdeplot(df_results['K9A2'], fill=True, label='Hanwha K9A2', color='red')
        sns.kdeplot(df_results['M109A7'], fill=True, label='BAE M109A7', color='blue')
        sns.kdeplot(df_results['RCH 155'], fill=True, label='RCH 155', color='green')
        
        plt.title(f"US Army SPH-M Selection Score Distribution (N={n_simulations})\nWin Rate: K9 {wins.get('K9A2', 0):.1f}% vs M109 {wins.get('M109A7', 0):.1f}%")
        plt.xlabel('Total Evaluation Score (0-100)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('us_bid_simulation_kde.png', dpi=300)
        print("Saved: us_bid_simulation_kde.png")
        plt.close()
        
        return wins

if __name__ == "__main__":
    analyzer = USHowitzerMarketAnalysis()
    analyzer.plot_radar_comparison()
    analyzer.run_monte_carlo_us_bid()
