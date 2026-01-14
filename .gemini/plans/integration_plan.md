## Plan Mode

- **Objective**: Integrate `k_league_timesfm_forecast.py` and `polars_duckdb_synergy.py` into the main execution pipeline `k_league_full_study_pipeline.py`.
- **Changes**:
  1.  **Imports**: Add necessary imports (`sys` modification to ensure sibling imports work).
  2.  **Step 7 Insertion**: Add a new analysis step "AI Forecast & Advanced Wrangling" before the report generation.
  3.  **Logic Integration**: 
      - Use `BigDataEngine` to fetch aggressive team stats via Polars/DuckDB.
      - Use `KLeagueForecaster` to get goal predictions via TimesFM logic.
  4.  **Report Upgrade**: Include the new AI findings (Predicted Top Scorer, Aggressive Teams) in the final text report.
- **Questions/Assumptions**:
  - Assumption: The script is run from project root or `src/`. `sys.path.append` will handle import resolution.
  - Assumption: The user wants the console output to show the progress of these new steps.

## Execution Steps
1.  Modify `src/k_league_full_study_pipeline.py` to import the new classes.
2.  Insert the logic execution block before the final report writing.
3.  Update the report writing block to include new variables (`predicted_top_team`, `aggressive_teams_list`).
