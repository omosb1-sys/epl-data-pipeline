## Plan Mode

- **Objective**: Fix the execution hang in `k_league_full_study_pipeline.py` and streamline the integration.
- **Problem**: The previous execution stalled because `plt.show()` waits for a GUI window to be closed, which isn't possible in this environment.
- **Changes**:
  1.  **Backend Config**: Set `matplotlib.use('Agg')` at the very beginning to force non-interactive mode (saving files only).
  2.  **Disable Blocking Calls**: Comment out or remove `plt.show()` and rely solely on `plt.savefig()`.
- **Validation**: Run the script again to generate the final report and prove the integration of AI modules works.

## Execution Steps
1.  Modify `src/k_league_full_study_pipeline.py` to use 'Agg' backend.
2.  Run the script and verify `reports/docs/study_insight_report.txt` is created with the new AI contents.
