# 💡 Knowledge & Findings

## Technical Stack Discovery
- **Streamlit Cloud Limitations**: The 'Viewer Badge' is injected by the host wrapper and cannot be reliably hidden via internal CSS without risking sidebar navigation breakage.
- **Dynamic CSS**: Targeting `[data-testid="stSidebarNav"]` is safer than global `header button` to avoid UI side effects.

## Data Insights
- **TOON Optimization**: Successfully implemented `pytoony` to reduce token footprint by ~15%, enabling deeper context for AI analysis.
- **EPL Scraping**: News sources from Google News and reliable insiders (Romano/Ornstein) are successfully categorized via keyword mapping.
- **Model Explainability**: SHAP values successfully visualize which team features (Possession, Shots on Target) drive the AI's win probability.
