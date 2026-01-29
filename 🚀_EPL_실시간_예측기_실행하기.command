#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate || echo "No venv found, trying global python"
streamlit run epl_project/epl_betting_dashboard.py
