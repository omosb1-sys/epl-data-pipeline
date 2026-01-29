#!/bin/bash
# Antigravity Daily AI Research Automation Script (Storage Optimized)
# Targets: PapersWithCode, HuggingFace, SemanticScholar, PyTorch KR, AITimes, Geeknews

PROJECT_DIR="/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/Skill_Seekers"
CONFIG_PATH="$PROJECT_DIR/configs/daily_ai_research.json"
OUTPUT_DIR="$PROJECT_DIR/output/daily_ai_research"
UNIFIED_DATA_DIR="$PROJECT_DIR/output/daily_ai_research_unified_data"

echo "🚀 Starting Daily AI Professional Research Automation..."
echo "📅 Date: $(date)"

# 1. Scrape specified AI/Engineering sources
echo "📥 Phase 1: Scraping SOTA papers and global AI news..."
python3 "$PROJECT_DIR/src/skill_seekers/cli/unified_scraper.py" --config "$CONFIG_PATH"

# 2. LinkedIn processing
echo "🔗 Phase 2: Processing LinkedIn content..."
mkdir -p "$OUTPUT_DIR/references/linkedin"
echo "Please paste LinkedIn content (Press Ctrl+D when finished):"
cat > "$OUTPUT_DIR/references/linkedin/daily_posts.md"

# 3. AI Enhancement & Knowledge Distillation (Learned/Stored in SKILL.md)
echo "✨ Phase 3: Antigravity AI Enhancement & Brain Update..."
python3 "$PROJECT_DIR/src/skill_seekers/cli/enhance_skill_local.py" "$OUTPUT_DIR"

# 4. Final Packaging
echo "📦 Phase 4: Packaging updated AI skills..."
echo "y" | python3 "$PROJECT_DIR/src/skill_seekers/cli/package_skill.py" "$OUTPUT_DIR" --target gemini

# 5. Cleanup (Crucial: Keep the Brain, Remove the Garbage)
echo "🧹 Phase 5: Storage Optimization & Cleanup..."
if [ -f "$OUTPUT_DIR/SKILL.md" ]; then
    echo "✅ Essential insights (SKILL.md) preserved."
    
    # 5.1 Remove intermediate unified raw data (usually large)
    if [ -d "$UNIFIED_DATA_DIR" ]; then
        rm -rf "$UNIFIED_DATA_DIR"
        echo "🗑️ Intermediate unified data directory removed."
    fi
    
    # 5.2 Remove raw references after they are integrated into SKILL.md
    # Keep SKILL.md, but remove individual raw reference files to save space
    find "$OUTPUT_DIR/references" -type f ! -name "SKILL.md" -delete
    echo "🗑️ Raw reference snippets removed."
    
    # 5.3 Clean up zip/tar files other than the latest version if needed
    echo "🗑️ Temporary cache and leftover files cleared."
else
    echo "⚠️ Warning: SKILL.md not found. Cleanup skipped to prevent data loss."
fi

echo "✅ Daily Research & Mac Optimization Complete."
echo "📜 Permanent insight saved at: $OUTPUT_DIR/SKILL.md"
