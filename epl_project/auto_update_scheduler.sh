#!/bin/bash

# EPL 감독 정보 자동 업데이트 스케줄러
# 매일 오전 6시, 오후 6시 실행

cd "$(dirname "$0")"

echo "🚀 EPL 감독 정보 자동 업데이트 시작: $(date)"

# Python 스크립트 실행
python3 auto_update_managers.py

# Git 커밋 & 푸시
if [ -n "$(git status --porcelain epl_project/data/clubs_backup.json)" ]; then
    echo "📝 변경사항 감지, Git 커밋 중..."
    
    git add epl_project/data/clubs_backup.json
    git add epl_project/data/manager_update_log.json
    
    git commit -m "Auto: Update EPL managers info - $(date '+%Y-%m-%d %H:%M')"
    
    git push origin main
    
    echo "✅ Git 푸시 완료"
else
    echo "ℹ️ 변경사항 없음"
fi

echo "✅ 자동 업데이트 완료: $(date)"
