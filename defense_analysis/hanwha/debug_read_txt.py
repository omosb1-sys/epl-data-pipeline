import os

file_path = '/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/defense_analysis/hanwha/raw_data_paste.txt'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f.read())
else:
    print("File not found.")
