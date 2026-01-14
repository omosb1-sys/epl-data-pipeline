import pandas as pd
import numpy as np

# Load
raw_data = pd.read_csv('data/raw/raw_data.csv', encoding='utf-8')
match_info = pd.read_csv('data/raw/match_info.csv', encoding='utf-8')
df = raw_data.merge(match_info, on='game_id', how='left')

# 2.2
def classify_pass_direction(dx, dy, type_name):
    if type_name not in ['Pass', 'Pass_Freekick', 'Pass_Corner', 'Cross', 'Throw-In']:
        return 'Not Applicable'
    if dx > 5:
        return '전진 패스'
    elif dx < -5:
        return '후방 패스'
    else:
        return '횡패스'

df['pass_direction'] = df.apply(
    lambda x: classify_pass_direction(x['dx'], x['dy'], x['type_name']), axis=1
)

# [2.7] part
pass_stats_adv = df.groupby('team_name_ko').apply(
    lambda x: pd.Series({
        'total_passes': len(x[x['type_name'].isin(['Pass', 'Pass_Freekick', 'Pass_Corner', 'Cross'])]),
        'forward_passes': len(x[x['pass_direction'] == '전진 패스'])
    })
)
print(pass_stats_adv.head())
