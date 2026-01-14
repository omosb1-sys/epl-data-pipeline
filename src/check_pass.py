import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/raw_data.csv')
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
print("--- counts ---")
print(df['pass_direction'].value_counts())
print("--- sample ---")
print(df[['team_name_ko', 'type_name', 'dx', 'pass_direction']].head(20))
