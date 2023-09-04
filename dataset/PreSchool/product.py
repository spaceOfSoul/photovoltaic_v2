import os
import pandas as pd

# c9 : 70.84kw
# 어린이집 : 25.35
VOLUME_RATE = 70.84/25.35

def adjust_volume(df_original):
    df_original.iloc[4:-1, 1] *= VOLUME_RATE
    return df_original

for month in range(1, 13):
    dir_name = f"2022_{month}"
    dir_path = os.path.join('', dir_name)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        files = os.listdir(dir_path)
        
        for file in files:
            file_path = os.path.join(dir_path, file)
            print(file_path)
            if file.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                adjusted = adjust_volume(df)
                adjusted.to_excel(file_path, index=False)