import pandas as pd
import os

def insert_missing(df_original):
    timestamps = pd.to_datetime(df_original.iloc[4:-1][df_original.columns[0]], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    data_df = pd.concat([timestamps, df_original.iloc[4:-1][df_original.columns[1]]], axis=1)
    
    #년,월,일
    year, month, day = timestamps.iloc[0].year, timestamps.iloc[0].month, timestamps.iloc[0].day
    
    all_hours = [pd.Timestamp(f"{year}-{month}-{day} {i:02d}:00:00") for i in range(24)]
    missing_hours = set(all_hours) - set(data_df[df_original.columns[0]].tolist()) # 누락확인
    
    for ts in missing_hours:
        data_df = data_df.append({df_original.columns[0]: ts, df_original.columns[1]: 0.0}, ignore_index=True)
    
    #정렬 후 이상한거 제거
    data_df = data_df.sort_values(by=df_original.columns[0], ignore_index=True)
    zero_hour_index = data_df.index[data_df[df_original.columns[0]] == f"{year}-{month}-{day} 00:00:00"].tolist()
    if zero_hour_index:
        data_df = data_df.loc[zero_hour_index[0]:]
    
    df_sorted = pd.concat([df_original.iloc[:3], data_df]).reset_index(drop=True)
    
    # 마지막에 혹시 모르니 행 추가
    df_sorted.loc[len(df_sorted)] = ["총합", None]
    
    return df_sorted

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
                df_filled = insert_missing(df)
                df_filled.to_excel(file_path, index=False)
