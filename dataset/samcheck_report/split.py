import pandas as pd
import os

def split_and_save_by_day(df, base_folder_path):
    for date, day_df in df.groupby(df['date'].str[:10]):
        year, month, day = date.split('-')
        folder_path = os.path.join(base_folder_path, f"{year}_{month}")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{day}.csv")
        day_df.to_csv(file_path, index=False)

# CSV 파일 로드
file_path = 'dataset/samcheck_report/pv_f16.csv'
data = pd.read_csv(file_path)

# 데이터 분할 및 저장
split_and_save_by_day(data,'dataset/samcheck_report')
