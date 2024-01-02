import pandas as pd
import os
import glob

def process(file_path, base_folder_path):
    df = pd.read_excel(file_path, header=3)

    df['일시'] = pd.to_datetime(df.iloc[:, 0])

    for date, day_df in df.groupby(df['일시'].dt.date):
        year = date.year
        month = date.month
        day = date.day

        full_day = pd.date_range(start=f"{year}-{month}-{day}", end=f"{year}-{month}-{day} 23:00:00", freq='H')
        full_day_df = pd.DataFrame(full_day, columns=['date'])  # '일시'를 'date'로 변경

        merged_df = full_day_df.merge(day_df, left_on='date', right_on='일시', how='left').fillna(0)
        merged_df.drop('일시', axis=1, inplace=True)  # 원본 '일시' 열 삭제

        folder_path = os.path.join(base_folder_path, f"{year}_{month}")
        os.makedirs(folder_path, exist_ok=True)
        save_file_path = os.path.join(folder_path, f"{day}.xlsx")

        merged_df.rename(columns={df.columns[15]: 'SOL_RAD_LEVEL'}, inplace=True)
        merged_df.to_excel(save_file_path, index=False, columns=['date', 'SOL_RAD_LEVEL'])

directory = 'dataset/pre_school_report/sr_file'
paths = glob.glob(os.path.join(directory, '*.xlsx'))
for file_path in paths:
    process(file_path, directory)

