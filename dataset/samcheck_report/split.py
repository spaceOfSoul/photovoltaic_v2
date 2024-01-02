import pandas as pd
import os

# def split_and_save_by_day(df, base_folder_path):
#     for date, day_df in df.groupby(df['date'].str[:10]):
#         year, month, day = date.split('-')
#         folder_path = os.path.join(base_folder_path, f"{year}_{month}")
#         os.makedirs(folder_path, exist_ok=True)
#         file_path = os.path.join(folder_path, f"{day}.csv")
#         day_df.to_csv(file_path, index=False)

# # CSV 파일 로드
# file_path = 'dataset/samcheck_report/pv_f16.csv'
# data = pd.read_csv(file_path)

# # 데이터 분할 및 저장
# split_and_save_by_day(data,'dataset/samcheck_report')

def process(file_path, base_folder_path):
    # 데이터 불러오기
    df = pd.read_csv(file_path)

    # 'date' 열을 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'])

    # 날짜별로 그룹화
    for date, day_df in df.groupby(df['date'].dt.date):
        year = date.year
        month = date.month
        day = date.day

        # 시간별데이터생성
        full_day = pd.date_range(start=f"{year}-{month}-{day}", end=f"{year}-{month}-{day} 23:00:00", freq='H')
        full_day_df = pd.DataFrame(full_day, columns=['date'])

        # 제로패딩
        merged_df = full_day_df.merge(day_df, on='date', how='left').fillna(0)

        folder_path = os.path.join(base_folder_path, f"{year}_{month}")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{day}.xlsx")

        merged_df.to_excel(file_path, index=False, columns=['date', 'SOL_RAD_LEVEL'])

file_path = 'dataset/samcheck_report/pv_f16.csv'
save_path = 'dataset/samcheck_report'

process(file_path, save_path)
