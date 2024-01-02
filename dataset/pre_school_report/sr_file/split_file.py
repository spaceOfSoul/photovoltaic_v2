import pandas as pd
import os
import glob

def find_header_row(file_path):
    # 파일을 10개의 행을 read.
    for i in range(10):
        df = pd.read_excel(file_path, header=i)
        if not any(df.columns.str.contains('Unnamed')):
            return i
    return 0  # 뭐 없으면 첫 번째

def save_excel_by_day_fixed(df, base_folder_path):
    date_time_col = df.columns[df.first_valid_index()]

    df_filtered = df[df[date_time_col].str.match(r'\d{4}-\d{2}-\d{2}', na=False)]

    for date, day_df in df_filtered.groupby(df_filtered[date_time_col].str[:10]):
        year, month, day = date.split('-')

        folder_path = os.path.join(base_folder_path, f"{year}_{month}")
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"{day}.xlsx")
        day_df.to_excel(file_path, index=False)

def process_all_excel_files(directory):
    excel_files = glob.glob(os.path.join(directory, '*.xlsx'))

    for file in excel_files:
        header_row = find_header_row(file)
        df = pd.read_excel(file, header=header_row)
        save_excel_by_day_fixed(df, directory)

directory = 'dataset/pre_school_report/sr_file'
process_all_excel_files(directory)
