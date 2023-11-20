from utility import list_up_weather
import numpy as np
import pandas as pd
import csv
import os

import torch
from torch.utils.data import Dataset

class Intensity(Dataset):
    def __init__(self, csv_file, start_date, end_date):
        self.data = pd.read_csv(csv_file)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data[(self.data['date'] >= pd.to_datetime(start_date)) & 
                              (self.data['date'] <= pd.to_datetime(end_date))]
        self.data = self.data.set_index('date').resample('H').sum()
        self.grouped_data = self.data.groupby(self.data.index.date)
    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        # 각 날짜별 데이터 추출
        date = list(self.grouped_data.groups)[idx]
        daily_data = self.grouped_data.get_group(date)
        
        # zero padding
        if len(daily_data) < 24:
            padded_data = daily_data.reindex(pd.date_range(start=date, periods=24, freq='H'), fill_value=0)
            daily_values = padded_data['SOL_RAD_LEVEL'].values
        else:
            daily_values = daily_data['SOL_RAD_LEVEL'].values
        return torch.tensor(daily_values, dtype=torch.float32)

# for test
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    start_date = '2022-01-01'
    end_date = '2022-08-31'
    
    dataset = Intensity('samcheok/pv_f16.csv', start_date, end_date)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, data in enumerate(data_loader):
        print(data.shape)
        if i == 89:
            plt.plot(data.numpy()[0])
            plt.title("Insolation Data Visualization")
            plt.xlabel("Time")
            plt.ylabel("Insolation")
            plt.show()
    
    print(i)