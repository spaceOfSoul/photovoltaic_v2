import numpy as np
import pandas as pd
import csv
import os

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
from datetime import datetime, timedelta

from utility import insolation_aprox

class WPD(Dataset):
    def __init__(self,aws_list,asos_list,energy_list, isolation_list, start2end,region_ID,input_dim=8,datapath="../dataset/",kernel_range=(6,20)):
        self.aws_list = aws_list  # all files for weather info
        self.asos_list = asos_list
        self.elist = energy_list  # all files for power gener.
        self.isolation_list = isolation_list # 
        
        if 'samcheck' in self.isolation_list[0]:
            self.isol_data = pd.read_csv(isolation_list[0])
            self.isol_data['date'] = pd.to_datetime(self.isol_data['date'])
            self.isol_data = self.isol_data[(self.isol_data['date'] >= pd.to_datetime(start2end[0])) & 
                                  (self.isol_data['date'] <= pd.to_datetime(start2end[1]))]
            self.isol_data = self.isol_data.set_index('date').resample('H').sum()
            self.grouped_data = self.isol_data.groupby(self.isol_data.index.date)
        
        self.rID = region_ID
        self.input_dim = input_dim
        self.kernel_range = kernel_range

        self.tags1 = ["지점","지점명","일시","기온(°C)","1분 강수량(mm)","풍향(deg)","풍속(m/s)","현지기압(hPa)","해면기압(hPa)","습도(%)",]
        self.tags2 = ["지점","지점명","일시","기온(°C)","누적강수량(mm)","풍향(deg)","풍속(m/s)","현지기압(hPa)","해면기압(hPa)","습도(%)",]

        if not os.path.isdir(datapath):
            os.makedirs(datapath)
            
        ## 연간 모든 데이터에 대해서 일단 한번 로드
        ## 평균, 표준편차를 구하기 위함
        all_data = []
        for aws_path, asos_path in zip(self.aws_list, self.asos_list):
            aws_csv = pd.read_csv(aws_path, encoding="CP949")[self.tags1]
            asos_csv = pd.read_csv(asos_path, encoding="CP949")[self.tags2]
            asos_csv.iloc[:, 4] = asos_csv.iloc[:, 4].diff()
            asos_csv = asos_csv.rename(columns={self.tags2[4]: self.tags1[4]})
            csv = pd.concat([aws_csv, asos_csv])
            csv = csv.drop(["지점명"], axis=1)\
            
            groups = csv.groupby(csv.columns[0])
            weather_by_region = {}
            for i in groups:
                if weather_by_region.get(i[0]) is not None:
                    weather_by_region[i[0]].append(list(i))
                else:
                    weather_by_region[i[0]] = list(i)

            # Choose region & Time alignment
            rid = self.rID
            region_data = weather_by_region[rid]
            region_data = region_data[1].values
            weather_data = np.zeros(
                [1440, self.input_dim]
            )  # hard coding for 1 day, 14 features & time
            timeflag = np.ones(1440)
            for i in range(len(region_data)):
                timestamp = region_data[i][1]
                date_, time_ = timestamp.split(" ")
                data = region_data[i][2:].astype(float)
                data = np.nan_to_num(data, nan=0)

                hh = int(time_[:2])
                mm = int(time_[-2:])
                idx = hh * 60 + mm - 1

                weather_data[idx, 0] = idx
                weather_data[idx, 1:] = data
                timeflag[idx] = 0

            # interpolation for missing data
            idx = np.where(timeflag == 1)[0]
            indices, temp = [], []
            if len(idx) == 1:
                indices.append(idx)
            else:
                diff = np.diff(idx)
                for i in range(len(diff)):
                    temp.append(idx[i].tolist())
                    if diff[i] == 1:
                        temp.append(idx[i + 1])
                    else:
                        indices.append(np.unique(temp).tolist())
                        temp = []
                if len(temp) > 0:  # add the last block
                    indices.append(np.unique(temp).tolist())
                    temp = []

            for n in range(len(indices)):
                idx = indices[n]
                maxV, minV = np.max(idx).astype(int), np.min(idx).astype(int)
                if minV > 0:
                    prev = weather_data[minV - 1, :]
                else:
                    prev = None
                if maxV < 1439:
                    post = weather_data[maxV + 1, :]
                else:
                    post = prev
                if prev is None:
                    prev = post

                nsteps = len(idx)
                for i in range(nsteps):
                    weather_data[i + minV] = (nsteps - i) * prev / (nsteps + 1) + (
                        i + 1
                    ) * post / (nsteps + 1)
            all_data.append(weather_data)
            
        all_data_concatenated = np.concatenate(all_data, axis=0)
        
        self.global_mean = np.mean(all_data_concatenated, axis=0)
        self.global_std = np.std(all_data_concatenated, axis=0)
        
        self.weather_scaler = StandardScaler()
        self.weather_scaler.mean_ = self.global_mean[1:]
        self.weather_scaler.scale_ = self.global_std[1:]

        
    def __len__(self):
        return len(self.aws_list)

    def __getitem__(self, idx):
        aws_path = self.aws_list[idx]
        asos_path = self.asos_list[idx]
        efilepath = self.elist[idx]

        weather_datapath = asos_path.replace(".csv", ".npy")
        weather_datapath = weather_datapath.replace("ASOS", "Weather")
        weather_datapath = weather_datapath.replace("asos", "weather_%d" % self.rID)
        if os.path.isfile(weather_datapath):
            weather_data = np.load(weather_datapath)
        else:
            ############## weather data loading  #################
            # Loading: weather data for all regions (thanks to chy)
            aws_csv = pd.read_csv(aws_path, encoding="CP949")
            aws_csv = aws_csv.get(self.tags1)
            asos_csv = pd.read_csv(asos_path, encoding="CP949")
            asos_csv = asos_csv.get(self.tags2)
            asos_csv.iloc[:, 4] = asos_csv.iloc[:, 4].diff()
            asos_csv = asos_csv.rename(columns={self.tags2[4]: self.tags1[4]})
            csv = pd.concat([aws_csv, asos_csv])

            csv = csv.drop(["지점명"], axis=1)
            groups = csv.groupby(csv.columns[0])
            weather_by_region = {}
            for i in groups:
                if weather_by_region.get(i[0]) is not None:
                    weather_by_region[i[0]].append(list(i))
                else:
                    weather_by_region[i[0]] = list(i)

            # Choose region & Time alignment
            rid = self.rID
            region_data = weather_by_region[rid]
            region_data = region_data[1].values
            weather_data = np.zeros(
                [1440, self.input_dim]
            )  # hard coding for 1 day, 14 features & time
            timeflag = np.ones(1440)
            for i in range(len(region_data)):
                timestamp = region_data[i][1]
                date_, time_ = timestamp.split(" ")
                data = region_data[i][2:].astype(float)
                data = np.nan_to_num(data, nan=0)

                hh = int(time_[:2])
                mm = int(time_[-2:])
                idx = hh * 60 + mm - 1

                weather_data[idx, 0] = idx
                weather_data[idx, 1:] = data
                timeflag[idx] = 0

            # interpolation for missing data
            idx = np.where(timeflag == 1)[0]
            indices, temp = [], []
            if len(idx) == 1:
                indices.append(idx)
            else:
                diff = np.diff(idx)
                for i in range(len(diff)):
                    temp.append(idx[i].tolist())
                    if diff[i] == 1:
                        temp.append(idx[i + 1])
                    else:
                        indices.append(np.unique(temp).tolist())
                        temp = []
                if len(temp) > 0:  # add the last block
                    indices.append(np.unique(temp).tolist())
                    temp = []

            for n in range(len(indices)):
                idx = indices[n]
                maxV, minV = np.max(idx).astype(int), np.min(idx).astype(int)
                if minV > 0:
                    prev = weather_data[minV - 1, :]
                else:
                    prev = None
                if maxV < 1439:
                    post = weather_data[maxV + 1, :]
                else:
                    post = prev
                if prev is None:
                    prev = post

                nsteps = len(idx)
                for i in range(nsteps):
                    weather_data[i + minV] = (nsteps - i) * prev / (nsteps + 1) + (
                        i + 1
                    ) * post / (nsteps + 1)

            dirname = os.path.dirname(weather_datapath)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            np.save(weather_datapath, weather_data)

        weather_data[:, 1:] = self.weather_scaler.transform(weather_data[:, 1:])
        
        insolation = np.array([insolation_aprox(t, n=4, I_max=1) for t in weather_data[:, 0]]) # (sin 기반 조정)
        weather_data[:, 0] = insolation
        weather_data = torch.tensor(weather_data)

        efile_npy = efilepath.replace(".xlsx", ".npy")
        if os.path.isfile(efile_npy):
            power_data = np.load(efile_npy)
        else:
            ############## Photovoltaic data loading  #################
            # Loading: power generation data written by chy
            xlsx = pd.read_excel(efilepath, engine="openpyxl", skiprows=range(3))
            xlsx = xlsx.iloc[:-1, :]  # row remove
            power = xlsx.to_numpy()
            power = pd.DataFrame(power, columns=["Datetime", "Power"])
            power_data = power.to_numpy()
            power_data = power_data[:, 1].astype(float)

            np.save(efile_npy, power_data)
            
            data = pd.read_html(efilepath, header=[0, 1])[0]
            data.columns = [' '.join(col).strip() for col in data.columns.values]

            # Extracting datetime using regex
            pattern = r'\d{4}-\d{2}-\d{2} \d{2}'
            data['datetime'] = data.iloc[:, 0].astype(str).apply(lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else None)
            data['irradiance'] = data['일사량(W/㎡) 경사']
            data = data.dropna(subset=['datetime'])
            data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H')

            # Assuming start and end dates of the month
            start_date = data['datetime'].min().strftime('%Y-%m-%d')
            end_date = (data['datetime'].max() + timedelta(days=1)).strftime('%Y-%m-%d')

            # Creating a full range of hourly timestamps
            full_range = pd.date_range(start=start_date, end=end_date, freq='H')
            full_range_df = pd.DataFrame(full_range, columns=['datetime'])

            # Merging and filling missing data
            merged_data = pd.merge(full_range_df, data, left_on='datetime', right_on='datetime', how='left')
            merged_data['irradiance'] = merged_data['irradiance'].fillna(0)

            # Extracting the solar irradiance values
            isolation_data = merged_data['irradiance'].values

        power_data = torch.tensor(power_data)
        
        # isolation
        if 'samcheck' in self.isolation_list[0]:
            date = list(self.grouped_data.groups)[idx]
            daily_data = self.grouped_data.get_group(date)

            # zero padding
            if len(daily_data) < 24:
                padded_data = daily_data.reindex(pd.date_range(start=date, periods=24, freq='H'), fill_value=0)
                daily_values = padded_data['SOL_RAD_LEVEL'].values
            else:
                daily_values = daily_data['SOL_RAD_LEVEL'].values
            isolation_data = torch.tensor(daily_values, dtype=torch.float32)
        elif 'PreSchool' in self.isolation_list:
            base_date = datetime(year=2022, month=1, day=1)  # Replace 2022 with the actual year
            specific_date = base_date + pd.Timedelta(days=idx - 1)
            month = specific_date.month
            day = specific_date.day

            # Access the file for the identified month
            monthly_iso_path = self.isolation_list[month - 1]  # Assuming list is 0-indexed for January

            # File path for the processed numpy file for the specific day
            iso_npy_path = monthly_iso_path.replace(".xlsx", f"_day{idx}.npy")

            if os.path.isfile(iso_npy_path):
                iso_data = np.load(iso_npy_path)
            else:
                # Read and process the solar irradiance data from the Excel file
                solar_irradiance = pd.read_html(monthly_iso_path, header=[0, 1])[0]
                solar_irradiance.columns = [' '.join(col).strip() for col in solar_irradiance.columns.values]

                # Extracting datetime using regex
                pattern = r'\d{4}-\d{2}-\d{2} \d{2}'
                solar_irradiance['datetime'] = solar_irradiance.iloc[:, 0].astype(str).apply(lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else None)
                solar_irradiance['irradiance'] = solar_irradiance['일사량(W/㎡) 경사']
                solar_irradiance = solar_irradiance.dropna(subset=['datetime'])
                solar_irradiance['datetime'] = pd.to_datetime(solar_irradiance['datetime'], format='%Y-%m-%d %H')

                # Filter out data for the specific day
                day_data = solar_irradiance[solar_irradiance['datetime'].dt.day == day]

                # Creating a full range of hourly timestamps for the day
                start_datetime = datetime(specific_date.year, specific_date.month, specific_date.day)
                end_datetime = start_datetime + pd.Timedelta(days=1)
                full_range = pd.date_range(start=start_datetime, end=end_datetime, freq='H', closed='left')
                full_range_df = pd.DataFrame(full_range, columns=['datetime'])

                # Merging and filling missing data
                merged_data = pd.merge(full_range_df, day_data, left_on='datetime', right_on='datetime', how='left')
                merged_data['irradiance'] = merged_data['irradiance'].fillna(0)

                # Extracting the solar irradiance values
                iso_data = merged_data['irradiance'].values
                np.save(iso_npy_path, iso_data)

                iso_data = torch.tensor(iso_data, dtype=torch.float32)

        #print(power_data.shape)
        #weather_data = weather_data[self.kernel_range[0]*60:self.kernel_range[1]*60+1, :]
        #power_data = power_data[self.kernel_range[0]:self.kernel_range[1]]
        #print(power_data.shape)
        return weather_data, isolation_data, power_data
