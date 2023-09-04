import numpy as np
import pandas as pd
import csv
import os

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utility import insolation_aprox

class WPD(Dataset):
    def __init__(self,aws_list,asos_list,energy_list,region_ID,input_dim=8,datapath="../dataset/",):
        self.aws_list = aws_list  # all files for weather info
        self.asos_list = asos_list
        self.elist = energy_list  # all files for power gener.
        
        self.rID = region_ID
        self.input_dim = input_dim

        self.tags1 = ["지점","지점명","일시","기온(°C)","1분 강수량(mm)","풍향(deg)","풍속(m/s)","현지기압(hPa)","해면기압(hPa)","습도(%)",]
        self.tags2 = ["지점","지점명","일시","기온(°C)","누적강수량(mm)","풍향(deg)","풍속(m/s)","현지기압(hPa)","해면기압(hPa)","습도(%)",]

        if not os.path.isdir(datapath):
            os.makedirs(datapath)
            
        # 연간 모든 데이터에 대해서 일단 한번 로드
        # 평균, 표준편차를 구하기 위함
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

        # 평균과 표준편차 계산
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

        weather_data[:, 1:] = self.weather_scaler.transform(weather_data[:, 1:]) # z 정규화
        
        insolation = np.array([insolation_aprox(t, n=4) for t in weather_data[:, 0]]) # 일조량(sin 기반)
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

        power_data = torch.tensor(power_data)

        return weather_data, power_data
