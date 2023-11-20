import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class Intensity(Dataset):
    def __init__(self, asos_list, region_ID, input_dim=1, datapath="../dataset/"):
        self.asos_list = asos_list  # Weather info files
        self.rID = region_ID
        self.input_dim = input_dim

        if not os.path.isdir(datapath):
            os.makedirs(datapath)
            
    def __len__(self):
        return len(self.asos_list)

    def __getitem__(self, idx):
        asos_path = self.asos_list[idx]
        weather_datapath = asos_path.replace(".csv", ".npy")
        weather_datapath = weather_datapath.replace("ASOS", "Insolation")

        if os.path.isfile(weather_datapath):
            insolation_data = np.load(weather_datapath, allow_pickle=True)
        else:
            asos_csv = pd.read_csv(asos_path, encoding="CP949")
            csv = asos_csv.drop(["지점명"], axis=1)
            insolation_data = csv.loc[csv["지점"] == self.rID, "일사(MJ/m^2)"].values
            print(insolation_data.shape)
            dirname = os.path.dirname(weather_datapath)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            np.save(weather_datapath, insolation_data)

        # zero padding
        if len(insolation_data) < 1440:
            insolation_data = np.pad(insolation_data, (0, 1440 - len(insolation_data)), 'constant')

        # 변화량 계산
        differences = np.diff(insolation_data)
        
        # 단위 변환
        differences *= 1000
        differences = differences / 60

        # 60분 간격으로 샘플링
        sampled_data = differences[::60]

        insolation_data = sampled_data.astype(np.float32)
        insolation_data = torch.tensor(insolation_data)

        return insolation_data

# for test
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from utility import list_up_weather
    
    dirs = list_up_weather("./dataset/ASOS/","20220101", "20221231")
    dataset = Intensity(dirs, 105)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, data in enumerate(data_loader):
        print(data.shape)
        if i == 29:
            plt.plot(data.numpy()[0])
            plt.title("Insolation Data Visualization")
            plt.xlabel("Time")
            plt.ylabel("Insolation")
            plt.show()
    
    print(i)