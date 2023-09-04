import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau
import torch

def correlations(pv_power: torch.Tensor, batch_mean: torch.Tensor, plot: bool=True):
    """
    Compute and optionally plot Pearson and Kendall's Tau Correlation Coefficients.
    """
    """    
    # 데이터의 크기
    n = 24
    # pv_power 데이터 세트를 임의로 생성하여 tensor로 변환
    pv_power = torch.linspace(1, 24, n).reshape(-1, 1) # torch.Size([24, 1])
    # 피어슨 상관 계수가 1인 데이터
    feature0 = pv_power.clone()
    # 피어슨 상관 계수가 0인 데이터. 랜덤 값 생성.
    torch.manual_seed(0)
    feature1 = torch.randn((n, 1)) # Normally distributed data with mean 0 and variance 1
    # 피어슨 상관 계수가 -1인 데이터
    feature2 = -pv_power.clone()
    # tensor로 합칩니다
    batch_mean = torch.cat([feature0, feature1, feature2], dim=1) # torch.Size([24, 3])
     """
    # print(f"pv_power: {pv_power}\nbatch_mean: {batch_mean}")
     
    # Convert tensors to numpy if they are torch.Tensor objects
    if isinstance(pv_power, torch.Tensor):
        pv_power_np = pv_power.cpu().numpy().squeeze()
    else:
        pv_power_np = pv_power

    if isinstance(batch_mean, torch.Tensor):
        batch_mean_np = batch_mean.cpu().numpy()
    else:
        batch_mean_np = batch_mean
    
    # print(f"pv_power_np: {pv_power_np.shape}\nbatch_mean_np: {batch_mean_np.shape}")
    
    # Compute Pearson and Kendall's Tau correlation between pv_power and each feature of batch_mean
    pv_pearson_correlations = [pearsonr(pv_power_np, batch_mean_np[:, i])[0] for i in range(batch_mean_np.shape[1])]
    pv_kendall_correlations = [kendalltau(pv_power_np, batch_mean_np[:, i]).correlation for i in range(batch_mean_np.shape[1])]

    # Compute Pearson and Kendall's Tau correlation between each feature of batch_mean and every other feature
    feature_pearson_correlations = np.corrcoef(batch_mean_np, rowvar=False)
    feature_kendall_correlations = np.array([[kendalltau(batch_mean_np[:, i], batch_mean_np[:, j]).correlation 
                                              for j in range(batch_mean_np.shape[1])] 
                                              for i in range(batch_mean_np.shape[1])])

    # Remove the upper triangle for clarity
    for i in range(feature_pearson_correlations.shape[0]):
        for j in range(i+1, feature_pearson_correlations.shape[1]):
            feature_pearson_correlations[i, j] = np.nan

    for i in range(feature_kendall_correlations.shape[0]):
        for j in range(i+1, feature_kendall_correlations.shape[1]):
            feature_kendall_correlations[i, j] = np.nan
            
    # Features list
    # features = ["x0", "x1", "x2"]
    # features = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    features = ["Time (min)", "Temp. (°C)", "Rainfall (1-min, mm)", "Wind Dir. (deg)", "Wind Spd. (m/s)", "Station Pres. (hPa)", "Sea Level Pres. (hPa)", "Humidity (%)"]
    # features = ["Time (min)", "Temp. (°C)", "Rainfall (1-min, mm)", "Wind Dir. (deg)", "Wind Spd. (m/s)", "Station Pres. (hPa)", "Humidity (%)"]

    # Plot if the plot flag is True
    if plot:
        # Figure 1: Plotting pv_power correlations for Pearson
        plt.figure(figsize=(6, 4))
        bars = plt.bar(range(len(pv_pearson_correlations)), pv_pearson_correlations)
        plt.title("Pearson Correlation between pv_power and each feature")
        plt.xlabel("Features")
        plt.ylabel("Correlation Coefficient")
        plt.xticks(range(len(features)), labels=features, rotation=45, ha='right')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

        # Figure 1.1: Plotting pv_power correlations for Kendall's Tau
        plt.figure(figsize=(6, 4))
        bars = plt.bar(range(len(pv_kendall_correlations)), pv_kendall_correlations)
        plt.title("Kendall's Tau Correlation between pv_power and each feature")
        plt.xlabel("Features")
        plt.ylabel("Correlation Coefficient")
        plt.xticks(range(len(features)), labels=features, rotation=45, ha='right')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

        # Figure 2: Plotting heatmap of feature correlations for Pearson
        cax = plt.matshow(feature_pearson_correlations, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(cax)
        plt.title("Pearson Correlation between features")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.xticks(range(len(features)), labels=features, rotation=90, ha='right')
        plt.yticks(range(len(features)), labels=features)
        rows, cols = feature_pearson_correlations.shape
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(feature_pearson_correlations[i, j]):
                    plt.text(j, i, round(feature_pearson_correlations[i, j], 2), ha='center', va='center', color='k')

        # Figure 2.1: Plotting heatmap of feature correlations for Kendall's Tau
        cax = plt.matshow(feature_kendall_correlations, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(cax)
        plt.title("Kendall's Tau Correlation between features")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.xticks(range(len(features)), labels=features, rotation=90, ha='right')
        plt.yticks(range(len(features)), labels=features)
        rows, cols = feature_kendall_correlations.shape
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(feature_kendall_correlations[i, j]):
                    plt.text(j, i, round(feature_kendall_correlations[i, j], 2), ha='center', va='center', color='k')
        
        # dot and font size (Fig 3 and 4)
        s = 2
        font_size = 7
        
        # Figure 3: Plotting scatter plots between pv_power and each feature
        plt.figure(figsize=(5, 15))  # figsize를 수정
        for i in range(batch_mean_np.shape[1]):
            plt.subplot(batch_mean_np.shape[1], 1, i+1)  # subplot 인자 수정
            plt.scatter(batch_mean_np[:, i], pv_power_np, s=s)
            correlation_value = np.corrcoef(pv_power_np, batch_mean_np[:, i])[0, 1]
            plt.title(f"{features[i]} vs pv_power", fontsize=font_size)
            plt.ylabel("pv_power", fontsize=font_size)
            plt.xlabel(features[i], fontsize=font_size)
            plt.annotate(f"PCC = {correlation_value:.2f}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize=font_size-2)

        # Figure 4: Plotting scatter plots for feature correlations
        total_features = batch_mean_np.shape[1]
        total_subplots = total_features * (total_features - 1) // 2  # half of the total pairs
        cols = total_features
        rows = total_features - 1

        plt.figure(figsize=(15, 10))
        subplot_count = 1
        for i in range(total_features):
            for j in range(i+1, total_features):  # changing this loop condition
                plt.subplot(rows, cols, subplot_count)
                plt.scatter(batch_mean_np[:, i], batch_mean_np[:, j], s=s)
                correlation_value = np.corrcoef(batch_mean_np[:, i], batch_mean_np[:, j])[0, 1]
                plt.title(f"{features[i]} vs {features[j]}", fontsize=font_size)
                plt.xlabel(features[i], fontsize=font_size)
                plt.ylabel(features[j], fontsize=font_size)
                plt.annotate(f"PCC: {correlation_value:.2f}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize=font_size-2)
                subplot_count += 1
        
        # Figure 5: Plotting histograms for each feature to see the distribution
        plt.figure(figsize=(15, 10))  # figsize 조절
        for i in range(batch_mean_np.shape[1]):
            plt.subplot(2, 4, i+1)  # 2x4 subplot으로 조절
            plt.hist(batch_mean_np[:, i], bins=30, edgecolor='k')  # Histogram
            plt.title(f"Distribution of {features[i]}")
            plt.xlabel(features[i])
            plt.ylabel("Frequency")
    
        plt.tight_layout()
        plt.show()

    return pv_pearson_correlations, feature_pearson_correlations, pv_kendall_correlations, feature_kendall_correlations
    
# Example usage:
# pv_correlations, feature_correlations = pearson_corr(None, None, plot=True)
