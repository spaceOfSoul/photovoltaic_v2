import os
import torch
import numpy as np
from scipy import integrate
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from Visualizer.SeriesDecomp import series_decomp  

class PlotGenerator:
    def __init__(self, image_dir, days_per_month, start_month):
        self.image_dir = image_dir
        self.days_per_month = days_per_month
        self.start_month = start_month

    def plot_monthly(self, y_true_chunks, result_chunks):
        for i in range(len(self.days_per_month)):
            fig, axes = plt.subplots(4, 1, figsize=(30, 40))
            y = np.concatenate(y_true_chunks[i])
            pred = np.concatenate(result_chunks[i]).squeeze()
            
            # For the time axis
            hours = np.arange(1, len(y) + 1)

            # Decompose data
            period = 24 #  24 hours
            seasonal = 707 # (29.5*24)-1 # 707 hours
            trend_y, seasonal_y, resid_y = series_decomp(y, period, seasonal)
            trend_pred, seasonal_pred, resid_pred = series_decomp(pred, period, seasonal)
            # fontsize
            fs_legend = 20
            fs_title = 25
            fs_xlabel = 20
            fs_ylabel = 20
            fs_xnum = 20
            fs_ynum = 20
            lw_label = 2
            lw_pred = 3
            for ax in axes:
                ax.xaxis.set_tick_params(labelsize=fs_xnum)
                ax.yaxis.set_tick_params(labelsize=fs_ynum)
            
            # Plot original
            axes[0].plot(hours, y, label='Ground Truth', linewidth=lw_label)
            axes[0].plot(hours, pred, label='Predicted', linewidth=lw_pred)
            axes[0].legend(loc='upper left', fontsize=fs_legend)
            axes[0].set_title(f'Month {i+self.start_month} - Original (Validation)', fontsize=fs_title)
            axes[0].set_xlabel('hours', fontsize=fs_xlabel)
            axes[0].set_ylabel('kW/h', fontsize=fs_ylabel)
        
            # Plot trend
            axes[1].plot(hours, trend_y, label='Ground Truth', linewidth=lw_label)
            axes[1].plot(hours, trend_pred, label='Predicted', linewidth=lw_pred)
            axes[1].legend(loc='upper left', fontsize=fs_legend)
            axes[1].set_title(f'Month {i+self.start_month} - Trend', fontsize=fs_title)
            axes[1].set_xlabel('hours', fontsize=fs_xlabel)
            axes[1].set_ylabel('kW/h', fontsize=fs_ylabel)
            
            # Plot seasonal
            axes[2].plot(hours, seasonal_y, label='Ground Truth', linewidth=lw_label)
            axes[2].plot(hours, seasonal_pred, label='Predicted', linewidth=lw_pred)
            axes[2].legend(loc='upper left', fontsize=fs_legend)
            axes[2].set_title(f'Month {i+self.start_month} - Seasonal', fontsize=fs_title)
            axes[2].set_xlabel('hours', fontsize=fs_xlabel)
            axes[2].set_ylabel('kW/h', fontsize=fs_ylabel)

            # Plot residual
            axes[3].plot(hours, resid_y, label='Ground Truth', linewidth=lw_label)
            axes[3].plot(hours, resid_pred, label='Predicted', linewidth=lw_pred)
            axes[3].legend(loc='upper left', fontsize=fs_legend)
            axes[3].set_title(f'Month {i+self.start_month} - Residual', fontsize=fs_title)
            axes[3].set_xlabel('hours', fontsize=fs_xlabel)
            axes[3].set_ylabel('kW/h', fontsize=fs_ylabel)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.image_dir, f"month_{i+self.start_month}.png"))
            plt.close() 

    def plot_annual(self, y, pred):
        fig, axes = plt.subplots(4, 1, figsize=(30, 40))  # Subplot count increased to 4
        # fig.suptitle("2022.Jan ~ Aug", fontsize=16)

        # Assuming there are 24 data points for each day
        points_per_day = 24 # 지구의 자전으로 인해 24시간 주기로 낮과 밤이 바뀐다.

        # Compute daily total values
        # y_sum and pred_sum represent the total kW produced per day
        y_sum = np.sum(np.array(y).reshape(-1, points_per_day), axis=1)
        pred_sum = np.sum(np.array(pred).reshape(-1, points_per_day), axis=1)

        logging.info(f"\n\nlen(val_y): {len(y_sum)}\nval_y [kW/day]\n{y_sum}")
        logging.info(f"\n\nlen(val_pred): {len(pred_sum)}\nval_pred [kW/day]\n{pred_sum}")

        # Compute integrals
        integral_y_sum = integrate.simps(y_sum)
        integral_pred_sum = integrate.simps(pred_sum)
        
        # For the time axis
        days = np.arange(1, len(y_sum) + 1)

        # Decompose data
        period = 7 # 7 days
        seasonal = 29 # 29 days
        trend_y, seasonal_y, resid_y = series_decomp(y_sum, period, seasonal)
        trend_pred, seasonal_pred, resid_pred = series_decomp(pred_sum, period, seasonal)
        # trend, seasonal, resid 저장
        logging.info(f"\n\nlen(val_trend_y): {len(trend_y)}\nval_trend_y [kW/day]\n{trend_y}")
        logging.info(f"\n\nlen(val_trend_pred): {len(trend_pred)}\nval_trend_pred [kW/day]\n{trend_pred}")
        logging.info(f"\n\nlen(val_seasonal_y): {len(seasonal_y)}\nval_seasonal_y [kW/day]\n{seasonal_y}")
        logging.info(f"\n\nlen(val_seasonal_pred): {len(seasonal_pred)}\nval_seasonal_pred [kW/day]\n{seasonal_pred}")        
        logging.info(f"\n\nlen(val_resid_y): {len(resid_y)}\nval_residual_y [kW/day]\n{resid_y}")
        logging.info(f"\n\nlen(val_resid_pred): {len(resid_pred)}\nval_residual_pred [kW/day]\n{resid_pred}\n")

        # original, trend, seasonal, resid의 MSE 계산
        criterion = torch.nn.MSELoss()        
        ori_mse = criterion(torch.tensor(y_sum), torch.tensor(pred_sum))
        trend_mse = criterion(torch.tensor(trend_y), torch.tensor(trend_pred))
        seasonal_mse = criterion(torch.tensor(seasonal_y), torch.tensor(seasonal_pred))
        resid_mse = criterion(torch.tensor(resid_y), torch.tensor(resid_pred))

          # 계산한 MSE 로깅
        logging.info(f"\n\nMSE for val original: {(ori_mse.item()):.4f} [(kW/day)^2]")
        logging.info(f"MSE for val trend: {(trend_mse.item()):.4f} [(kW/day)^2]")
        logging.info(f"MSE for val seasonal: {(seasonal_mse.item()):.4f} [(kW/day)^2]")
        logging.info(f"MSE for val resid: {(resid_mse.item()):.4f} [(kW/day)^2]")

        # fontsize
        fs_legend = 35
        fs_title = 35
        fs_xlabel = 30
        fs_ylabel = 30
        fs_xnum = 30
        fs_ynum = 30
        lw_label = 3
        lw_pred = 4
        for ax in axes:
            ax.xaxis.set_tick_params(labelsize=fs_xnum)
            ax.yaxis.set_tick_params(labelsize=fs_ynum)
        
        # Plot original
        axes[0].plot(days, y_sum, label='Ground Truth', linewidth=lw_label)
        axes[0].plot(days, pred_sum, label='Predicted', linewidth=lw_pred)
        axes[0].legend(loc='upper left', fontsize=fs_legend)
        # Add text to the title
        title_text = (f'Daily PV Power (Validation)\n'
              f'Ground Truth Sum: {integral_y_sum:.2f} [kW / {len(self.days_per_month)} months]\n'
              f'Predicted Sum: {integral_pred_sum:.2f} [kW / {len(self.days_per_month)} months]')
        logging.info(f'\nVal y Sum: {integral_y_sum:.2f} [kW / {len(self.days_per_month)} months]\nVal Pred Sum: {integral_pred_sum:.2f} [kW / {len(self.days_per_month)} months]\n')

        axes[0].set_title(title_text, fontsize=fs_title)
        axes[0].set_ylabel('kW/day', fontsize=fs_xlabel)
        axes[0].set_xlabel('days', fontsize=fs_ylabel)

        # Plot trend
        axes[1].plot(days, trend_y, label='Ground Truth', linewidth=lw_label)
        axes[1].plot(days, trend_pred, label='Predicted', linewidth=lw_pred)
        axes[1].legend(loc='upper left', fontsize=fs_legend)
        axes[1].set_title('Trend', fontsize=fs_title)
        axes[1].set_xlabel('days', fontsize=fs_xlabel)
        axes[1].set_ylabel('kW/day', fontsize=fs_ylabel)
        
        # Plot seasonal
        axes[2].plot(days, seasonal_y, label='Ground Truth', linewidth=lw_label)
        axes[2].plot(days, seasonal_pred, label='Predicted', linewidth=lw_pred)
        axes[2].legend(loc='upper left', fontsize=fs_legend)
        axes[2].set_title('Seasonal', fontsize=fs_title)
        axes[2].set_xlabel('days', fontsize=fs_xlabel)
        axes[2].set_ylabel('kW/day', fontsize=fs_ylabel)

        # Plot residual
        axes[3].plot(days, resid_y, label='Ground Truth', linewidth=lw_label)
        axes[3].plot(days, resid_pred, label='Predicted', linewidth=lw_pred)
        axes[3].legend(loc='upper left', fontsize=fs_legend)
        axes[3].set_title('Residual', fontsize=fs_title)
        axes[3].set_xlabel('days', fontsize=fs_xlabel)
        axes[3].set_ylabel('kW/day', fontsize=fs_ylabel)
                      
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "annual_pattern.png"))
        plt.close()
        
    def plot_monthly_loss(self, y, pred):

        # Assuming there are 24 data points for each day
        points_per_day = 24 # 지구의 자전으로 인해 24시간 주기로 낮과 밤이 바뀐다.

        # Compute daily total values
        # y_sum and pred_sum represent the total kW produced per day
        y_sum = np.sum(np.array(y).reshape(-1, points_per_day), axis=1)
        pred_sum = np.sum(np.array(pred).reshape(-1, points_per_day), axis=1)

        # Initialize the monthly loss array
        monthly_loss = np.zeros(len(self.days_per_month))

        # Calculate the monthly loss
        start_day = 0
        for i, days in enumerate(self.days_per_month):
            end_day = start_day + days
            monthly_loss[i] = np.sum((y_sum[start_day:end_day] - pred_sum[start_day:end_day])**2)/days
            logging.info(f"{i+self.start_month} month MSE Val Loss: {monthly_loss[i]:.2f} [(kW/day)^2]")
            start_day = end_day
        
        # Plot the monthly loss
        fig, ax = plt.subplots(figsize=(10, 6))
        months = np.arange(1, len(self.days_per_month)+1)
        ax.bar(months, monthly_loss)
        ax.set_xticks(months)
        ax.set_xticklabels(['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        ax.set_xlabel('Months')
        ax.set_ylabel('[(kW/day)^2]')
        ax.set_title('Monthly MSE Val Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "monthly_loss.png"))
        plt.close()
